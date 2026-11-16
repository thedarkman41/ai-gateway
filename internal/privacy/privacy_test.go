package privacy

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestScrub(t *testing.T) {
	p := &Policy{ExtraSecrets: []string{`ACCT-\d{6}`}}
	if err := p.prepare(); err != nil {
		t.Fatal(err)
	}
	in := strings.Join([]string{
		`GROQ = "gsk_abcdefghijklmnopqrstuvwxyz0123456789"`,
		`client = OpenAI(api_key="sk-proj-abcdefghijklmnopqrstuvwx")`,
		`db = connect("postgres://bill:hunter2secret@db.internal:5432/app")`,
		`headers = {"Authorization": "Bearer abcdef0123456789abcdef"}`,
		`password = "correct horse battery"`,
		`DB_PASSWORD=s3cr3t-value-here`,
		`export GITHUB_TOKEN=ghp_abcdefghijklmnopqrstuvwxyz012345`,
		"-----BEGIN OPENSSH PRIVATE KEY-----\nb3BlbnNzaC1rZXk\n-----END OPENSSH PRIVATE KEY-----",
		`jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.abcdefghijklmnop"`,
		`account = "ACCT-123456"`,
		`mine = "my-own-gateway-key-value"`,
		// None of these is a secret:
		`token_count = len(tokens)`,
		`password = input("Password: ")`,
		`url = "https://example.com/path?x=1"`,
		`def is_prime(n): return n > 1  # sk- is a prefix, not a key`,
	}, "\n")
	out, found := p.Scrub(in, []string{"my-own-gateway-key-value"})
	for _, leak := range []string{"gsk_abc", "sk-proj-abc", "hunter2secret", "abcdef0123456789abcdef", "correct horse", "s3cr3t-value", "ghp_abc", "b3BlbnNzaC1rZXk", "eyJhbGci", "ACCT-123456", "my-own-gateway-key-value"} {
		if strings.Contains(out, leak) {
			t.Errorf("leaked %q:\n%s", leak, out)
		}
	}
	for _, keep := range []string{`token_count = len(tokens)`, `password = input("Password: ")`, `url = "https://example.com/path?x=1"`, `# sk- is a prefix, not a key`,
		`postgres://bill:[REDACTED:password]@db.internal:5432/app`, `"Authorization": "Bearer [REDACTED:token]"`, `password = "[REDACTED:secret]"`, `DB_PASSWORD=[REDACTED:secret]`,
		`OpenAI(api_key="[REDACTED:api-key]")`, `export GITHUB_TOKEN=[REDACTED:api-key]`} {
		if !strings.Contains(out, keep) {
			t.Errorf("lost %q:\n%s", keep, out)
		}
	}
	if s := Summary(found); !strings.Contains(s, "secrets (") || found["api-key"] < 2 || found["private-key"] != 1 || found["custom"] != 1 || found["gateway-key"] != 1 {
		t.Errorf("found %v (%s)", found, s)
	}
	if out, found := (*Policy)(nil).Scrub("nothing here", nil); out != "nothing here" || Summary(found) != "" {
		t.Errorf("clean text changed: %q %v", out, found)
	}
}

func TestPrivate(t *testing.T) {
	home, _ := os.UserHomeDir()
	p := &Policy{BlockedPaths: []string{"~/work/client-x"}}
	p.prepare()
	file := func(path string) string {
		return "TASK:\nfix it\n\nFILE " + path + " (its full current text, supplied by the harness):\n```\nx = 1\n```"
	}
	for text, want := range map[string]string{
		file(home + "/work/client-x/src/a.py"):       "blocked directory",
		file(home + "/work/client-xyz/a.py"):         "",
		file(home + "/projects/app/.env"):            "credentials file",
		file(home + "/projects/app/.env.production"): "credentials file",
		file(home + "/.ssh/config"):                  "credentials file",
		file("/Users/x/My Projects/app/deploy.pem"):  "credentials file",
		file(home + "/projects/app/environment.py"):  "",
		file(home + "/projects/app/keyboard.py"):     "",
		"look at " + home + "/work/client-x/notes":   "names a blocked directory",
		"an ordinary question":                       "",
	} {
		if got := p.Private(text); (want == "") != (got == "") || !strings.Contains(got, want) {
			t.Errorf("Private(%q) = %q, want %q", text[len(text)-min(60, len(text)):], got, want)
		}
	}
	// With an allow list, only files under it may leave.
	p = &Policy{AllowedPaths: []string{"~/open"}}
	p.prepare()
	if p.Private(file(home+"/open/lib/a.py")) != "" || !strings.Contains(p.Private(file(home+"/other/a.py")), "outside the directories allowed") {
		t.Error("allow list not applied")
	}
}

func TestStore(t *testing.T) {
	path := filepath.Join(t.TempDir(), "privacy.json")
	s := Open(path)
	if p, err := s.Policy(); err != nil || len(p.BlockedPaths) != 0 {
		t.Fatalf("no file: %v %v", p, err)
	}
	if err := s.Edit(func(p *Policy) { p.BlockedPaths = append(p.BlockedPaths, "/srv/private") }); err != nil {
		t.Fatal(err)
	}
	if p, err := s.Policy(); err != nil || len(p.BlockedPaths) != 1 || p.BlockedPaths[0] != "/srv/private" {
		t.Errorf("after edit: %v %v", p, err)
	}
	// A file that cannot be read as written is an error, not silently "no rules".
	os.WriteFile(path, []byte(`{"blocked_paths": [`), 0o600)
	os.Chtimes(path, timeNow().Add(5e9), timeNow().Add(5e9))
	if _, err := s.Policy(); err == nil {
		t.Error("a broken policy file was accepted")
	}
}

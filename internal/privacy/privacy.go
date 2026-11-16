// Package privacy decides what may leave this machine. Every prompt the gateway sends to a provider
// outside the local network passes through it, whichever tool or endpoint the prompt came in by.
//
// Two rules:
//
//   - Secrets are redacted. API keys, private keys, tokens, passwords and credentials in URLs are
//     replaced with a marker before the text is sent.
//   - Private paths stay home. A prompt that carries a file from a blocked directory, or a file that
//     by its name holds credentials, is not sent outside at all: only local models may see it.
package privacy

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"sync"
	"time"
)

// Policy is the user's part of the rules, stored in <state>/privacy.json.
type Policy struct {
	BlockedPaths []string `json:"blocked_paths"` // files under these never leave the machine
	AllowedPaths []string `json:"allowed_paths"` // if any are listed, only files under these may leave
	ExtraSecrets []string `json:"extra_secrets"` // more regexes whose matches are redacted

	extra []*regexp.Regexp
}

type rule struct {
	kind string
	re   *regexp.Regexp
	repl string // "" = the whole match becomes the marker; else a template where {m} is the marker
}

var rules = []rule{
	{"private-key", regexp.MustCompile(`-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----`), ""},
	{"api-key", regexp.MustCompile(`\b(?:sk-ant-[\w-]{20,}|sk-or-v1-[0-9a-f]{20,}|sk-(?:proj-)?[A-Za-z0-9_-]{20,}|gsk_[A-Za-z0-9]{20,}|AIza[0-9A-Za-z_-]{30,}|AQ\.[0-9A-Za-z_-]{30,}|gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|glpat-[\w-]{20,}|xox[abprs]-[A-Za-z0-9-]{10,}|AKIA[0-9A-Z]{16}|nvapi-[\w-]{20,}|cfat_[\w-]{20,}|hf_[A-Za-z0-9]{30,}|npm_[A-Za-z0-9]{30,})`), ""},
	{"token", regexp.MustCompile(`\beyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}`), ""},
	{"token", regexp.MustCompile(`(?i)(authorization["']?\s*[:=]\s*["']?(?:bearer|token|basic)\s+)[A-Za-z0-9._~+/=-]{16,}`), "${1}{m}"},
	{"password", regexp.MustCompile(`([a-zA-Z][a-zA-Z0-9+.-]*://[^/\s:@"']+:)[^@\s"'/]{3,}(@)`), "${1}{m}${2}"},
	// name = "value" where the name says it is a secret
	{"secret", regexp.MustCompile(`(?i)(\b[\w.-]*(?:password|passwd|secret|token|api[_-]?key|access[_-]?key|private[_-]?key)[\w.-]*["']?\s*[:=]\s*["'])[^"'\n]{8,}(["'])`), "${1}{m}${2}"},
	// NAME=value as in a .env file
	{"secret", regexp.MustCompile(`(?m)^((?:export\s+)?[A-Z0-9_]*(?:PASSWORD|PASSWD|SECRET|TOKEN|API_?KEY|ACCESS_?KEY|PRIVATE_?KEY)[A-Z0-9_]*\s*=\s*)[^\s"'#]{8,}[ \t]*$`), "${1}{m}"},
}

// privateName matches files that by their name hold credentials.
var privateName = regexp.MustCompile(`(?i)(?:^|/)(?:\.env(?:\.[\w.-]+)?|\.netrc|\.npmrc|\.pypirc|id_(?:rsa|dsa|ecdsa|ed25519)|credentials(?:\.json)?|keys\.json|secrets?\.(?:json|ya?ml|toml|env)|[^/]*\.(?:pem|p12|pfx|key|keychain(?:-db)?))$|/\.(?:ssh|aws|gnupg|ai-gateway)/`)

// fileRe finds the files Bill the Cat attaches to a hand-off: "FILE <path> (its full current text, ...".
var fileRe = regexp.MustCompile(`(?m)^FILE (.+?) \(its full current text`)

// Scrub redacts secrets in text. known are exact secret values the caller holds (the gateway's own
// API keys); they are redacted wherever they appear. It reports how many of each kind it found.
func (p *Policy) Scrub(text string, known []string) (string, map[string]int) {
	found := map[string]int{}
	for _, k := range known {
		if len(k) >= 8 && strings.Contains(text, k) {
			found["gateway-key"] += strings.Count(text, k)
			text = strings.ReplaceAll(text, k, "[REDACTED:gateway-key]")
		}
	}
	apply := func(kind string, re *regexp.Regexp, repl string) {
		marker := "[REDACTED:" + kind + "]"
		text = re.ReplaceAllStringFunc(text, func(m string) string {
			if strings.Contains(m, "[REDACTED:") { // an earlier rule already dealt with this value
				return m
			}
			found[kind]++
			if repl == "" {
				return marker
			}
			return re.ReplaceAllString(m, strings.ReplaceAll(repl, "{m}", marker))
		})
	}
	for _, r := range rules {
		apply(r.kind, r.re, r.repl)
	}
	if p != nil {
		for _, re := range p.extra {
			apply("custom", re, "")
		}
	}
	return text, found
}

// Private says why text must not leave the machine, or "". It looks at the files attached to the
// text and at any blocked directory named in it.
func (p *Policy) Private(text string) string {
	var blocked, allowed []string
	if p != nil {
		blocked, allowed = p.BlockedPaths, p.AllowedPaths
	}
	under := func(path string, dirs []string) string {
		for _, d := range dirs {
			if path == d || strings.HasPrefix(path, strings.TrimSuffix(d, "/")+"/") {
				return d
			}
		}
		return ""
	}
	for _, m := range fileRe.FindAllStringSubmatch(text, -1) {
		path := expand(m[1])
		if d := under(path, blocked); d != "" {
			return fmt.Sprintf("%s is in a blocked directory (%s)", m[1], d)
		}
		if privateName.MatchString(path) {
			return m[1] + " is a credentials file by its name"
		}
		if len(allowed) > 0 && under(path, allowed) == "" {
			return m[1] + " is outside the directories allowed to leave this machine"
		}
	}
	for _, d := range blocked { // a blocked directory named anywhere, attached file or not
		if strings.Contains(text, strings.TrimSuffix(d, "/")+"/") {
			return "the text names a blocked directory (" + d + ")"
		}
	}
	return ""
}

// Summary describes findings for a log line or a note: "2 secrets (api-key, password)".
func Summary(found map[string]int) string {
	total := 0
	var kinds []string
	for k, n := range found {
		total += n
		kinds = append(kinds, k)
	}
	if total == 0 {
		return ""
	}
	sort.Strings(kinds)
	word := "secrets"
	if total == 1 {
		word = "secret"
	}
	return fmt.Sprintf("%d %s (%s)", total, word, strings.Join(kinds, ", "))
}

func expand(path string) string {
	if home, err := os.UserHomeDir(); err == nil && (path == "~" || strings.HasPrefix(path, "~/")) {
		path = filepath.Join(home, path[1:])
	}
	return filepath.Clean(path)
}

func (p *Policy) prepare() error {
	for i, d := range p.BlockedPaths {
		p.BlockedPaths[i] = expand(d)
	}
	for i, d := range p.AllowedPaths {
		p.AllowedPaths[i] = expand(d)
	}
	p.extra = nil
	for _, s := range p.ExtraSecrets {
		re, err := regexp.Compile(s)
		if err != nil {
			return fmt.Errorf("extra_secrets %q: %w", s, err)
		}
		p.extra = append(p.extra, re)
	}
	return nil
}

// Store is the policy file, reread when it changes so that a running gateway follows an edit.
type Store struct {
	path string
	mu   sync.Mutex
	mod  time.Time
	cur  *Policy
	err  error
}

func Open(path string) *Store { return &Store{path: path, cur: &Policy{}} }

// Policy returns the current rules. If the file cannot be read as written, the error is returned
// with the last good rules: the caller must treat that as "send nothing outside".
func (s *Store) Policy() (*Policy, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.path == "" {
		return s.cur, nil
	}
	info, err := os.Stat(s.path)
	if err != nil { // no file: the built-in rules alone
		s.cur, s.mod, s.err = &Policy{}, time.Time{}, nil
		return s.cur, nil
	}
	if info.ModTime().Equal(s.mod) {
		return s.cur, s.err
	}
	s.mod = info.ModTime()
	data, err := os.ReadFile(s.path)
	var p Policy
	if err == nil {
		err = json.Unmarshal(data, &p)
	}
	if err == nil {
		err = p.prepare()
	}
	if err != nil {
		s.err = fmt.Errorf("%s: %w", s.path, err)
		return s.cur, s.err
	}
	s.cur, s.err = &p, nil
	return s.cur, nil
}

// Edit changes the policy file.
func (s *Store) Edit(change func(*Policy)) error {
	var p Policy
	if data, err := os.ReadFile(s.path); err == nil {
		if err := json.Unmarshal(data, &p); err != nil {
			return fmt.Errorf("%s: %w", s.path, err)
		}
	}
	change(&p)
	if err := p.prepare(); err != nil {
		return err
	}
	data, _ := json.MarshalIndent(p, "", "  ")
	if err := os.MkdirAll(filepath.Dir(s.path), 0o700); err != nil {
		return err
	}
	return os.WriteFile(s.path, append(data, '\n'), 0o600)
}

// Expand is the absolute form of a path as the policy stores it.
func Expand(path string) string {
	if abs, err := filepath.Abs(expand(path)); err == nil {
		return abs
	}
	return expand(path)
}

var timeNow = time.Now

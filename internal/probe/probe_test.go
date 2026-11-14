package probe

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func TestRun(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/chat":
			w.Write([]byte(`{"choices":[{"message":{"content":"pong"}}]}`))
		case "/models":
			w.Write([]byte(`{"data":[{"id":"a"},{"id":"b"}]}`))
		case "/wrong":
			w.Write([]byte(`{"choices":[{"message":{"content":"hello"}}]}`))
		case "/blocked":
			w.WriteHeader(403)
		case "/down":
			w.WriteHeader(502)
		default:
			if strings.HasPrefix(r.URL.Path, "/get/") {
				w.Write([]byte("PONG"))
				return
			}
			w.WriteHeader(404)
		}
	}))
	defer srv.Close()

	f := File{Prompt: "Reply with exactly one word: PONG", Expect: "PONG", Candidates: []Candidate{
		{ID: "ok", Tier: "A", Kind: "openai", URL: srv.URL + "/chat", ModelsURL: srv.URL + "/models"},
		{ID: "wrong", Tier: "A", Kind: "openai", URL: srv.URL + "/wrong"},
		{ID: "gone", Tier: "A", Kind: "openai", URL: srv.URL + "/nope"},
		{ID: "get", Tier: "A", Kind: "get", URL: srv.URL + "/get/{prompt}"},
		{ID: "blocked", Tier: "D", Kind: "reach", URL: srv.URL + "/blocked"},
		{ID: "down", Tier: "D", Kind: "reach", URL: srv.URL + "/down"},
		{ID: "nocli", Tier: "B", Kind: "cli", Bin: "no-such-binary-xyz"},
		{ID: "cli", Tier: "B", Kind: "cli", Bin: "echo", AskArgs: []string{"{prompt}"}},
		{ID: "box", Tier: "C", Kind: "none"},
	}}
	want := map[string]string{"ok": Answered, "wrong": WrongAnswer, "gone": Failed, "get": Answered,
		"blocked": Reachable, "down": Failed, "nocli": Missing, "cli": Answered, "box": Skipped}

	rs := Run(context.Background(), f, Options{Timeout: 5 * time.Second, AskCLI: true})
	if len(rs) != len(want) {
		t.Fatalf("got %d results, want %d", len(rs), len(want))
	}
	for _, r := range rs {
		if r.Status != want[r.ID] {
			t.Errorf("%s: status %q (%s), want %q", r.ID, r.Status, r.Detail, want[r.ID])
		}
		if r.ID == "ok" && len(r.Models) != 2 {
			t.Errorf("ok: models %v, want 2", r.Models)
		}
	}
	if rs[0].Tier != "A" || rs[len(rs)-1].Tier != "D" {
		t.Errorf("results not ordered by tier")
	}
	if md := Markdown(rs, time.Now()); !strings.Contains(md, "| A | ok | answered |") {
		t.Errorf("markdown missing the ok row:\n%s", md)
	}

	// Without AskCLI an installed CLI is only reported as present.
	rs = Run(context.Background(), File{Prompt: "p", Expect: "PONG", Candidates: []Candidate{{ID: "cli", Kind: "cli", Bin: "echo", AskArgs: []string{"{prompt}"}}}}, Options{})
	if rs[0].Status != Reachable {
		t.Errorf("cli without AskCLI: %q, want %q", rs[0].Status, Reachable)
	}
}

func TestLoadCandidates(t *testing.T) {
	f, err := Load("../../providers/candidates.json")
	if err != nil {
		t.Fatal(err)
	}
	seen := map[string]bool{}
	for _, c := range f.Candidates {
		if seen[c.ID] {
			t.Errorf("duplicate id %s", c.ID)
		}
		seen[c.ID] = true
		if !strings.Contains("ABCD", c.Tier) || c.Tier == "" {
			t.Errorf("%s: bad tier %q", c.ID, c.Tier)
		}
	}
}

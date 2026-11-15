package gateway

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/thedarkman41/ai-gateway/internal/mcp"
	"github.com/thedarkman41/ai-gateway/internal/privacy"
	"github.com/thedarkman41/ai-gateway/internal/provider"
)

// fake is a scripted OpenAI-compatible backend.
type fake struct {
	mu    sync.Mutex
	calls []map[string]any // decoded request bodies
	down  bool
}

func (f *fake) server(t *testing.T) *httptest.Server {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/models" {
			w.Write([]byte(`{"data":[{"id":"dead"},{"id":"live"},{"id":"other"}]}`))
			return
		}
		var body map[string]any
		b, _ := io.ReadAll(r.Body)
		json.Unmarshal(b, &body)
		body["auth"] = r.Header.Get("Authorization")
		if m, _ := body["model"].(string); m == "echoauth" {
			http.Error(w, "bad request from "+r.Header.Get("Authorization"), 400)
			return
		}
		f.mu.Lock()
		f.calls = append(f.calls, body)
		down := f.down
		f.mu.Unlock()
		if down {
			http.Error(w, "upstream broke", 502)
			return
		}
		switch m, _ := body["model"].(string); {
		case strings.HasPrefix(m, "keyed"):
			http.Error(w, `{"error":{"message":"Missing API key.","code":"missing_api_key"}}`, 401)
			return
		case strings.HasPrefix(m, "spent"):
			http.Error(w, `Error: Free usage limit reached.`, 400)
			return
		case m == "testcalls":
			w.Write([]byte(`{"choices":[{"message":{"content":"Sure:\n` + "```" + `json\n[[9], [11], 15]\n` + "```" + `"}}]}`))
			return
		case strings.HasPrefix(m, "cutfence"):
			w.Write([]byte(`{"choices":[{"message":{"content":"Here:\n` + "```" + `python\ndef f():\n    pass\n# more"},"finish_reason":"stop"}]}`))
			return
		case strings.HasPrefix(m, "cutlength"):
			w.Write([]byte(`{"choices":[{"message":{"content":"The answer is forty"},"finish_reason":"length"}]}`))
			return
		case strings.HasPrefix(m, "slow"):
			time.Sleep(700 * time.Millisecond)
		case strings.HasPrefix(m, "broken"):
			http.Error(w, "boom", 500)
			return
		case strings.HasPrefix(m, "busy"):
			http.Error(w, `{"error":{"message":"This model is temporarily busy."}}`, 503)
			return
		}
		if body["model"] == "dead" {
			http.Error(w, `{"error":{"message":"Model 'dead' is currently unavailable."}}`, 400)
			return
		}
		msgs := body["messages"].([]any)
		last := msgs[len(msgs)-1].(map[string]any)["content"].(string)
		fmt.Fprintf(w, `{"choices":[{"message":{"content":%q}}]}`, fmt.Sprintf("%v says: %s (%d msgs)", body["model"], last, len(msgs)))
	}))
	t.Cleanup(srv.Close)
	return srv
}

func (f *fake) count() int { f.mu.Lock(); defer f.mu.Unlock(); return len(f.calls) }

func build(t *testing.T, cfg string) *Gateway {
	t.Helper()
	ps, err := provider.Parse([]byte(cfg))
	if err != nil {
		t.Fatal(err)
	}
	return New(ps, OpenLedger(filepath.Join(t.TempDir(), "ledger.json")))
}

func TestAskRoutingAndFallback(t *testing.T) {
	a, b := &fake{}, &fake{}
	sa, sb := a.server(t), b.server(t)
	g := build(t, fmt.Sprintf(`{"providers":[
		{"id":"nocli","tier":"B","kind":"cli","bin":"no-such-binary-xyz","ask_args":["{prompt}"]},
		{"id":"a","tier":"A","kind":"openai","url":"%s/chat","models_url":"%s/models"},
		{"id":"b","tier":"A","kind":"openai","url":"%s/chat","default_model":"live"},
		{"id":"echo","tier":"B","kind":"cli","bin":"echo","ask_args":["cli got: {prompt}"],"model_args":["{model}"]},
		{"id":"sh","tier":"B","kind":"cli","bin":"sh","ask_args":["-c","echo $(pwd) $TRUST # {prompt}"],"env":{"TRUST":"trusted"}},
		{"id":"off","tier":"B","kind":"cli","bin":"echo","ask_args":["{prompt}"],"disabled":true}
	]}`, sa.URL, sa.URL, sb.URL))
	ctx := context.Background()

	// auto: skips the missing CLI, then provider a retries past its dead first model.
	ans, err := g.Ask(ctx, "auto", "", "hi", "")
	if err != nil || ans.Used != "a/live" || !strings.Contains(ans.Text, "live says: hi") {
		t.Fatalf("auto: %+v, %v", ans, err)
	}
	if a.count() != 2 || b.count() != 0 {
		t.Errorf("auto: calls a=%d b=%d, want 2 and 0", a.count(), b.count())
	}

	// The model that worked is remembered: the next unpinned ask does not retry the dead one.
	if ans, err = g.Ask(ctx, "a", "", "hi", ""); err != nil || ans.Used != "a/live" || a.count() != 3 {
		t.Errorf("remembered model: %+v, %v, calls a=%d want 3", ans, err, a.count())
	}

	// A pinned model is not swapped for another one.
	if _, err := g.Ask(ctx, "a/dead", "", "hi", ""); err == nil || !strings.Contains(err.Error(), "unavailable") {
		t.Errorf("pinned dead model: err = %v", err)
	}

	// A provider id alone uses its default; system and session history are sent along.
	if _, err := g.Ask(ctx, "b", "be brief", "first", "s1"); err != nil {
		t.Fatal(err)
	}
	ans, err = g.Ask(ctx, "b", "be brief", "second", "s1")
	if err != nil || ans.Used != "b/live" || !strings.Contains(ans.Text, "second (4 msgs)") {
		t.Errorf("session: %+v, %v", ans, err)
	}

	// CLI adapter: prompt substituted, output returned.
	ans, err = g.Ask(ctx, "echo", "", "ping", "")
	if err != nil || ans.Text != "cli got: ping" || ans.Used != "echo/default" {
		t.Errorf("cli: %+v, %v", ans, err)
	}

	// A pinned CLI model is passed after the ask arguments.
	if ans, err = g.Ask(ctx, "echo/-n", "", "ping", ""); err != nil || ans.Text != "cli got: ping -n" {
		t.Errorf("cli model args: %+v, %v", ans, err)
	}

	// A CLI runs in the scratch directory with the entry's env, not in the caller's directory.
	ans, err = g.Ask(ctx, "sh", "", "x", "")
	if err != nil || !strings.Contains(ans.Text, "/ai-gateway-runs/aigw-") || !strings.HasSuffix(ans.Text, " trusted") {
		t.Errorf("cli cwd and env: %+v, %v", ans, err)
	}

	// Unknown provider and empty prompt are refused with a usable message.
	if _, err := g.Ask(ctx, "nope", "", "hi", ""); err == nil || !strings.Contains(err.Error(), "one of: nocli, a, b, echo, sh") {
		t.Errorf("unknown provider: %v", err)
	}
	if _, err := g.Ask(ctx, "", "", "  ", ""); err == nil {
		t.Error("empty prompt accepted")
	}

	// A provider that fails falls through to the next one.
	a.down = true
	ans, err = g.Ask(ctx, "auto", "", "hi", "")
	if err != nil || ans.Used != "b/live" {
		t.Errorf("fallback: %+v, %v", ans, err)
	}
	if st := g.Status(ctx); !strings.Contains(st, "nocli (tier B): unavailable") || !strings.Contains(st, "b (tier A): ready;") {
		t.Errorf("status:\n%s", st)
	}
}

func TestBreakerLimitAndLedgerPersistence(t *testing.T) {
	f := &fake{down: true}
	srv := f.server(t)
	cfg := fmt.Sprintf(`{"providers":[{"id":"p","tier":"A","kind":"openai","url":"%s/chat","default_model":"live","daily_limit":5}]}`, srv.URL)
	ps, _ := provider.Parse([]byte(cfg))
	path := filepath.Join(t.TempDir(), "ledger.json")
	g := New(ps, OpenLedger(path))
	clock := time.Date(2026, 11, 14, 10, 0, 0, 0, time.UTC)
	g.now = func() time.Time { return clock }
	ctx := context.Background()

	for i := 0; i < breakerFailures; i++ {
		g.Ask(ctx, "p", "", "hi", "")
	}
	if _, err := g.Ask(ctx, "p", "", "hi", ""); err == nil || !strings.Contains(err.Error(), "resting until 10:10") {
		t.Fatalf("breaker not tripped: %v", err)
	}
	if f.count() != breakerFailures {
		t.Errorf("a resting provider was still called: %d calls", f.count())
	}

	// A new gateway on the same ledger file still sees the breaker; it clears after the rest.
	g = New(ps, OpenLedger(path))
	g.now = func() time.Time { return clock }
	if _, err := g.Ask(ctx, "p", "", "hi", ""); err == nil || !strings.Contains(err.Error(), "resting") {
		t.Errorf("breaker not persisted: %v", err)
	}
	clock = clock.Add(breakerRest + time.Minute)
	f.down = false
	if _, err := g.Ask(ctx, "p", "", "hi", ""); err != nil {
		t.Fatalf("after rest: %v", err)
	}
	if _, err := g.Ask(ctx, "p", "", "hi", ""); err != nil { // 5th call today
		t.Fatal(err)
	}
	if _, err := g.Ask(ctx, "p", "", "hi", ""); err == nil || !strings.Contains(err.Error(), "daily limit reached") {
		t.Errorf("daily limit: %v", err)
	}
	clock = clock.Add(24 * time.Hour)
	if _, err := g.Ask(ctx, "p", "", "hi", ""); err != nil {
		t.Errorf("next day: %v", err)
	}
}

func TestMinInterval(t *testing.T) {
	f := &fake{}
	srv := f.server(t)
	g := build(t, fmt.Sprintf(`{"providers":[{"id":"p","tier":"A","kind":"openai","url":"%s/chat","default_model":"live","min_interval_seconds":0.3}]}`, srv.URL))
	start := time.Now()
	for i := 0; i < 3; i++ {
		if _, err := g.Ask(context.Background(), "p", "", "hi", ""); err != nil {
			t.Fatal(err)
		}
	}
	if d := time.Since(start); d < 600*time.Millisecond {
		t.Errorf("3 calls took %s, want at least 600ms", d)
	}
}

// TestMCPEndToEnd drives the tools the way Bill's client does, over stdio and over HTTP.
func TestMCPEndToEnd(t *testing.T) {
	f := &fake{}
	up := f.server(t)
	g := build(t, fmt.Sprintf(`{"providers":[
		{"id":"p","tier":"A","kind":"openai","url":"%s/chat","default_model":"live"},
		{"id":"q","tier":"A","kind":"openai","url":"%s/chat","default_model":"other"}]}`, up.URL, up.URL))
	srv := &mcp.Server{Name: "ai-gateway", Version: "test", Tools: g.Tools()}

	in := strings.Join([]string{
		`{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"BillCat","version":"1"}}}`,
		`{"jsonrpc":"2.0","method":"notifications/initialized"}`,
		`{"jsonrpc":"2.0","id":2,"method":"tools/list"}`,
		`{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"ask","arguments":{"prompt":"hello","model":"p"}}}`,
		`{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"ask_many","arguments":{"prompt":"hello","models":["p","q","zzz"]}}}`,
		`{"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"ask","arguments":{"prompt":"hello","model":"zzz"}}}`,
		`{"jsonrpc":"2.0","id":6,"method":"tools/call","params":{"name":"nope","arguments":{}}}`,
		`{"jsonrpc":"2.0","id":7,"method":"resources/list"}`,
		`not json`,
	}, "\n") + "\n"
	var out bytes.Buffer
	if err := srv.ServeStdio(context.Background(), strings.NewReader(in), &out); err != nil {
		t.Fatal(err)
	}
	got := map[string]map[string]any{}
	for _, line := range strings.Split(strings.TrimSpace(out.String()), "\n") {
		var m map[string]any
		if err := json.Unmarshal([]byte(line), &m); err != nil {
			t.Fatalf("reply is not one JSON object per line: %q", line)
		}
		got[fmt.Sprint(m["id"])] = m
	}
	if len(got) != 8 { // ids 1-7 and the parse error (id null); the notification gets no reply
		t.Fatalf("got %d replies, want 8:\n%s", len(got), out.String())
	}
	text := func(id string) (string, bool) {
		r := got[id]["result"].(map[string]any)
		return r["content"].([]any)[0].(map[string]any)["text"].(string), r["isError"].(bool)
	}
	if v := got["1"]["result"].(map[string]any)["protocolVersion"]; v != "2025-06-18" {
		t.Errorf("protocolVersion %v", v)
	}
	if n := len(got["2"]["result"].(map[string]any)["tools"].([]any)); n != 7 {
		t.Errorf("%d tools, want 7", n)
	}
	if s, isErr := text("3"); isErr || !strings.HasPrefix(s, "[answered by p/live]\n\nlive says: hello") {
		t.Errorf("ask: %q %v", s, isErr)
	}
	if s, isErr := text("4"); isErr || !strings.Contains(s, "## p/live") || !strings.Contains(s, "## q/other") || !strings.Contains(s, "## zzz (failed)") {
		t.Errorf("ask_many: %q %v", s, isErr)
	}
	if s, isErr := text("5"); !isErr || !strings.Contains(s, "unknown provider") {
		t.Errorf("ask unknown: %q %v", s, isErr)
	}
	for id, code := range map[string]float64{"6": -32602, "7": -32601, "<nil>": -32700} {
		if e, _ := got[id]["error"].(map[string]any); e == nil || e["code"] != code {
			t.Errorf("id %s: error %v, want code %v", id, got[id]["error"], code)
		}
	}

	// HTTP: plain JSON reply, 202 for a notification, and no access for another site's page.
	hs := httptest.NewServer(srv.HTTPHandler())
	defer hs.Close()
	post := func(body, origin string) (*http.Response, string) {
		req, _ := http.NewRequest("POST", hs.URL, strings.NewReader(body))
		req.Header.Set("Accept", "application/json, text/event-stream")
		if origin != "" {
			req.Header.Set("Origin", origin)
		}
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			t.Fatal(err)
		}
		b, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		return resp, string(b)
	}
	if resp, body := post(`{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"list_models","arguments":{}}}`, ""); resp.StatusCode != 200 ||
		!strings.Contains(resp.Header.Get("Content-Type"), "application/json") || !strings.Contains(body, "p (tier A)") {
		t.Errorf("http call: %d %s", resp.StatusCode, body)
	}
	if resp, _ := post(`{"jsonrpc":"2.0","method":"notifications/initialized"}`, ""); resp.StatusCode != 202 {
		t.Errorf("notification: %d, want 202", resp.StatusCode)
	}
	if resp, _ := post(`{"jsonrpc":"2.0","id":1,"method":"ping"}`, "https://evil.example"); resp.StatusCode != 403 {
		t.Errorf("foreign origin: %d, want 403", resp.StatusCode)
	}
}

// TestLearnsFromFailures covers what the first real checks ran into: models that need a key, a model
// that is busy, and a spent free allowance.
func TestLearnsFromFailures(t *testing.T) {
	f := &fake{}
	srv := f.server(t)
	g := build(t, fmt.Sprintf(`{"providers":[
		{"id":"k","tier":"A","kind":"openai","url":"%[1]s/chat","models":["keyed1","keyed2","keyed3","keyed4","busy1","live"]},
		{"id":"q","tier":"B","kind":"openai","url":"%[1]s/chat","models":["spent1","spent2","live"]}]}`, srv.URL))
	clock := time.Date(2026, 11, 14, 22, 0, 0, 0, time.Local)
	g.now = func() time.Time { return clock }
	ctx := context.Background()

	// Keyed models are passed over without using up the three real tries; the busy one is one try.
	ans, err := g.Ask(ctx, "k", "", "hi", "")
	if err != nil || ans.Used != "k/live" || f.count() != 6 {
		t.Fatalf("keyed and busy: %+v, %v, %d calls want 6", ans, err, f.count())
	}
	// Next time only the model that worked is called, and a pinned keyed model is refused unsent.
	if _, err := g.Ask(ctx, "k", "", "hi", ""); err != nil || f.count() != 7 {
		t.Errorf("second ask: %v, %d calls want 7", err, f.count())
	}
	if _, err := g.Ask(ctx, "k/keyed2", "", "hi", ""); err == nil || !strings.Contains(err.Error(), "needs an API key") || f.count() != 7 {
		t.Errorf("pinned keyed model: %v, %d calls want 7", err, f.count())
	}
	if st := g.Status(ctx); !strings.Contains(st, "needs an API key: keyed1, keyed2, keyed3, keyed4") || !strings.Contains(st, "busy: busy1") {
		t.Errorf("status:\n%s", st)
	}

	// Two models reporting a spent allowance rest the whole provider until midnight.
	if _, err := g.Ask(ctx, "q", "", "hi", ""); err == nil || f.count() != 9 {
		t.Fatalf("quota: %v, %d calls want 9", err, f.count())
	}
	if _, err := g.Ask(ctx, "q", "", "hi", ""); err == nil || !strings.Contains(err.Error(), "resting until Nov 15 00:00: free allowance used up") || f.count() != 9 {
		t.Errorf("resting provider: %v, %d calls want 9", err, f.count())
	}
	clock = clock.Add(3 * time.Hour) // past midnight
	if ans, err := g.Ask(ctx, "q", "", "hi", ""); err == nil || f.count() != 11 {
		t.Errorf("after midnight the provider is tried again: %+v, %v, %d calls want 11", ans, err, f.count())
	}
}

// A CLI that hides its cause in a report file (Gemini CLI) still gets classified as a spent quota.
func TestCLIReportFile(t *testing.T) {
	report := filepath.Join(t.TempDir(), "gemini-client-error-test.json")
	os.WriteFile(report, []byte(`{"error":{"message":"You have exhausted your daily quota on this model."}}`), 0o644)
	g := build(t, fmt.Sprintf(`{"providers":[{"id":"c","tier":"B","kind":"cli","bin":"sh",
		"ask_args":["-c","echo 'Error generating content via API. Full report available at: %s' >&2; exit 1 # {prompt}"]}]}`, report))
	_, err := g.Ask(context.Background(), "c", "", "hi", "")
	if err == nil || !strings.Contains(err.Error(), "exhausted your daily quota") || classify(err) != quota {
		t.Errorf("report file: %v", err)
	}
}

// Copilot CLI prints which model it picked before the answer; the entry's strip pattern removes it.
func TestCopilotNoticeIsStripped(t *testing.T) {
	data, err := os.ReadFile("../../providers/providers.json")
	if err != nil {
		t.Fatal(err)
	}
	var f struct {
		Providers []map[string]any `json:"providers"`
	}
	json.Unmarshal(data, &f)
	var strip any
	for _, p := range f.Providers {
		if p["id"] == "copilot-cli" {
			strip = p["strip"]
		}
	}
	cfg, _ := json.Marshal(map[string]any{"providers": []map[string]any{{"id": "c", "tier": "B", "kind": "cli", "bin": "printf", "strip": strip,
		"ask_args": []string{"● Copilot Fusion selected MAI-Code-1.1-Flash because it was the best choice out of the models\n   available in your plan.\n\n%s\n\n● a bullet in the answer stays\n", "{prompt}"}}}})
	g := build(t, string(cfg))
	ans, err := g.Ask(context.Background(), "c", "", "PONG", "")
	if err != nil || ans.Text != "PONG\n\n● a bullet in the answer stays" {
		t.Errorf("got %q, %v", ans.Text, err)
	}
}

// A model that runs into the provider's timeout rests, so the next ask goes straight to one that works.
func TestTimedOutModelRests(t *testing.T) {
	g := build(t, `{"providers":[{"id":"c","tier":"B","kind":"cli","bin":"sh","timeout_seconds":1,
		"models":["hang","ok"],"model_args":["{model}"],
		"ask_args":["-c","if [ \"$0\" = hang ]; then sleep 30; fi; echo answer # {prompt}"]}]}`)
	ctx := context.Background()
	start := time.Now()
	ans, err := g.Ask(ctx, "c", "", "hi", "")
	if err != nil || ans.Used != "c/ok" {
		t.Fatalf("first ask: %+v, %v", ans, err)
	}
	if why := g.ledger.ModelBlocked("c", "hang", time.Now()); !strings.Contains(why, "timed out") {
		t.Errorf("the model that timed out is not resting: %q", why)
	}
	if d := time.Since(start); d > 8*time.Second {
		t.Errorf("took %s", d)
	}
}

// What the first real fan-out on betelgeuse showed: coding-agent CLIs write files into the directory
// they run in, and the next one read them as "the project". Each run now gets its own empty
// directory, deleted afterwards along with what the entry's cleanup names, and a standing
// instruction to answer in the reply.
func TestEachCLIRunIsIsolated(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	g := build(t, `{"preamble":"Answer in the reply.","providers":[
		{"id":"c","tier":"B","kind":"cli","bin":"sh","cleanup":["~/tool-state/{scratch_base}"],
		 "ask_args":["-c","ls | wc -l | tr -d ' '; pwd; touch leftover.py; mkdir -p $HOME/tool-state/$(basename $(pwd)); echo \"$0\"","{prompt}"]},
		{"id":"plain","tier":"B","kind":"cli","bin":"echo","ask_args":["{prompt}"],"preamble":""}]}`)
	ctx := context.Background()
	var dirs []string
	for i := 0; i < 2; i++ {
		ans, err := g.Ask(ctx, "c", "", "do it", "")
		if err != nil {
			t.Fatal(err)
		}
		lines := strings.Split(ans.Text, "\n")
		if lines[0] != "0" {
			t.Errorf("run %d did not start in an empty directory: %q", i, ans.Text)
		}
		if !strings.Contains(ans.Text, "Answer in the reply.\n\ndo it") {
			t.Errorf("preamble missing: %q", ans.Text)
		}
		dirs = append(dirs, lines[1])
	}
	if dirs[0] == dirs[1] {
		t.Error("two runs shared a directory")
	}
	for _, d := range dirs {
		if _, err := os.Stat(d); !os.IsNotExist(err) {
			t.Errorf("scratch directory %s was not removed", d)
		}
	}
	if left, _ := os.ReadDir(filepath.Join(home, "tool-state")); len(left) != 0 {
		t.Errorf("the tool's own per-run state was not cleaned up: %v", left)
	}
	if ans, _ := g.Ask(ctx, "plain", "", "just this", ""); ans.Text != "just this" {
		t.Errorf("an entry with an empty preamble still got one: %q", ans.Text)
	}
	if _, err := provider.Parse([]byte(`{"providers":[{"id":"x","kind":"cli","bin":"sh","ask_args":["{prompt}"],"cleanup":["~/everything"]}]}`)); err == nil {
		t.Error("a cleanup glob without {scratch_base} was accepted")
	}
}

// A rate limit is a short rest, not a spent daily allowance, however it is worded. (Pollinations was
// rested until midnight on betelgeuse and answered again minutes later.)
func TestRateLimitIsNotADailyQuota(t *testing.T) {
	for msg, want := range map[string]failure{
		"HTTP 429: Rate limit exceeded. Quota: 1 request per 15 seconds":                                   busy,
		"HTTP 429: Too many requests, limit reached":                                                       busy,
		"HTTP 429: You have exceeded your daily quota":                                                     quota,
		"exit status 1: You have exhausted your daily quota on this model.":                                quota,
		"exit status 1: Error: Free usage limit reached. Free models work in OpenCode":                     quota,
		"empty output: Error: Out of Credits Add credits to keep using Amp.":                               quota,
		"HTTP 503: This model is temporarily busy. Please try again shortly":                               busy,
		"HTTP 401: Missing API key.":                                                                       needsKey,
		"timed out after 3m0s":                                                                             slow,
		`HTTP 402: {"error":{"code":"PAYMENT_METHOD_REQUIRED","message":"A payment method is required."}}`: paid,
		"HTTP 500: boom": other,
	} {
		if got := classify(errors.New(msg)); got != want {
			t.Errorf("%q classified as %d, want %d", msg, got, want)
		}
	}

	// reset clears a rest.
	f := &fake{}
	srv := f.server(t)
	g := build(t, fmt.Sprintf(`{"providers":[{"id":"q","tier":"A","kind":"openai","url":"%s/chat","models":["spent1","spent2"]}]}`, srv.URL))
	g.Ask(context.Background(), "q", "", "hi", "")
	if len(g.Free()) != 0 {
		t.Fatal("provider not resting")
	}
	if err := g.Reset("q"); err != nil || len(g.Free()) != 1 {
		t.Errorf("after reset: %v, free %v", err, g.Free())
	}
	if g.Reset("nope") == nil {
		t.Error("reset of an unknown provider accepted")
	}
}

func TestUnfence(t *testing.T) {
	code := "def is_prime(n):\n    return n > 1"
	for in, want := range map[string]string{
		"```python\n" + code + "\n```":              code,
		"```\n" + code + "\n```\n":                  code,
		"  ```c++\r\n" + code + "\r\n```":           code,
		code:                                        code,
		"Here it is:\n```python\n" + code + "\n```": "Here it is:\n```python\n" + code + "\n```",
		"```py\na\n```\nand\n```py\nb\n```":         "```py\na\n```\nand\n```py\nb\n```",
		"```python\n```":                            "```python\n```",
	} {
		if got := unfence(in); got != want {
			t.Errorf("unfence(%q) = %q, want %q", in, got, want)
		}
	}
	// End to end: a provider that wraps its whole answer in a fence returns bare code.
	g := build(t, `{"providers":[{"id":"c","tier":"B","kind":"cli","bin":"printf","preamble":"","ask_args":["`+"```"+`python\\nprint(1)\\n`+"```"+`\\n%.0s","{prompt}"]}]}`)
	if ans, err := g.Ask(context.Background(), "c", "", "x", ""); err != nil || ans.Text != "print(1)" {
		t.Errorf("got %q, %v", ans.Text, err)
	}
}

// A reply that stops mid-answer is a failure to retry elsewhere, not an answer.
func TestCutOffReplyIsRetried(t *testing.T) {
	f := &fake{}
	srv := f.server(t)
	g := build(t, fmt.Sprintf(`{"providers":[
		{"id":"p","tier":"A","kind":"openai","url":"%[1]s/chat","models":["cutfence","cutlength","live"]}]}`, srv.URL))
	ans, err := g.Ask(context.Background(), "p", "", "hi", "")
	if err != nil || ans.Used != "p/live" || f.count() != 3 {
		t.Errorf("got %+v, %v after %d calls; want p/live after 3", ans, err, f.count())
	}
	if _, err := g.Ask(context.Background(), "p/cutfence", "", "hi", ""); err == nil || !strings.Contains(err.Error(), "cut off inside a code block") {
		t.Errorf("pinned: %v", err)
	}
	if cutOff("Use ` for inline and ``` for blocks.\n```py\nx\n```") {
		t.Error("a fence mentioned mid-line was counted")
	}
}

// A provider that needs a free API key is skipped until the key is set, sends it as a bearer token,
// never lets it leak into an error, and uses only the models its filter allows.
func TestKeyedProvider(t *testing.T) {
	f := &fake{}
	srv := f.server(t)
	g := build(t, fmt.Sprintf(`{"providers":[
		{"id":"k","tier":"K","kind":"openai","url":"%[1]s/chat","models_url":"%[1]s/models","model_filter":"^(live|other)$","key_env":"AIGW_TEST_KEY"},
		{"id":"opt","tier":"A","kind":"openai","url":"%[1]s/chat","default_model":"live","key_env":"AIGW_TEST_KEY","key_optional":true}]}`, srv.URL))
	ctx := context.Background()

	if _, err := g.Ask(ctx, "k", "", "hi", ""); err == nil || !strings.Contains(err.Error(), "ai-gateway key set k") || f.count() != 0 {
		t.Errorf("without a key: %v, %d calls", err, f.count())
	}
	if free := g.Free(); len(free) != 1 || free[0] != "opt" {
		t.Errorf("free without a key: %v", free)
	}
	if _, err := g.Ask(ctx, "opt", "", "hi", ""); err != nil || f.calls[0]["auth"] != "" {
		t.Errorf("optional key, unset: %v, auth %q", err, f.calls[0]["auth"])
	}

	t.Setenv("AIGW_TEST_KEY", "sekret-123")
	ans, err := g.Ask(ctx, "k", "", "hi", "")
	if err != nil || ans.Used != "k/live" { // "dead" is listed first but the filter leaves it out
		t.Fatalf("with a key: %+v, %v", ans, err)
	}
	if got := f.calls[len(f.calls)-1]["auth"]; got != "Bearer sekret-123" {
		t.Errorf("Authorization %q", got)
	}
	if _, err := g.Ask(ctx, "k/echoauth", "", "hi", ""); err == nil || strings.Contains(err.Error(), "sekret-123") || !strings.Contains(err.Error(), "[key]") {
		t.Errorf("the key leaked into an error: %v", err)
	}
}

// Cloudflare's address carries an account id; the provider is skipped until both id and key are set.
func TestAccountInURL(t *testing.T) {
	f := &fake{}
	srv := f.server(t)
	g := build(t, fmt.Sprintf(`{"providers":[{"id":"cf","tier":"K","kind":"openai","url":"%s/{account}/chat","default_model":"live",
		"key_env":"AIGW_TEST_TOKEN","account_env":"AIGW_TEST_ACCOUNT","max_tokens":900}]}`, srv.URL))
	t.Setenv("AIGW_TEST_TOKEN", "tok")
	if len(g.Free()) != 0 {
		t.Error("usable without an account id")
	}
	t.Setenv("AIGW_TEST_ACCOUNT", "acct42")
	defer func() {
		if got := f.calls[len(f.calls)-1]["max_tokens"]; got != float64(900) {
			t.Errorf("max_tokens sent as %v, want 900", got)
		}
	}()
	if ans, err := g.Ask(context.Background(), "cf", "", "hi", ""); err != nil || ans.Used != "cf/live" {
		t.Errorf("with account: %+v, %v", ans, err)
	}
}

// An empty env value removes the variable: a CLI must not find an API key it would prefer to the
// account the user signed in with.
func TestEmptyEnvValueRemovesTheVariable(t *testing.T) {
	t.Setenv("AIGW_LEAKY_KEY", "from-the-shell")
	g := build(t, `{"providers":[{"id":"c","tier":"B","kind":"cli","bin":"sh","preamble":"",
		"env":{"AIGW_LEAKY_KEY":"","AIGW_ADDED":"yes"},
		"ask_args":["-c","echo key=${AIGW_LEAKY_KEY-unset} added=$AIGW_ADDED # {prompt}"]}]}`)
	if ans, err := g.Ask(context.Background(), "c", "", "x", ""); err != nil || ans.Text != "key=unset added=yes" {
		t.Errorf("got %q, %v", ans.Text, err)
	}
}

// agent is a scripted coding agent: it prints fixed code, optionally after a delay.
func agent(id string, strength int, delay float64, code string) string {
	b, _ := json.Marshal(map[string]any{"id": id, "tier": "B", "kind": "cli", "bin": "sh", "strength": strength, "preamble": "",
		"ask_args": []string{"-c", fmt.Sprintf("sleep %v; cat <<'CODE'\n%s\nCODE", delay, code), "{prompt}"}})
	return string(b)
}

func agents(list ...string) string { return `{"providers":[` + strings.Join(list, ",") + `]}` }

// One function, five agents: the version most of them agree on wins, and agreement ignores
// comments, docstrings, blank lines, spacing and Markdown wrapping.
func TestConsensus(t *testing.T) {
	t.Setenv("AI_GATEWAY_VOTE", "text") // this test is about comparing text; TestBehaviourVote runs the code
	plain := "def f(n):\n    return n + 1"
	g := build(t, agents(
		agent("weak", 1, 0, "def f(n): return n+1  # never asked: too weak"),
		agent("top", 9, 0, plain),
		agent("chatty", 8, 0, "Here you go:\n```python\ndef f(n):\n    # add one\n\n    return n  +  1\n```\nHope that helps!"),
		agent("docs", 7, 0, "def f(n):\n    \"\"\"Add one.\"\"\"\n    return n + 1"),
		agent("odd", 6, 0, "def f(n):\n    return 1 + n"),
		agent("other", 5, 0, "f = lambda n: n + 1"),
		`{"id":"down","tier":"B","kind":"cli","bin":"no-such-binary-xyz","strength":10,"ask_args":["{prompt}"]}`))
	v, err := g.Consensus(context.Background(), Request{Prompt: "write f"})
	if err != nil {
		t.Fatal(err)
	}
	if !v.Majority || v.Answers != 5 || strings.Join(v.Groups[0].By, ",") != "top,chatty,docs" || v.Winner() != plain {
		t.Errorf("verdict: majority=%v answers=%d groups=%+v", v.Majority, v.Answers, v.Groups)
	}
	out := v.String()
	if !strings.HasPrefix(out, "MAJORITY: 3 of 5 agents agree (top, chatty, docs); compared as text. Differed: odd, other.") || !strings.Contains(out, plain) || strings.Contains(out, "lambda") {
		t.Errorf("text:\n%s", out)
	}
	// Who sided with the majority is kept, to show which agents to trust.
	if e := g.ledger.Snapshot("odd", time.Now()); e.Votes != 1 || e.Agreed != 0 {
		t.Errorf("odd's record: %+v", e)
	}
	if e := g.ledger.Snapshot("docs", time.Now()); e.Votes != 1 || e.Agreed != 1 {
		t.Errorf("docs's record: %+v", e)
	}

	// No majority: every candidate comes back for the planner to choose, strongest group first.
	g = build(t, agents(agent("a", 9, 0, "x = 1"), agent("b", 8, 0, "x = 2"), agent("c", 7, 0, "x = 3")))
	v, _ = g.Consensus(context.Background(), Request{Prompt: "write x"})
	if v.Majority || v.Winner() != "x = 1" || !strings.Contains(v.String(), "NO MAJORITY among 3 answers") || strings.Count(v.String(), "### candidate") != 3 {
		t.Errorf("no majority:\n%s", v)
	}

	// A seat whose agent fails moves on to the next strongest, so five answers still come back.
	g = build(t, agents(
		`{"id":"broken","tier":"B","kind":"cli","bin":"sh","strength":10,"preamble":"","ask_args":["-c","exit 1","{prompt}"]}`,
		agent("a", 9, 0, "y = 1"), agent("b", 8, 0, "y = 1"), agent("c", 7, 0, "y = 1"), agent("d", 6, 0, "y = 1"), agent("e", 5, 0, "y = 2")))
	v, _ = g.Consensus(context.Background(), Request{Prompt: "write y"})
	if v.Answers != 5 || !v.Majority || len(v.Failed) != 1 || !strings.HasPrefix(v.Failed[0], "broken:") {
		t.Errorf("replacement: answers=%d majority=%v failed=%v", v.Answers, v.Majority, v.Failed)
	}
}

// A program of several functions: agents pull tasks from one queue, the fast one writes more, a
// task that fails goes back for another agent, and the parts come back as one program in task order.
// Parts the planner rejects are rewritten by majority vote.
func TestBuildQueueAndRework(t *testing.T) {
	g := build(t, agents(
		agent("fast", 9, 0.05, "def fast_part(): pass"),
		agent("slow", 8, 1.2, "def slow_part(): pass"),
		`{"id":"broken","tier":"B","kind":"cli","bin":"sh","strength":7,"preamble":"","ask_args":["-c","sleep 0.1; exit 1","{prompt}"]}`))
	dir := t.TempDir()
	g.SetJobsDir(dir)
	ctx := context.Background()

	var tasks []Task
	for i := 1; i <= 6; i++ {
		tasks = append(tasks, Task{ID: fmt.Sprintf("f%d", i), Prompt: "write it"})
	}
	id, plan, err := g.Build(tasks, "", "import os", "main()")
	if err != nil || !strings.Contains(plan, "6 tasks in the queue, 3 agents free to take them, strongest first: fast, slow, broken") {
		t.Fatalf("build: %v\n%s", err, plan)
	}
	if out, _ := g.Collect(ctx, id, 50*time.Millisecond); !strings.Contains(out, "is still running") {
		t.Errorf("early collect:\n%s", out)
	}
	out, err := g.Collect(ctx, id, 20*time.Second)
	if err != nil || !strings.Contains(out, "is complete: 6 of 6 parts written") {
		t.Fatalf("collect: %v\n%s", err, out)
	}
	fast, slow := strings.Count(out, ": fast/default"), strings.Count(out, ": slow/default")
	if fast+slow != 6 || fast <= slow {
		t.Errorf("fast wrote %d parts and slow %d; the fast agent should write more, and the broken agent's tasks should be redone", fast, slow)
	}
	program := out[strings.Index(out, "===== program =====\n")+len("===== program =====\n"):]
	if !strings.HasPrefix(program, "import os\n\ndef ") || !strings.HasSuffix(strings.TrimSpace(program), "main()") || strings.Count(program, "def ") != 6 {
		t.Errorf("program:\n%s", program)
	}

	// Rework: the rejected part is rewritten by vote, seeing the error and the old code.
	if _, err := g.Rework(id, map[string]string{"nope": "x"}); err == nil || !strings.Contains(err.Error(), "f1, f2") {
		t.Errorf("unknown task accepted: %v", err)
	}
	msg, err := g.Rework(id, map[string]string{"f2": "NameError: name 'q' is not defined"})
	if err != nil || !strings.Contains(msg, "reworking 1 parts") {
		t.Fatalf("rework: %v %s", err, msg)
	}
	out, _ = g.Collect(ctx, id, 20*time.Second)
	if !strings.Contains(out, "is complete: 6 of 6") || !strings.Contains(out, "- f2: no majority among 2 answers; used the largest group, 1 (fast)") {
		t.Errorf("after rework:\n%s", out)
	}

	// A gateway restarted afterwards (Bill stops an idle server) still has the program.
	g2 := build(t, agents(agent("fast", 9, 0, "x")))
	g2.SetJobsDir(dir)
	if out, err := g2.Collect(ctx, id, time.Second); err != nil || !strings.Contains(out, "is complete: 6 of 6") {
		t.Errorf("after restart: %v\n%s", err, out)
	}
	if _, err := g.Collect(ctx, "../etc", time.Second); err == nil {
		t.Error("a path was accepted as a build id")
	}
	if _, _, err := g.Build([]Task{{ID: "x", Prompt: "p"}, {ID: "x", Prompt: "q"}}, "", "", ""); err == nil {
		t.Error("duplicate task ids accepted")
	}
}

// The five answers the real agents gave for is_prime on betelgeuse: all correct, all worded
// differently, so comparing text found five groups of one. Compared by behaviour they agree.
var isPrimeAnswers = []string{
	"def is_prime(n):\n    if not isinstance(n, int) or n < 2:\n        return False\n    if n < 4:\n        return True\n    if n % 2 == 0 or n % 3 == 0:\n        return False\n    i = 5\n    while i * i <= n:\n        if n % i == 0 or n % (i + 2) == 0:\n            return False\n        i += 6\n    return True",
	"def is_prime(n: int) -> bool:\n    if n < 2:\n        return False\n    if n in (2, 3):\n        return True\n    if n % 2 == 0 or n % 3 == 0:\n        return False\n    i = 5\n    while i * i <= n:\n        if n % i == 0 or n % (i + 2) == 0:\n            return False\n        i += 6\n    return True",
	"def is_prime(n):\n    if not isinstance(n, int) or isinstance(n, bool) or n < 2:\n        return False\n    if n % 2 == 0:\n        return n == 2\n    divisor = 3\n    while divisor * divisor <= n:\n        if n % divisor == 0:\n            return False\n        divisor += 2\n    return True",
	"import math\n\ndef _helper(n):\n    return n < 2\n\ndef is_prime(n):\n    print('checking', n)\n    if _helper(n):\n        return False\n    return all(n % d for d in range(2, math.isqrt(n) + 1))",
}

func TestBehaviourVote(t *testing.T) {
	if _, err := exec.LookPath("python3"); err != nil {
		t.Skip("python3 is not installed")
	}
	t.Setenv("AI_GATEWAY_UNSANDBOXED", "1") // the test machine has no sandbox-exec; the candidates here are ours
	wrong := "def is_prime(n):\n    return n % 2 == 1"
	loop := "def is_prime(n):\n    while True:\n        pass"
	g := build(t, agents(
		agent("a", 9, 0, isPrimeAnswers[0]), agent("b", 8, 0, isPrimeAnswers[1]), agent("c", 7, 0, isPrimeAnswers[2]),
		agent("d", 6, 0, isPrimeAnswers[3]), agent("e", 5, 0, wrong)))
	ctx := context.Background()

	// Four differently-written correct answers agree; the wrong one is shown with a call that exposes it.
	v, err := g.Consensus(ctx, Request{Prompt: "Write a Python function is_prime(n).", Tests: [][]any{{1}, {2}, {9}, {17}, {25}, {97}}})
	if err != nil {
		t.Fatal(err)
	}
	out := v.String()
	if !v.Majority || strings.Join(v.Groups[0].By, ",") != "a,b,c,d" || v.Winner() != isPrimeAnswers[0] {
		t.Fatalf("groups %+v\n%s", v.Groups, out)
	}
	if !strings.Contains(out, "MAJORITY: 4 of 5 agents agree (a, b, c, d); compared by behaviour on 6 test calls to is_prime. Differed: e.") ||
		!strings.Contains(out, "Where they differ: is_prime(1): the majority gives False, candidate 2 gives True") {
		t.Errorf("text:\n%s", out)
	}

	// With no test calls supplied and no model to write them, plain integers are used.
	if v, _ = g.Consensus(ctx, Request{Prompt: "Write a Python function is_prime(n)."}); !v.Majority || !strings.Contains(v.Method, "compared by behaviour on 16 test calls") {
		t.Errorf("default inputs: %s %+v", v.Method, v.Groups)
	}

	// Code that does not load, or never returns, cannot win and does not stall the vote.
	g = build(t, agents(agent("syntax", 9, 0, "def is_prime(n)\n    return True"), agent("loop", 8, 0, loop), agent("renamed", 7, 0, "def prime(n):\n    return True"),
		agent("ok", 6, 0, isPrimeAnswers[0])))
	start := time.Now()
	v, _ = g.Consensus(ctx, Request{Prompt: "Write a Python function is_prime(n).", Tests: [][]any{{7}, {8}}})
	if v.Majority || v.Groups[0].By[0] != "ok" || time.Since(start) > 20*time.Second {
		t.Errorf("broken candidates: winner %v after %s\n%s", v.Groups[0].By, time.Since(start), v)
	}
	if out := v.String(); !strings.Contains(out, "does not run: SyntaxError") || !strings.Contains(out, "does not run: does not define is_prime") || !strings.Contains(out, "is_prime(7): candidate 1 gives True, candidate 3 gives timed out") {
		t.Errorf("text:\n%s", out)
	}

	// A part of a larger program is tested alongside the code it relies on.
	g = build(t, agents(agent("a", 9, 0, "def area(r):\n    return PI * square(r)"), agent("b", 8, 0, "def area(r):\n    return square(r) * PI"),
		agent("c", 7, 0, "def area(r):\n    return PI * r * r")))
	v, _ = g.Consensus(ctx, Request{Prompt: "Write a Python function area(r).", Tests: [][]any{{1}, {2.5}},
		Context: "PI = 3.14159\n\ndef square(x):\n    return x * x"})
	if !v.Majority || len(v.Groups) != 1 {
		t.Errorf("with context: %s %+v", v.Method, v.Groups)
	}

	// Test calls can come from a model: the first API provider is asked for them.
	f := &fake{}
	srv := f.server(t)
	g = build(t, agents(agent("a", 9, 0, isPrimeAnswers[0]), agent("b", 8, 0, isPrimeAnswers[1]), agent("c", 7, 0, wrong),
		fmt.Sprintf(`{"id":"api","tier":"K","kind":"openai","strength":1,"url":"%s/chat","default_model":"testcalls"}`, srv.URL)))
	v, _ = g.Consensus(ctx, Request{Prompt: "Write a Python function is_prime(n).", Agents: 3})
	if !v.Majority || !strings.Contains(v.Method, "on 3 test calls") || strings.Join(v.Groups[0].By, ",") != "a,b" {
		t.Errorf("model-written calls: %s %+v", v.Method, v.Groups)
	}
}

// Each agent writes its own imports; the assembled Python program has them once, at the top.
func TestBuildHoistsPythonImports(t *testing.T) {
	imports, rest := hoistImports("import os\nLIMIT = 3", []string{
		"import re\nfrom collections import Counter\n\ndef a(s):\n    import json\n    return re.sub('x', 'y', s)",
		"import re\nimport os\n\n\ndef b(s):\n    return Counter(s)",
	})
	if strings.Join(imports, "|") != "import re|from collections import Counter" {
		t.Errorf("imports %q", imports)
	}
	if rest[0] != "def a(s):\n    import json\n    return re.sub('x', 'y', s)" || rest[1] != "def b(s):\n    return Counter(s)" {
		t.Errorf("rest %q", rest)
	}
	// Code that is not Python is left exactly as written.
	parts := []string{"import \"fmt\"\n\nfunc A() {}", "func B() {}"}
	if imports, rest := hoistImports("", parts); imports != nil || rest[0] != parts[0] {
		t.Errorf("non-Python parts were changed: %q %q", imports, rest)
	}
}

// The OpenAI-compatible endpoint, called the way Bill's delegate_code calls its coder: a streamed
// chat completion for model "coder", with the context and the task as two user messages.
func TestOpenAIEndpoint(t *testing.T) {
	t.Setenv("AI_GATEWAY_VOTE", "text")
	g := build(t, agents(agent("a", 9, 0, "def f(n):\n    return n + 1"), agent("b", 8, 0, "def f(n):\n    # the same\n    return n + 1"),
		agent("c", 7, 0, "def f(n):\n    return 1 + n")))
	srv := httptest.NewServer(g.OpenAIHandler())
	defer srv.Close()
	post := func(body string) (int, string) {
		resp, err := http.Post(srv.URL+"/v1/chat/completions", "application/json", strings.NewReader(body))
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()
		b, _ := io.ReadAll(resp.Body)
		return resp.StatusCode, string(b)
	}

	// Streamed, as Bill sends it. The reply is the majority's code in one content delta, then a stop and [DONE].
	code, body := post(`{"model":"coder","stream":true,"temperature":0,"max_tokens":8192,"stream_options":{"include_usage":true},
		"messages":[{"role":"system","content":"You are an expert software engineer."},
		            {"role":"user","content":"CONTEXT:\nnothing yet"},{"role":"user","content":"TASK:\nwrite f"}]}`)
	if code != 200 {
		t.Fatalf("status %d: %s", code, body)
	}
	var content, finish string
	done := false
	for _, line := range strings.Split(body, "\n") {
		payload, ok := strings.CutPrefix(line, "data: ")
		if !ok {
			continue
		}
		if payload == "[DONE]" {
			done = true
			continue
		}
		var c struct {
			Choices []struct {
				Delta        struct{ Content string }
				FinishReason string `json:"finish_reason"`
			}
		}
		if err := json.Unmarshal([]byte(payload), &c); err != nil {
			t.Fatalf("bad chunk %q", payload)
		}
		content += c.Choices[0].Delta.Content
		finish += c.Choices[0].FinishReason
	}
	want := "[ai-gateway vote: majority, 2 of 3 agree (a, b); compared as text]\n\n```python\ndef f(n):\n    return n + 1\n```\n"
	if !done || finish != "stop" || content != want {
		t.Errorf("done=%v finish=%q content=%q", done, finish, content)
	}

	// Not streamed; a provider named instead of a vote; an unknown model; the model list.
	if code, body := post(`{"model":"vote","messages":[{"role":"user","content":[{"type":"text","text":"write f"}]}]}`); code != 200 || !strings.Contains(body, `"finish_reason":"stop"`) || !strings.Contains(body, "majority, 2 of 3 agree") {
		t.Errorf("non-stream: %d %s", code, body)
	}
	if code, body := post(`{"model":"c","messages":[{"role":"user","content":"hi"}]}`); code != 200 || !strings.Contains(body, "return 1 + n") {
		t.Errorf("named provider: %d %s", code, body)
	}
	if code, body := post(`{"model":"nope","messages":[{"role":"user","content":"hi"}]}`); code != 404 || !strings.Contains(body, "unknown provider") {
		t.Errorf("unknown model: %d %s", code, body)
	}
	if code, _ := post(`{"model":"coder","messages":[]}`); code != 400 {
		t.Errorf("empty messages: %d", code)
	}
	resp, _ := http.Get(srv.URL + "/v1/models")
	b, _ := io.ReadAll(resp.Body)
	if !strings.Contains(string(b), `"id":"coder"`) || !strings.Contains(string(b), `"id":"c"`) {
		t.Errorf("models: %s", b)
	}
}

// A vote with a deadline is decided among the agents that answered in time; the slow one is
// named, and is not marked as failing.
func TestVoteDeadline(t *testing.T) {
	t.Setenv("AI_GATEWAY_VOTE", "text")
	g := build(t, agents(agent("slow", 9, 5, "x = 0"), agent("a", 8, 0, "x = 1"), agent("b", 7, 0, "x = 1"), agent("c", 6, 0, "x = 1")))
	start := time.Now()
	v, err := g.Consensus(context.Background(), Request{Prompt: "write x", Agents: 4, Within: time.Second})
	if err != nil || !v.Majority || v.Answers != 3 || time.Since(start) > 4*time.Second {
		t.Fatalf("verdict %+v, %v, after %s", v, err, time.Since(start))
	}
	if len(v.Failed) != 1 || v.Failed[0] != "slow: had not answered after 1s" {
		t.Errorf("failed: %v", v.Failed)
	}
	if len(g.Free()) != 4 {
		t.Errorf("the slow agent was rested for being slow: free %v", g.Free())
	}
}

// Nothing secret or private leaves the machine, whichever way a prompt comes in. An outside provider
// gets the prompt with secrets redacted; a local one gets it whole; a prompt that carries a private
// file is seen by local models only.
func TestPrivacy(t *testing.T) {
	t.Setenv("AI_GATEWAY_VOTE", "text")
	outside, home := &fake{}, &fake{}
	so, sh := outside.server(t), home.server(t)
	cfg := fmt.Sprintf(`{"providers":[
		{"id":"out","tier":"K","kind":"openai","strength":9,"url":"%s/chat","default_model":"live"},
		{"id":"home","tier":"L","kind":"openai","strength":4,"local":true,"url":"%s/chat","default_model":"live"}]}`, so.URL, sh.URL)
	g := build(t, cfg)
	dir := t.TempDir()
	store := privacy.Open(filepath.Join(dir, "privacy.json"))
	if err := store.Edit(func(p *privacy.Policy) { p.BlockedPaths = []string{"/srv/client-x"} }); err != nil {
		t.Fatal(err)
	}
	g.SetPrivacy(store, func() []string { return []string{"the-gateways-own-key"} })
	var logged []string
	g.Logf = func(f string, a ...any) { logged = append(logged, fmt.Sprintf(f, a...)) }
	ctx := context.Background()
	sent := func(f *fake) string { // the last user message a backend received
		f.mu.Lock()
		defer f.mu.Unlock()
		msgs := f.calls[len(f.calls)-1]["messages"].([]any)
		return msgs[len(msgs)-1].(map[string]any)["content"].(string)
	}

	secret := `fix this: client = Thing(api_key="sk-proj-abcdefghijklmnopqrstuvwx") and also the-gateways-own-key`
	ans, err := g.Ask(ctx, "out", "", secret, "")
	if err != nil {
		t.Fatal(err)
	}
	if got := sent(outside); strings.Contains(got, "sk-proj-") || strings.Contains(got, "the-gateways-own-key") || !strings.Contains(got, `api_key="[REDACTED:api-key]"`) {
		t.Errorf("the outside provider was sent: %q", got)
	}
	if ans.Redacted != "2 secrets (api-key, gateway-key)" || len(logged) != 1 || !strings.Contains(logged[0], "redacted 2 secrets") || strings.Contains(logged[0], "sk-proj") {
		t.Errorf("redacted %q, log %q", ans.Redacted, logged)
	}
	if _, err := g.Ask(ctx, "home", "", secret, ""); err != nil || sent(home) != secret {
		t.Errorf("the local provider should get the prompt whole: %v %q", err, sent(home))
	}

	// A hand-off carrying a file from a blocked directory: outside providers are not asked at all.
	private := "TASK:\nfix it\n\nFILE /srv/client-x/app.py (its full current text, supplied by the harness):\n```\nx = 1\n```"
	before := outside.count()
	if _, err := g.Ask(ctx, "out", "", private, ""); err == nil || !strings.Contains(err.Error(), "kept on this machine: /srv/client-x/app.py is in a blocked directory") {
		t.Errorf("a private prompt to an outside provider: %v", err)
	}
	if ans, err := g.Ask(ctx, "auto", "", private, ""); err != nil || ans.Used != "home/live" || ans.Private == "" {
		t.Errorf("auto with a private prompt: %+v, %v", ans, err)
	}
	v, err := g.Consensus(ctx, Request{Prompt: private})
	if err != nil || v.Answers != 1 || v.Groups[0].By[0] != "home" || !strings.Contains(v.String(), "Privacy: kept on this machine, only local models were asked") {
		t.Errorf("a private vote: %+v, %v", v, err)
	}
	id, _, err := g.Build([]Task{{ID: "open", Prompt: "write a"}, {ID: "closed", Prompt: private}}, "", "", "")
	if err != nil {
		t.Fatal(err)
	}
	out, _ := g.Collect(ctx, id, 20*time.Second)
	if !strings.Contains(out, "- closed: home/live") {
		t.Errorf("the private task should be written by the local agent:\n%s", out)
	}
	for _, c := range outside.calls[before:] {
		if b, _ := json.Marshal(c); strings.Contains(string(b), "client-x") {
			t.Errorf("an outside provider saw the private file: %s", b)
		}
	}

	// A credentials file is kept home by its name alone, and rules that cannot be read stop everything.
	env := "FILE /Users/me/app/.env (its full current text, supplied by the harness):\nA=1"
	if _, err := g.Ask(ctx, "out", "", env, ""); err == nil || !strings.Contains(err.Error(), "credentials file") {
		t.Errorf(".env file: %v", err)
	}
	os.WriteFile(filepath.Join(dir, "privacy.json"), []byte("{broken"), 0o600)
	os.Chtimes(filepath.Join(dir, "privacy.json"), time.Now().Add(time.Minute), time.Now().Add(time.Minute))
	if _, err := g.Ask(ctx, "out", "", "an ordinary question", ""); err == nil || !strings.Contains(err.Error(), "privacy rules could not be read") {
		t.Errorf("broken rules: %v", err)
	}
}

package gateway

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/thedarkman41/ai-gateway/internal/mcp"
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
	if err != nil || !strings.HasSuffix(ans.Text, "ai-gateway-scratch trusted") {
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
	if n := len(got["2"]["result"].(map[string]any)["tools"].([]any)); n != 3 {
		t.Errorf("%d tools, want 3", n)
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

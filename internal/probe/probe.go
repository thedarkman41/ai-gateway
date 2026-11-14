// Package probe runs the provider survey: it tests each candidate as it stands today and reports
// which ones answer. Nothing here needs or accepts an API key.
package probe

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"sort"
	"strings"
	"sync"
	"time"
)

// Candidate is one provider entry from providers/candidates.json.
type Candidate struct {
	ID          string   `json:"id"`
	Tier        string   `json:"tier"`
	Kind        string   `json:"kind"` // openai | get | reach | cli | none
	URL         string   `json:"url,omitempty"`
	Model       string   `json:"model,omitempty"`
	ModelsURL   string   `json:"models_url,omitempty"`
	Bin         string   `json:"bin,omitempty"`
	VersionArgs []string `json:"version_args,omitempty"`
	AskArgs     []string `json:"ask_args,omitempty"`
	Note        string   `json:"note,omitempty"`
}

// File is the candidates file.
type File struct {
	Prompt     string      `json:"prompt"`
	Expect     string      `json:"expect"`
	Candidates []Candidate `json:"candidates"`
}

// Status values, from best to worst.
const (
	Answered    = "answered"    // sent the prompt and got the expected reply
	Reachable   = "reachable"   // host or binary is there; no prompt was sent (or none can be yet)
	WrongAnswer = "wrong-reply" // replied, but not with the expected text
	Failed      = "failed"
	Missing     = "not-installed"
	Skipped     = "not-probed"
)

// Result is the outcome for one candidate.
type Result struct {
	Candidate
	Status  string   `json:"status"`
	Detail  string   `json:"detail,omitempty"`
	Models  []string `json:"models,omitempty"`
	Seconds float64  `json:"seconds"`
}

// Options control a survey run.
type Options struct {
	Timeout time.Duration // per request or command
	AskCLI  bool          // also send the prompt through installed CLIs (uses a little free quota)
	Client  *http.Client
}

func Load(path string) (File, error) {
	var f File
	b, err := os.ReadFile(path)
	if err != nil {
		return f, err
	}
	if err := json.Unmarshal(b, &f); err != nil {
		return f, fmt.Errorf("%s: %w", path, err)
	}
	if f.Prompt == "" || f.Expect == "" {
		return f, fmt.Errorf("%s: prompt and expect are required", path)
	}
	return f, nil
}

// Run probes every candidate concurrently and returns results ordered by tier, then id.
func Run(ctx context.Context, f File, o Options) []Result {
	if o.Timeout == 0 {
		o.Timeout = 45 * time.Second
	}
	if o.Client == nil {
		o.Client = &http.Client{}
	}
	out := make([]Result, len(f.Candidates))
	var wg sync.WaitGroup
	for i, c := range f.Candidates {
		wg.Add(1)
		go func() {
			defer wg.Done()
			start := time.Now()
			cctx, cancel := context.WithTimeout(ctx, o.Timeout)
			defer cancel()
			r := one(cctx, c, f, o)
			r.Candidate = c
			r.Seconds = float64(time.Since(start).Milliseconds()) / 1000
			out[i] = r
		}()
	}
	wg.Wait()
	sort.SliceStable(out, func(a, b int) bool {
		if out[a].Tier != out[b].Tier {
			return out[a].Tier < out[b].Tier
		}
		return out[a].ID < out[b].ID
	})
	return out
}

func one(ctx context.Context, c Candidate, f File, o Options) Result {
	switch c.Kind {
	case "openai":
		return probeOpenAI(ctx, c, f, o)
	case "get":
		u := strings.ReplaceAll(c.URL, "{prompt}", url.PathEscape(f.Prompt))
		code, body, err := do(ctx, o.Client, http.MethodGet, u, nil)
		return judge(code, body, err, f.Expect)
	case "reach":
		code, _, err := do(ctx, o.Client, http.MethodGet, c.URL, nil)
		if err != nil {
			return Result{Status: Failed, Detail: short(err.Error())}
		}
		// A 403 or a challenge page still means the host is up; the browser tier will see what a
		// real session gets. Only a server error or no answer counts as failed.
		if code >= 500 {
			return Result{Status: Failed, Detail: fmt.Sprintf("HTTP %d", code)}
		}
		return Result{Status: Reachable, Detail: fmt.Sprintf("HTTP %d", code)}
	case "cli":
		return probeCLI(ctx, c, f, o)
	case "none":
		return Result{Status: Skipped}
	}
	return Result{Status: Failed, Detail: "unknown kind " + c.Kind}
}

func probeOpenAI(ctx context.Context, c Candidate, f File, o Options) Result {
	var models []string
	if c.ModelsURL != "" {
		if code, body, err := do(ctx, o.Client, http.MethodGet, c.ModelsURL, nil); err == nil && code == 200 {
			models = modelIDs(body)
		}
	}
	req, _ := json.Marshal(map[string]any{
		"model":    c.Model,
		"messages": []map[string]string{{"role": "user", "content": f.Prompt}},
	})
	code, body, err := do(ctx, o.Client, http.MethodPost, c.URL, req)
	r := judge(code, []byte(chatText(body)), err, f.Expect)
	r.Models = models
	return r
}

func probeCLI(ctx context.Context, c Candidate, f File, o Options) Result {
	path, err := exec.LookPath(c.Bin)
	if err != nil {
		return Result{Status: Missing}
	}
	ver := ""
	if len(c.VersionArgs) > 0 {
		if b, err := exec.CommandContext(ctx, path, c.VersionArgs...).CombinedOutput(); err == nil {
			ver = short(strings.TrimSpace(string(b)))
		}
	}
	if !o.AskCLI || len(c.AskArgs) == 0 {
		return Result{Status: Reachable, Detail: strings.TrimSpace("installed " + ver)}
	}
	args := make([]string, len(c.AskArgs))
	for i, a := range c.AskArgs {
		args[i] = strings.ReplaceAll(a, "{prompt}", f.Prompt)
	}
	cmd := exec.CommandContext(ctx, path, args...)
	cmd.Stdin = nil // never wait on a login or approval prompt
	b, err := cmd.CombinedOutput()
	if err != nil && !strings.Contains(string(b), f.Expect) {
		return Result{Status: Failed, Detail: short(firstNonEmpty(strings.TrimSpace(string(b)), err.Error()))}
	}
	return judge(200, b, nil, f.Expect)
}

func judge(code int, body []byte, err error, expect string) Result {
	switch {
	case err != nil:
		return Result{Status: Failed, Detail: short(err.Error())}
	case code != 200:
		return Result{Status: Failed, Detail: short(fmt.Sprintf("HTTP %d %s", code, strings.TrimSpace(string(body))))}
	case strings.Contains(strings.ToUpper(string(body)), strings.ToUpper(expect)):
		return Result{Status: Answered}
	}
	return Result{Status: WrongAnswer, Detail: short(strings.TrimSpace(string(body)))}
}

func do(ctx context.Context, cl *http.Client, method, u string, body []byte) (int, []byte, error) {
	req, err := http.NewRequestWithContext(ctx, method, u, bytes.NewReader(body))
	if err != nil {
		return 0, nil, err
	}
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	req.Header.Set("User-Agent", "ai-gateway-probe/0.1")
	resp, err := cl.Do(req)
	if err != nil {
		return 0, nil, err
	}
	defer resp.Body.Close()
	b, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	return resp.StatusCode, b, err
}

// chatText pulls the reply out of an OpenAI-style chat completion; anything else is returned as is.
func chatText(body []byte) string {
	var v struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if json.Unmarshal(body, &v) == nil && len(v.Choices) > 0 {
		return v.Choices[0].Message.Content
	}
	return string(body)
}

// modelIDs reads a model list in either common shape: {"data":[{"id":..}]} or [{"id"|"name":..}].
func modelIDs(body []byte) []string {
	type m struct {
		ID   string `json:"id"`
		Name string `json:"name"`
	}
	var wrapped struct {
		Data []m `json:"data"`
	}
	var list []m
	if json.Unmarshal(body, &wrapped) == nil && len(wrapped.Data) > 0 {
		list = wrapped.Data
	} else if json.Unmarshal(body, &list) != nil {
		return nil
	}
	var ids []string
	for _, x := range list {
		if id := firstNonEmpty(x.ID, x.Name); id != "" {
			ids = append(ids, id)
		}
	}
	return ids
}

// Markdown renders the provider matrix.
func Markdown(rs []Result, when time.Time) string {
	var b strings.Builder
	fmt.Fprintf(&b, "# Provider survey\n\nRun %s. `answered` means the prompt went out and the expected reply came back.\n\n", when.Format("2006-01-02 15:04 MST"))
	b.WriteString("| Tier | Provider | Status | Time | Detail |\n|---|---|---|---|---|\n")
	for _, r := range rs {
		d := firstNonEmpty(r.Detail, r.Note)
		if len(r.Models) > 0 {
			d = strings.TrimSpace(fmt.Sprintf("%d models listed. %s", len(r.Models), d))
		}
		fmt.Fprintf(&b, "| %s | %s | %s | %.1fs | %s |\n", r.Tier, r.ID, r.Status, r.Seconds, strings.ReplaceAll(d, "|", "\\|"))
	}
	return b.String()
}

func short(s string) string {
	s = strings.Join(strings.Fields(s), " ")
	if len(s) > 160 {
		return s[:160] + "…"
	}
	return s
}

func firstNonEmpty(a ...string) string {
	for _, s := range a {
		if s != "" {
			return s
		}
	}
	return ""
}

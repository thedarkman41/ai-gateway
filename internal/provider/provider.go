// Package provider holds the adapters that reach other AI models. Every adapter is free and
// keyless: none takes an API key.
package provider

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os/exec"
	"regexp"
	"strings"
	"sync"
	"time"
)

type Message struct {
	Role    string `json:"role"` // system | user | assistant
	Content string `json:"content"`
}

// Request is one question. Messages holds the conversation so far and ends with the user's turn.
type Request struct {
	Model    string // empty = the provider's default
	Messages []Message
}

type Provider interface {
	Config() Config
	// Available reports why the provider cannot be used at all right now (a CLI that is not
	// installed), or nil.
	Available() error
	// Models lists the model ids the provider offers; the first is the default.
	Models(ctx context.Context) []string
	Ask(ctx context.Context, r Request) (string, error)
}

// Config is one entry of providers.json.
type Config struct {
	ID                 string   `json:"id"`
	Tier               string   `json:"tier"`
	Kind               string   `json:"kind"` // openai | cli
	URL                string   `json:"url,omitempty"`
	ModelsURL          string   `json:"models_url,omitempty"`
	DefaultModel       string   `json:"default_model,omitempty"`
	StaticModels       []string `json:"models,omitempty"`
	Bin                string   `json:"bin,omitempty"`
	AskArgs            []string `json:"ask_args,omitempty"`
	ModelArgs          []string `json:"model_args,omitempty"`
	TimeoutSeconds     int      `json:"timeout_seconds,omitempty"`
	MinIntervalSeconds float64  `json:"min_interval_seconds,omitempty"`
	DailyLimit         int      `json:"daily_limit,omitempty"`
}

func (c Config) Timeout() time.Duration {
	if c.TimeoutSeconds <= 0 {
		return 120 * time.Second
	}
	return time.Duration(c.TimeoutSeconds) * time.Second
}

// Parse reads a providers.json document.
func Parse(data []byte) ([]Provider, error) {
	var f struct {
		Providers []Config `json:"providers"`
	}
	if err := json.Unmarshal(data, &f); err != nil {
		return nil, err
	}
	var out []Provider
	seen := map[string]bool{}
	for _, c := range f.Providers {
		if c.ID == "" || strings.ContainsAny(c.ID, "/ ") || seen[c.ID] {
			return nil, fmt.Errorf("provider id %q is empty, repeated, or contains a slash or space", c.ID)
		}
		seen[c.ID] = true
		switch c.Kind {
		case "openai":
			if c.URL == "" {
				return nil, fmt.Errorf("%s: url is required", c.ID)
			}
			out = append(out, &openAI{cfg: c, client: &http.Client{}})
		case "cli":
			if c.Bin == "" || !hasPrompt(c.AskArgs) {
				return nil, fmt.Errorf("%s: bin and ask_args with {prompt} are required", c.ID)
			}
			out = append(out, &cli{cfg: c})
		default:
			return nil, fmt.Errorf("%s: unknown kind %q", c.ID, c.Kind)
		}
	}
	if len(out) == 0 {
		return nil, errors.New("no providers configured")
	}
	return out, nil
}

func hasPrompt(args []string) bool {
	for _, a := range args {
		if strings.Contains(a, "{prompt}") {
			return true
		}
	}
	return false
}

// ---- OpenAI-compatible HTTP endpoint ----

type openAI struct {
	cfg    Config
	client *http.Client

	mu      sync.Mutex
	models  []string
	fetched time.Time
}

func (p *openAI) Config() Config   { return p.cfg }
func (p *openAI) Available() error { return nil }

func (p *openAI) Models(ctx context.Context) []string {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.cfg.ModelsURL != "" && time.Since(p.fetched) > 10*time.Minute {
		p.fetched = time.Now() // also after a failure, so a dead list is not refetched on every call
		cctx, cancel := context.WithTimeout(ctx, 15*time.Second)
		if code, body, err := HTTPDo(cctx, p.client, http.MethodGet, p.cfg.ModelsURL, nil); err == nil && code == 200 {
			if ids := ModelIDs(body); len(ids) > 0 {
				p.models = ids
			}
		}
		cancel()
	}
	return order(p.cfg.DefaultModel, p.models, p.cfg.StaticModels)
}

func (p *openAI) Ask(ctx context.Context, r Request) (string, error) {
	body, _ := json.Marshal(map[string]any{"model": r.Model, "messages": r.Messages})
	code, out, err := HTTPDo(ctx, p.client, http.MethodPost, p.cfg.URL, body)
	if err != nil {
		return "", err
	}
	if code != 200 {
		return "", fmt.Errorf("HTTP %d: %s", code, Short(string(out)))
	}
	text, ok := ChatText(out)
	if !ok || strings.TrimSpace(text) == "" {
		return "", fmt.Errorf("no reply text in: %s", Short(string(out)))
	}
	return strings.TrimSpace(text), nil
}

// ---- vendor CLI, run headless ----

type cli struct{ cfg Config }

func (p *cli) Config() Config { return p.cfg }

func (p *cli) Available() error {
	if _, err := exec.LookPath(p.cfg.Bin); err != nil {
		return fmt.Errorf("%s is not installed (or not on PATH)", p.cfg.Bin)
	}
	return nil
}

func (p *cli) Models(context.Context) []string {
	return order(p.cfg.DefaultModel, p.cfg.StaticModels, []string{"default"})
}

var ansi = regexp.MustCompile(`\x1b\[[0-9;?]*[ -/]*[@-~]`)

func (p *cli) Ask(ctx context.Context, r Request) (string, error) {
	path, err := exec.LookPath(p.cfg.Bin)
	if err != nil {
		return "", p.Available()
	}
	prompt := Flatten(r.Messages)
	if len(prompt) > 200_000 {
		return "", errors.New("prompt is too long to pass to a CLI (200,000 characters)")
	}
	var args []string
	if r.Model != "" && r.Model != "default" {
		for _, a := range p.cfg.ModelArgs {
			args = append(args, strings.ReplaceAll(a, "{model}", r.Model))
		}
	}
	for _, a := range p.cfg.AskArgs {
		args = append(args, strings.ReplaceAll(a, "{prompt}", prompt))
	}
	var stdout, stderr bytes.Buffer
	cmd := exec.CommandContext(ctx, path, args...)
	cmd.Stdout, cmd.Stderr = &stdout, &stderr // stdin stays closed: never wait on a login prompt
	cmd.WaitDelay = 5 * time.Second
	runErr := cmd.Run()
	text := strings.TrimSpace(ansi.ReplaceAllString(stdout.String(), ""))
	if runErr != nil {
		return "", fmt.Errorf("%s: %s", Short(runErr.Error()), Short(firstNonEmpty(strings.TrimSpace(stderr.String()), text)))
	}
	if text == "" {
		return "", fmt.Errorf("empty output: %s", Short(stderr.String()))
	}
	return text, nil
}

// Flatten renders a conversation as one prompt, for backends that take a single string.
func Flatten(ms []Message) string {
	if len(ms) == 1 {
		return ms[0].Content
	}
	var b strings.Builder
	for i, m := range ms {
		if i == len(ms)-1 && m.Role == "user" {
			b.WriteString(m.Content)
			break
		}
		label := map[string]string{"system": "Instructions", "user": "User", "assistant": "Assistant"}[m.Role]
		fmt.Fprintf(&b, "%s: %s\n\n", label, m.Content)
	}
	return b.String()
}

// ---- shared helpers (also used by the survey probe) ----

func HTTPDo(ctx context.Context, cl *http.Client, method, u string, body []byte) (int, []byte, error) {
	req, err := http.NewRequestWithContext(ctx, method, u, bytes.NewReader(body))
	if err != nil {
		return 0, nil, err
	}
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	req.Header.Set("User-Agent", "ai-gateway/0.2")
	resp, err := cl.Do(req)
	if err != nil {
		return 0, nil, err
	}
	defer resp.Body.Close()
	b, err := io.ReadAll(io.LimitReader(resp.Body, 4<<20))
	return resp.StatusCode, b, err
}

// ChatText pulls the reply out of an OpenAI-style chat completion.
func ChatText(body []byte) (string, bool) {
	var v struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if json.Unmarshal(body, &v) == nil && len(v.Choices) > 0 {
		return v.Choices[0].Message.Content, true
	}
	return "", false
}

// ModelIDs reads a model list in either common shape: {"data":[{"id":..}]} or [{"id"|"name":..}].
func ModelIDs(body []byte) []string {
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

// Short collapses whitespace and caps the length, for error text.
func Short(s string) string {
	s = strings.Join(strings.Fields(s), " ")
	if len(s) > 200 {
		return s[:200] + "…"
	}
	return s
}

// order returns the default first, then every other id once.
func order(def string, lists ...[]string) []string {
	var out []string
	seen := map[string]bool{}
	add := func(s string) {
		if s != "" && !seen[s] {
			seen[s] = true
			out = append(out, s)
		}
	}
	add(def)
	for _, l := range lists {
		for _, s := range l {
			add(s)
		}
	}
	return out
}

func firstNonEmpty(a ...string) string {
	for _, s := range a {
		if s != "" {
			return s
		}
	}
	return ""
}

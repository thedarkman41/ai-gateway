// Package provider holds the adapters that reach other AI models. Every adapter is free and
// keyless: none takes an API key.
package provider

import (
	"bytes"
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"slices"
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
	ID           string            `json:"id"`
	Tier         string            `json:"tier"`
	Kind         string            `json:"kind"` // openai | cli | tty
	URL          string            `json:"url,omitempty"`
	ModelsURL    string            `json:"models_url,omitempty"`
	DefaultModel string            `json:"default_model,omitempty"`
	StaticModels []string          `json:"models,omitempty"`
	ModelFilter  string            `json:"model_filter,omitempty"` // regex: only listed models matching it are used (a provider's free ones)
	KeyEnv       string            `json:"key_env,omitempty"`      // the provider needs a free API key: the environment variable that may hold it (else the key file)
	KeyOptional  bool              `json:"key_optional,omitempty"` // works without the key; with one, limits are higher
	KeyURL       string            `json:"key_url,omitempty"`      // where to create the key
	AccountEnv   string            `json:"account_env,omitempty"`  // the url has {account}: the variable that may hold the account id (else the key file, under "<id>.account")
	MaxTokens    int               `json:"max_tokens,omitempty"`   // sent as max_tokens, for a provider whose default cuts replies short
	Bin          string            `json:"bin,omitempty"`
	AskArgs      []string          `json:"ask_args,omitempty"`
	ModelArgs    []string          `json:"model_args,omitempty"`
	Env          map[string]string `json:"env,omitempty"`      // extra environment for a CLI
	Strip        []string          `json:"strip,omitempty"`    // regexes removed from a CLI's output, for a notice the CLI prints around its answer
	Preamble     *string           `json:"preamble,omitempty"` // put before every prompt to a cli or tty entry; unset = the file's top-level preamble
	Cleanup      []string          `json:"cleanup,omitempty"`  // globs the tool leaves behind per run, removed afterwards; ~ and {scratch_base} expand, and each must contain {scratch_base}

	// kind tty: an interactive terminal app driven through a pseudo-terminal.
	Args               []string `json:"args,omitempty"`               // launch arguments; {scratch} is the scratch directory
	ReadyIdleSeconds   float64  `json:"ready_idle_seconds,omitempty"` // quiet this long after start = ready (default 2)
	DoneIdleSeconds    float64  `json:"done_idle_seconds,omitempty"`  // quiet this long after the prompt = reply finished (default 6)
	MinReplySeconds    float64  `json:"min_reply_seconds,omitempty"`  // never finish sooner than this (default 2)
	DoneRegex          string   `json:"done_regex,omitempty"`         // if set, the reply is finished when this appears below the prompt
	ReplyStart         string   `json:"reply_start,omitempty"`        // the reply begins after the last line matching this (default: the typed prompt)
	ReplyEnd           string   `json:"reply_end,omitempty"`          // the reply ends before the first line matching this
	Drop               []string `json:"drop,omitempty"`               // reply lines matching any of these are removed
	NoPaste            bool     `json:"no_paste,omitempty"`           // type the prompt as plain keys on one line, not as a bracketed paste
	Submit             string   `json:"submit,omitempty"`             // keys that send the prompt (default Enter)
	Quit               string   `json:"quit,omitempty"`               // keys that close the app (default Ctrl-C twice)
	Transcript         string   `json:"transcript,omitempty"`         // glob of the app's saved-conversation files; ~ and {scratch_base} expand. When set, the reply is read from there and the screen is the fallback
	TranscriptKeep     bool     `json:"transcript_keep,omitempty"`    // leave the saved conversation on disk (default: delete it after reading)
	Cols               int      `json:"cols,omitempty"`
	Rows               int      `json:"rows,omitempty"`
	Disabled           bool     `json:"disabled,omitempty"` // listed in the file but not served
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

// Parse reads a providers.json document, leaving out disabled entries.
func Parse(data []byte) ([]Provider, error) { return parse(data, false) }

// ParseAll is Parse including disabled entries, for trying one out before turning it on.
func ParseAll(data []byte) ([]Provider, error) { return parse(data, true) }

func parse(data []byte, all bool) ([]Provider, error) {
	var f struct {
		Preamble  string   `json:"preamble"`
		Providers []Config `json:"providers"`
	}
	if err := json.Unmarshal(data, &f); err != nil {
		return nil, err
	}
	for i, c := range f.Providers {
		if c.Preamble == nil && c.Kind != "openai" {
			f.Providers[i].Preamble = &f.Preamble
		}
		for _, g := range c.Cleanup {
			if !strings.Contains(g, "{scratch_base}") {
				return nil, fmt.Errorf("%s: cleanup %q must contain {scratch_base}, so it can only remove this run's leftovers", c.ID, g)
			}
		}
	}
	var out []Provider
	seen := map[string]bool{}
	for _, c := range f.Providers {
		if c.ID == "" || strings.ContainsAny(c.ID, "/ ") || seen[c.ID] {
			return nil, fmt.Errorf("provider id %q is empty, repeated, or contains a slash or space", c.ID)
		}
		seen[c.ID] = true
		if c.Disabled && !all {
			continue
		}
		switch c.Kind {
		case "openai":
			if c.URL == "" {
				return nil, fmt.Errorf("%s: url is required", c.ID)
			}
			p := &openAI{cfg: c, client: &http.Client{}}
			if c.ModelFilter != "" {
				re, err := regexp.Compile(c.ModelFilter)
				if err != nil {
					return nil, fmt.Errorf("%s: model_filter: %w", c.ID, err)
				}
				p.filter = re
			}
			out = append(out, p)
		case "cli":
			if c.Bin == "" || !hasPrompt(c.AskArgs) {
				return nil, fmt.Errorf("%s: bin and ask_args with {prompt} are required", c.ID)
			}
			p := &cli{cfg: c}
			for _, st := range c.Strip {
				re, err := regexp.Compile(st)
				if err != nil {
					return nil, fmt.Errorf("%s: strip: %w", c.ID, err)
				}
				p.strip = append(p.strip, re)
			}
			out = append(out, p)
		case "tty":
			if c.Bin == "" {
				return nil, fmt.Errorf("%s: bin is required", c.ID)
			}
			p, err := newTTY(c)
			if err != nil {
				return nil, err
			}
			out = append(out, p)
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

	filter *regexp.Regexp

	mu      sync.Mutex
	models  []string
	fetched time.Time
}

func (p *openAI) Config() Config { return p.cfg }

// KeyLookup finds a provider's API key. The default reads only the environment variable; the
// gateway command replaces it with one that also reads the key file. Keys are never in config.
var KeyLookup = func(id, env string) string { return os.Getenv(env) }

func (p *openAI) key() string {
	if p.cfg.KeyEnv == "" {
		return ""
	}
	return strings.TrimSpace(KeyLookup(p.cfg.ID, p.cfg.KeyEnv))
}

func (p *openAI) account() string {
	if p.cfg.AccountEnv == "" {
		return ""
	}
	return strings.TrimSpace(KeyLookup(p.cfg.ID+".account", p.cfg.AccountEnv))
}

// url fills the account id into an address that carries one (Cloudflare).
func (p *openAI) url(u string) string { return strings.ReplaceAll(u, "{account}", p.account()) }

func (p *openAI) Available() error {
	if (p.cfg.KeyEnv != "" && !p.cfg.KeyOptional && p.key() == "") || (p.cfg.AccountEnv != "" && p.account() == "") {
		return fmt.Errorf("no API key set (run: ai-gateway key set %s)", p.cfg.ID)
	}
	return nil
}

// notChat matches model ids that are not chat models, for lists that do not say.
var notChat = regexp.MustCompile(`(?i)whisper|embed|tts|guard|rerank|moderation|speech|transcri|bge-|ocr|image|video|audio`)

func (p *openAI) Models(ctx context.Context) []string {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.cfg.ModelsURL != "" && time.Since(p.fetched) > 10*time.Minute {
		p.fetched = time.Now() // also after a failure, so a dead list is not refetched on every call
		cctx, cancel := context.WithTimeout(ctx, 15*time.Second)
		if code, body, err := HTTPDo(cctx, p.client, http.MethodGet, p.url(p.cfg.ModelsURL), nil, p.key()); err == nil && code == 200 {
			ids := slices.DeleteFunc(ModelIDs(body), func(id string) bool {
				return notChat.MatchString(id) || (p.filter != nil && !p.filter.MatchString(id))
			})
			if len(ids) > 0 {
				p.models = ids
			}
		}
		cancel()
	}
	return order(p.cfg.DefaultModel, p.models, p.cfg.StaticModels)
}

func (p *openAI) Ask(ctx context.Context, r Request) (string, error) {
	req := map[string]any{"model": r.Model, "messages": r.Messages}
	if p.cfg.MaxTokens > 0 {
		req["max_tokens"] = p.cfg.MaxTokens
	}
	body, _ := json.Marshal(req)
	key := p.key()
	code, out, err := HTTPDo(ctx, p.client, http.MethodPost, p.url(p.cfg.URL), body, key)
	if err != nil {
		return "", err
	}
	if code != 200 {
		msg := string(out)
		if key != "" { // a provider must never be able to echo the key into a log or a reply
			msg = strings.ReplaceAll(msg, key, "[key]")
		}
		return "", fmt.Errorf("HTTP %d: %s", code, Short(msg))
	}
	text, ok := ChatText(out)
	if !ok || strings.TrimSpace(text) == "" {
		return "", fmt.Errorf("no reply text in: %s", Short(string(out)))
	}
	var fin struct {
		Choices []struct {
			FinishReason string `json:"finish_reason"`
		} `json:"choices"`
	}
	if json.Unmarshal(out, &fin) == nil && len(fin.Choices) > 0 && fin.Choices[0].FinishReason == "length" {
		return "", fmt.Errorf("the reply was cut off at the provider's output limit after %d characters", len(text))
	}
	return strings.TrimSpace(text), nil
}

// ---- vendor CLI, run headless ----

type cli struct {
	cfg   Config
	strip []*regexp.Regexp
}

func (p *cli) Config() Config { return p.cfg }

func (p *cli) Available() error {
	if _, err := exec.LookPath(p.cfg.Bin); err != nil {
		return fmt.Errorf("%s is not installed (or not on PATH)", p.cfg.Bin)
	}
	return nil
}

func (p *cli) Models(context.Context) []string {
	// "default" (the CLI's own choice, no model flag) is offered only when the entry names no model.
	if ms := order(p.cfg.DefaultModel, p.cfg.StaticModels); len(ms) > 0 {
		return ms
	}
	return []string{"default"}
}

var ansi = regexp.MustCompile(`\x1b\[[0-9;?]*[ -/]*[@-~]`)

func (p *cli) Ask(ctx context.Context, r Request) (string, error) {
	path, err := exec.LookPath(p.cfg.Bin)
	if err != nil {
		return "", p.Available()
	}
	prompt := p.cfg.withPreamble(Flatten(r.Messages))
	if len(prompt) > 200_000 {
		return "", errors.New("prompt is too long to pass to a CLI (200,000 characters)")
	}
	// The model flags go after the ask arguments: some CLIs (opencode) take them on the subcommand.
	var args []string
	for _, a := range p.cfg.AskArgs {
		args = append(args, strings.ReplaceAll(a, "{prompt}", prompt))
	}
	if r.Model != "" && r.Model != "default" {
		for _, a := range p.cfg.ModelArgs {
			args = append(args, strings.ReplaceAll(a, "{model}", r.Model))
		}
	}
	var stdout, stderr bytes.Buffer
	cmd := exec.CommandContext(ctx, path, args...)
	cmd.Stdout, cmd.Stderr = &stdout, &stderr // stdin stays closed: never wait on a login prompt
	// These CLIs are coding agents that read and may edit the directory they start in, so each run
	// gets its own empty scratch directory, never a project, and it is deleted afterwards.
	scratch, cleanup, err := newScratch(p.cfg)
	if err != nil {
		return "", err
	}
	defer cleanup()
	cmd.Dir = scratch
	cmd.Env = os.Environ()
	for k, v := range p.cfg.Env {
		cmd.Env = append(cmd.Env, k+"="+v)
	}
	cmd.WaitDelay = 5 * time.Second
	runErr := cmd.Run()
	text := ansi.ReplaceAllString(stdout.String(), "")
	for _, re := range p.strip {
		text = re.ReplaceAllString(text, "")
	}
	text = strings.TrimSpace(text)
	if runErr != nil {
		detail := firstNonEmpty(strings.TrimSpace(stderr.String()), text)
		if why := reportedError(detail); why != "" { // the real cause, ahead of the CLI's generic line
			detail = why + " | " + detail
		}
		return "", fmt.Errorf("%s: %s", Short(runErr.Error()), Short(detail))
	}
	if text == "" {
		return "", fmt.Errorf("empty output: %s", Short(stderr.String()))
	}
	return text, nil
}

// newScratch makes a fresh empty directory for one run of a coding-agent CLI and returns a function
// that deletes it, along with whatever the entry's cleanup globs name for this run.
//
// The directory is new every time because these tools write files there even when asked a plain
// question, and the next tool to run in a shared directory reads them as "the project": answers
// then leak from one task into another.
func newScratch(c Config) (dir string, cleanup func(), err error) {
	base := filepath.Join(os.TempDir(), "ai-gateway-runs")
	if err := os.MkdirAll(base, 0o700); err != nil {
		return "", nil, err
	}
	if dir, err = os.MkdirTemp(base, "aigw-"); err != nil {
		return "", nil, err
	}
	name := filepath.Base(dir)
	return dir, func() {
		os.RemoveAll(dir)
		for _, g := range c.Cleanup {
			matches, _ := filepath.Glob(expandHome(strings.ReplaceAll(g, "{scratch_base}", name)))
			for _, m := range matches {
				if strings.Contains(m, name) { // never anything but this run's leftovers
					os.RemoveAll(m)
				}
			}
		}
	}, nil
}

func expandHome(path string) string {
	if home, err := os.UserHomeDir(); err == nil && strings.HasPrefix(path, "~/") {
		return filepath.Join(home, path[2:])
	}
	return path
}

// withPreamble puts the entry's standing instruction before a prompt.
func (c Config) withPreamble(prompt string) string {
	if c.Preamble == nil || strings.TrimSpace(*c.Preamble) == "" {
		return prompt
	}
	return strings.TrimSpace(*c.Preamble) + "\n\n" + prompt
}

var reportRe = regexp.MustCompile(`Full report available at: (\S+\.json)`)

// reportedError digs the cause out of an error report file. Gemini CLI prints only "Error generating
// content via API" and writes the reason (a spent daily quota, say) to a JSON report in the temp
// directory; without it the gateway cannot tell a quota from a fault.
func reportedError(stderr string) string {
	m := reportRe.FindStringSubmatch(stderr)
	if m == nil || !strings.HasPrefix(filepath.Base(m[1]), "gemini-client-error-") {
		return ""
	}
	f, err := os.Open(m[1])
	if err != nil {
		return ""
	}
	defer f.Close()
	b, _ := io.ReadAll(io.LimitReader(f, 1<<20))
	var v struct {
		Error struct {
			Message string `json:"message"`
		} `json:"error"`
	}
	json.Unmarshal(b, &v)
	return v.Error.Message
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

func HTTPDo(ctx context.Context, cl *http.Client, method, u string, body []byte, key string) (int, []byte, error) {
	req, err := http.NewRequestWithContext(ctx, method, u, bytes.NewReader(body))
	if err != nil {
		return 0, nil, err
	}
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	req.Header.Set("User-Agent", "ai-gateway/0.2")
	if key != "" {
		req.Header.Set("Authorization", "Bearer "+key)
	}
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
//
// Where the list says so, it keeps only models a keyless caller can use for chat: an entry marked
// usage_based_only (paid per token, so it wants an API key) or with a model_type other than chat is
// left out. LLM7's list carries both fields. Entries that report their availability over the last
// hour are ordered most available first.
func ModelIDs(body []byte) []string {
	type m struct {
		ID           string   `json:"id"`
		Name         string   `json:"name"`
		UsageOnly    bool     `json:"usage_based_only"`
		ModelType    string   `json:"model_type"`
		Availability *float64 `json:"availability_last_hour_percent"`
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
	list = slices.DeleteFunc(list, func(x m) bool {
		return x.UsageOnly || (x.ModelType != "" && x.ModelType != "chat")
	})
	slices.SortStableFunc(list, func(a, b m) int {
		if a.Availability == nil || b.Availability == nil {
			return 0
		}
		return cmp.Compare(*b.Availability, *a.Availability)
	})
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

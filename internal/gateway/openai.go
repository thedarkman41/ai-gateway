package gateway

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"regexp"
	"strconv"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/thedarkman41/ai-gateway/internal/provider"
)

// The HTTP interface for a client that is a program, not a model: an OpenAI-compatible chat
// endpoint, and builds. Bill the Cat is that client. Its delegate_code tool posts a coding task to
// its "coder" model, and pointing it here turns every hand-off into a vote; its delegate_build
// tool drives the build endpoints.
//
// The shapes here are a contract shared with Bill (phase12-gateway-contract.md, v1). Change them
// only together with Bill.
//
//	POST /v1/chat/completions   model "coder", "consensus" or "vote": a vote; any other model: one answer
//	POST /v1/builds             start a build
//	GET  /v1/builds/{id}?wait=N where it stands, waiting up to N seconds for it to finish
//	POST /v1/builds/{id}/rework send failed parts back through the vote
//	DELETE /v1/builds/{id}      cancel
//	GET  /v1/models
//
// A request with the header "X-AI-Gateway-Scope: local" is local-only: no part of it is shown to
// a model outside the user's machines, and every response to it carries the same header back.

const (
	scopeHeader = "X-AI-Gateway-Scope"
	// voteWithin bounds how long a vote waits for its agents. Bill gives its coder a fixed time,
	// so a slow agent is dropped and the vote is decided among those that have answered.
	voteWithin    = 150 * time.Second
	maxStatusLine = 300 // characters
	maxBuildWait  = 120 // seconds
)

// keepAliveEvery is how often a comment line is sent on a stream that has no data yet, so the
// client can tell a slow vote from a dead service. The contract asks for at least every 15 s.
var keepAliveEvery = 10 * time.Second

var (
	voteModels = map[string]bool{"coder": true, "consensus": true, "vote": true}
	taskIDRe   = regexp.MustCompile(`^[A-Za-z0-9_-]{1,40}$`)
)

type chatRequest struct {
	Model    string `json:"model"`
	Stream   bool   `json:"stream"`
	Messages []struct {
		Role    string `json:"role"`
		Content any    `json:"content"` // a string, or a list of {type: "text", text: "..."} parts
	} `json:"messages"`
}

// OpenAIHandler serves everything under /v1/.
func (g *Gateway) OpenAIHandler() http.Handler {
	mux := http.NewServeMux()
	only := func(method string, h http.HandlerFunc) http.HandlerFunc {
		return func(w http.ResponseWriter, r *http.Request) {
			if r.Method != method {
				apiError(w, http.StatusMethodNotAllowed, "method_not_allowed", method+" only")
				return
			}
			h(w, r)
		}
	}
	mux.HandleFunc("/v1/models", only(http.MethodGet, g.listModels))
	mux.HandleFunc("/v1/chat/completions", only(http.MethodPost, g.chat))
	mux.HandleFunc("/v1/builds", only(http.MethodPost, g.createBuild))
	mux.HandleFunc("/v1/builds/{id}/rework", only(http.MethodPost, g.reworkBuild))
	mux.HandleFunc("/v1/builds/{id}", func(w http.ResponseWriter, r *http.Request) {
		switch r.Method {
		case http.MethodGet:
			g.getBuild(w, r)
		case http.MethodDelete:
			g.deleteBuild(w, r)
		default:
			apiError(w, http.StatusMethodNotAllowed, "method_not_allowed", "GET or DELETE only")
		}
	})
	mux.HandleFunc("/v1/", func(w http.ResponseWriter, _ *http.Request) {
		apiError(w, http.StatusNotFound, "not_found", "no such endpoint")
	})
	return mux
}

// HealthHandler answers /healthz the way Bill the Cat's check reads it (LoadoutHealth.check):
// healthy only on HTTP 200 with a JSON body whose "ok" is true. When no agent can take work it
// answers 503 with "ok": false and a short "state", which Bill shows as the reason.
func (g *Gateway) HealthHandler() http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// A local-only probe gets the scope header back like any other scoped request: Bill checks
		// here, before sending anything private, that this gateway honours the scope at all.
		if strings.EqualFold(strings.TrimSpace(r.Header.Get(scopeHeader)), "local") {
			w.Header().Set(scopeHeader, "local")
		}
		if len(g.Free()) == 0 {
			writeJSON(w, http.StatusServiceUnavailable, map[string]any{"ok": false, "state": "no agent is free: all are resting, out of quota or not installed"})
			return
		}
		writeJSON(w, http.StatusOK, map[string]any{"ok": true})
	})
}

// scope reads the scope header. A scoped request gets the header echoed on whatever is answered.
// ok is false when the value is not one the gateway knows; the error has then been sent.
func scope(w http.ResponseWriter, r *http.Request) (local, ok bool) {
	switch v := strings.TrimSpace(r.Header.Get(scopeHeader)); {
	case v == "":
		return false, true
	case strings.EqualFold(v, "local"):
		w.Header().Set(scopeHeader, "local")
		return true, true
	default:
		apiError(w, http.StatusBadRequest, "bad_scope", fmt.Sprintf("%s must be \"local\" or absent, not %q", scopeHeader, v))
		return false, false
	}
}

func (g *Gateway) listModels(w http.ResponseWriter, _ *http.Request) {
	type model struct {
		ID     string `json:"id"`
		Object string `json:"object"`
	}
	list := []model{{"coder", "model"}, {"auto", "model"}}
	for _, id := range g.ids() {
		list = append(list, model{id, "model"})
	}
	writeJSON(w, http.StatusOK, map[string]any{"object": "list", "data": list})
}

func (g *Gateway) chat(w http.ResponseWriter, r *http.Request) {
	local, ok := scope(w, r)
	if !ok {
		return
	}
	body, err := io.ReadAll(io.LimitReader(r.Body, 8<<20))
	var req chatRequest
	if err != nil || json.Unmarshal(body, &req) != nil || len(req.Messages) == 0 {
		apiError(w, http.StatusBadRequest, "bad_request", "messages: the body must be a chat completion request with messages")
		return
	}
	var system, turns []string
	var history []provider.Message
	for _, m := range req.Messages {
		text := contentText(m.Content)
		if strings.TrimSpace(text) == "" {
			continue
		}
		if m.Role == "system" || m.Role == "developer" {
			system = append(system, text)
			continue
		}
		role := "user"
		if m.Role == "assistant" {
			role = "assistant"
		} else {
			turns = append(turns, text)
		}
		history = append(history, provider.Message{Role: role, Content: text})
	}
	if len(turns) == 0 {
		apiError(w, http.StatusBadRequest, "bad_request", "messages: no user message")
		return
	}
	vote := voteModels[req.Model]
	if !vote {
		if _, _, err := g.resolve(req.Model); err != nil {
			apiError(w, http.StatusNotFound, "unknown_model", oneLine(err.Error())+` (or "coder" for a vote)`)
			return
		}
	}
	ctx := r.Context() // cancelled when the client disconnects, which stops the outstanding agent calls
	if local {
		ctx = WithLocalOnly(ctx)
	}
	sys, prompt := strings.Join(system, "\n\n"), strings.Join(turns, "\n\n")

	// Whatever can be refused is refused now, while a proper HTTP status can still be sent.
	private := g.private(ctx, sys, prompt) != ""
	switch {
	case private && len(g.ranked(true)) == 0:
		apiProblem(w, g.noAgent(true))
		return
	case !private && vote && len(g.ranked(false)) == 0:
		apiProblem(w, g.noAgent(false))
		return
	}
	run := func() (string, *Problem) {
		if vote {
			v, err := g.Consensus(ctx, Request{Prompt: prompt, System: sys, Within: voteWithin})
			if err != nil {
				return "", asProblem(err, "vote_failed", private)
			}
			return voteReply(v), nil
		}
		ans, err := g.Ask(ctx, req.Model, sys, provider.Flatten(history), "")
		if err != nil {
			return "", asProblem(err, "no_provider", private)
		}
		return ans.Text, nil
	}

	id := fmt.Sprintf("aigw-%d", time.Now().UnixNano())
	usage := map[string]int{"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
	if !req.Stream {
		text, prob := run()
		if prob != nil {
			apiProblem(w, prob)
			return
		}
		writeJSON(w, http.StatusOK, map[string]any{"id": id, "object": "chat.completion", "created": time.Now().Unix(), "model": req.Model, "usage": usage,
			"choices": []map[string]any{{"index": 0, "finish_reason": "stop", "message": map[string]string{"role": "assistant", "content": text}}}})
		return
	}

	// Streamed: the headers go out before the work starts, and comment lines keep the connection
	// visibly alive until there is something to send. A failure from here on is an error event.
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.WriteHeader(http.StatusOK)
	flush := func() {
		if f, ok := w.(http.Flusher); ok {
			f.Flush()
		}
	}
	flush()
	type result struct {
		text string
		prob *Problem
	}
	done := make(chan result, 1)
	go func() {
		text, prob := run()
		done <- result{text, prob}
	}()
	tick := time.NewTicker(keepAliveEvery)
	defer tick.Stop()
	event := func(v any) {
		b, _ := json.Marshal(v)
		fmt.Fprintf(w, "data: %s\n\n", b)
	}
	for {
		select {
		case <-tick.C:
			fmt.Fprint(w, ": keep-alive\n\n")
			flush()
		case <-ctx.Done():
			return
		case res := <-done:
			if res.prob != nil {
				event(map[string]any{"error": map[string]string{"code": streamCode(res.prob.Code), "message": res.prob.Message}})
			} else {
				chunk := func(delta map[string]any, finish any, extra map[string]any) {
					c := map[string]any{"id": id, "object": "chat.completion.chunk", "created": time.Now().Unix(), "model": req.Model,
						"choices": []map[string]any{{"index": 0, "delta": delta, "finish_reason": finish}}}
					for k, v := range extra {
						c[k] = v
					}
					event(c)
				}
				chunk(map[string]any{"role": "assistant", "content": res.text}, nil, nil)
				chunk(map[string]any{}, "stop", map[string]any{"usage": usage})
			}
			fmt.Fprint(w, "data: [DONE]\n\n")
			flush()
			return
		}
	}
}

// The two ways a vote's status line can open. Bill tells the cases apart by these exact prefixes.
const (
	statusMajority   = "[ai-gateway vote: majority"
	statusNoMajority = "[ai-gateway vote: no majority"
)

// voteReply is a vote's outcome as a coder's answer: the status line, one empty line, the body.
//
// When there is one answer to use, the line opens with statusMajority and the body is that code in
// one fence. That is a true majority, and also the case where every agent that answered gave the
// same thing, however few answered: a local-only request with one local model is "1 of 1". The
// counts in the line say which.
//
// When the answers differ and none has a majority, the line opens with statusNoMajority and the
// body carries every candidate.
func voteReply(v Verdict) string {
	lang := ""
	if defRe.MatchString(v.Winner()) {
		lang = "python"
	}
	var status, body string
	if v.Majority || len(v.Groups) == 1 {
		status = fmt.Sprintf("majority, %d of %d agree (%s); %s", len(v.Groups[0].By), v.Answers, strings.Join(v.Groups[0].By, ", "), v.Method)
		body = fmt.Sprintf("```%s\n%s\n```\n", lang, v.Winner())
	} else {
		status = fmt.Sprintf("no majority among %d answers (%s); the %d candidates follow, strongest group first", v.Answers, v.Method, len(v.Groups))
		var b strings.Builder
		for _, d := range v.Diffs {
			fmt.Fprintf(&b, "Where they differ: %s\n", d)
		}
		for i, gr := range v.Groups {
			fmt.Fprintf(&b, "\n### candidate %d, from %s\n\n```%s\n%s\n```\n", i+1, strings.Join(gr.By, ", "), lang, gr.Text)
		}
		body = strings.TrimLeft(b.String(), "\n")
	}
	if note := v.PrivacyNote(); note != "" {
		status += ". Privacy: " + note
	}
	return statusLine("vote: "+status) + "\n\n" + body
}

// statusLine wraps text as the one line a client can find and strip: it starts with "[ai-gateway "
// and ends with "]", holds no line break, and is at most 300 characters.
func statusLine(text string) string {
	const open, shut = "[ai-gateway ", "]"
	text = strings.NewReplacer("\r", " ", "\n", " ", "]", ")", "[", "(").Replace(text)
	if room := maxStatusLine - utf8.RuneCountInString(open+shut); utf8.RuneCountInString(text) > room {
		text = string([]rune(text)[:room-1]) + "…"
	}
	return open + text + shut
}

// ---- builds ----

func (g *Gateway) createBuild(w http.ResponseWriter, r *http.Request) {
	local, ok := scope(w, r)
	if !ok {
		return
	}
	body, err := io.ReadAll(http.MaxBytesReader(w, r.Body, 8<<20))
	if err != nil {
		apiError(w, http.StatusRequestEntityTooLarge, "too_large", "the request body is over 8 MB")
		return
	}
	var req struct {
		Language string `json:"language"`
		Header   string `json:"header"`
		Footer   string `json:"footer"`
		Context  string `json:"context"`
		System   string `json:"system"`
		Tasks    []Task `json:"tasks"`
	}
	bad := func(format string, a ...any) {
		apiError(w, http.StatusBadRequest, "bad_request", fmt.Sprintf(format, a...))
	}
	if err := json.Unmarshal(body, &req); err != nil {
		bad("body: not a JSON object of the expected shape (%v)", err)
		return
	}
	if strings.TrimSpace(req.Language) == "" {
		bad("language: required, e.g. \"python\"")
		return
	}
	if len(req.Tasks) == 0 || len(req.Tasks) > maxTasks {
		bad("tasks: a build takes 1 to %d tasks, got %d", maxTasks, len(req.Tasks))
		return
	}
	seen := map[string]bool{}
	for i, t := range req.Tasks {
		switch {
		case !taskIDRe.MatchString(t.ID):
			bad("tasks[%d].id: must be 1 to 40 characters of A-Z, a-z, 0-9, _ or -", i)
			return
		case seen[t.ID]:
			bad("tasks[%d].id: %q is used twice", i, t.ID)
			return
		case strings.TrimSpace(t.Prompt) == "":
			bad("tasks[%d].prompt: required", i)
			return
		}
		seen[t.ID] = true
	}
	id, _, err := g.Build(BuildSpec{Tasks: req.Tasks, System: req.System, Language: req.Language, Header: req.Header, Footer: req.Footer, Context: req.Context, LocalOnly: local})
	if err != nil {
		apiProblem(w, asProblem(err, "no_provider", local))
		return
	}
	writeJSON(w, http.StatusAccepted, map[string]any{"build_id": id, "status": "running", "tasks": len(req.Tasks)})
}

func (g *Gateway) getBuild(w http.ResponseWriter, r *http.Request) {
	if _, ok := scope(w, r); !ok {
		return
	}
	wait := 0
	if s := r.URL.Query().Get("wait"); s != "" {
		n, err := strconv.Atoi(s)
		if err != nil || n < 0 {
			apiError(w, http.StatusBadRequest, "bad_request", "wait: a whole number of seconds from 0 to 120")
			return
		}
		wait = min(n, maxBuildWait)
	}
	v, err := g.View(r.Context(), r.PathValue("id"), time.Duration(wait)*time.Second)
	if err != nil {
		if r.Context().Err() != nil {
			return // the client went away
		}
		apiProblem(w, asProblem(err, "unknown_build", false))
		return
	}
	if v.LocalOnly { // a scoped build says so on every answer about it
		w.Header().Set(scopeHeader, "local")
	}
	writeJSON(w, http.StatusOK, v)
}

func (g *Gateway) reworkBuild(w http.ResponseWriter, r *http.Request) {
	if _, ok := scope(w, r); !ok {
		return
	}
	id := r.PathValue("id")
	v, err := g.View(r.Context(), id, 0)
	if err != nil {
		apiProblem(w, asProblem(err, "unknown_build", false))
		return
	}
	if v.LocalOnly { // a build keeps the scope it was created with, header or no header
		w.Header().Set(scopeHeader, "local")
	}
	body, _ := io.ReadAll(io.LimitReader(r.Body, 8<<20))
	var req struct {
		Failures []struct {
			ID    string `json:"id"`
			Error string `json:"error"`
		} `json:"failures"`
	}
	if json.Unmarshal(body, &req) != nil || len(req.Failures) == 0 {
		apiError(w, http.StatusBadRequest, "bad_request", "failures: a non-empty list of {id, error}")
		return
	}
	failed := map[string]string{}
	var ids []string
	for _, f := range req.Failures {
		failed[f.ID] = f.Error
	}
	for _, p := range v.Parts { // in task order
		if _, ok := failed[p.ID]; ok {
			ids = append(ids, p.ID)
		}
	}
	if _, err := g.Rework(id, failed); err != nil {
		apiProblem(w, asProblem(err, "bad_request", false))
		return
	}
	writeJSON(w, http.StatusAccepted, map[string]any{"build_id": id, "status": "running", "reworking": ids})
}

func (g *Gateway) deleteBuild(w http.ResponseWriter, r *http.Request) {
	if _, ok := scope(w, r); !ok {
		return
	}
	id := r.PathValue("id")
	v, err := g.View(r.Context(), id, 0)
	if err != nil {
		apiProblem(w, asProblem(err, "unknown_build", false))
		return
	}
	if v.LocalOnly {
		w.Header().Set(scopeHeader, "local")
	}
	g.Cancel(id)
	w.WriteHeader(http.StatusNoContent)
}

// ---- helpers ----

// asProblem turns any error into one a client can act on. fallback is the code for an error that
// carries none; a failure on a local-only request is reported as local.
func asProblem(err error, fallback string, local bool) *Problem {
	var p *Problem
	if errors.As(err, &p) {
		return &Problem{Status: p.Status, Code: p.Code, Message: oneLine(p.Message), RetryAfter: p.RetryAfter}
	}
	code := fallback
	if local {
		code = "local_failed"
	}
	status := http.StatusServiceUnavailable
	if errors.Is(err, context.DeadlineExceeded) {
		code = "timeout"
	}
	return &Problem{Status: status, Code: code, Message: oneLine(err.Error())}
}

// streamCode maps a problem's code onto the four an error event may carry.
func streamCode(code string) string {
	switch code {
	case "vote_failed", "no_provider", "timeout", "local_failed":
		return code
	case "no_local_provider":
		return "local_failed"
	case "quota":
		return "no_provider"
	}
	return "vote_failed"
}

// oneLine makes an error message one sentence on one line.
func oneLine(s string) string {
	s = strings.Join(strings.Fields(s), " ")
	if r := []rune(s); len(r) > 600 {
		s = string(r[:600]) + "…"
	}
	return s
}

func contentText(c any) string {
	switch v := c.(type) {
	case string:
		return v
	case []any:
		var parts []string
		for _, p := range v {
			if m, ok := p.(map[string]any); ok {
				if t, ok := m["text"].(string); ok {
					parts = append(parts, t)
				}
			}
		}
		return strings.Join(parts, "\n")
	}
	return ""
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(v)
}

func apiError(w http.ResponseWriter, status int, code, msg string) {
	writeJSON(w, status, map[string]any{"error": map[string]any{"code": code, "message": msg}})
}

func apiProblem(w http.ResponseWriter, p *Problem) {
	e := map[string]any{"code": p.Code, "message": p.Message}
	if p.RetryAfter > 0 {
		e["retry_after_seconds"] = p.RetryAfter
		w.Header().Set("Retry-After", strconv.Itoa(p.RetryAfter))
	}
	writeJSON(w, p.Status, map[string]any{"error": e})
}

package gateway

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/thedarkman41/ai-gateway/internal/provider"
)

// The OpenAI-compatible endpoint. It lets a client that already speaks chat/completions use the
// gateway as if it were one model server. Bill the Cat's delegate_code tool is such a client: it
// posts a coding task to its "coder" model, and pointing it here turns every hand-off into a vote.
//
//	model "coder" (also "consensus", "vote"): the strongest free agents each write the code, and the
//	    version most of them agree on is the reply.
//	model "auto", a provider id, or "provider/model": one answer through the usual routing.

// voteWithin bounds how long a vote waits for its agents. Bill gives its coder a fixed time to
// answer, so a slow agent is dropped and the vote is decided among those that have answered.
const voteWithin = 150 * time.Second

var voteModels = map[string]bool{"coder": true, "consensus": true, "vote": true}

type chatRequest struct {
	Model    string `json:"model"`
	Stream   bool   `json:"stream"`
	Messages []struct {
		Role    string `json:"role"`
		Content any    `json:"content"` // a string, or a list of {type: "text", text: "..."} parts
	} `json:"messages"`
}

// OpenAIHandler serves POST /v1/chat/completions and GET /v1/models.
func (g *Gateway) OpenAIHandler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("/v1/models", func(w http.ResponseWriter, _ *http.Request) {
		type model struct {
			ID     string `json:"id"`
			Object string `json:"object"`
		}
		list := []model{{"coder", "model"}, {"auto", "model"}}
		for _, id := range g.ids() {
			list = append(list, model{id, "model"})
		}
		writeJSON(w, http.StatusOK, map[string]any{"object": "list", "data": list})
	})
	mux.HandleFunc("/v1/chat/completions", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			apiError(w, http.StatusMethodNotAllowed, "POST only")
			return
		}
		body, err := io.ReadAll(io.LimitReader(r.Body, 8<<20))
		var req chatRequest
		if err != nil || json.Unmarshal(body, &req) != nil || len(req.Messages) == 0 {
			apiError(w, http.StatusBadRequest, "the body must be a chat completion request with messages")
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
		if len(history) == 0 {
			apiError(w, http.StatusBadRequest, "no user message")
			return
		}

		var text string
		if voteModels[req.Model] {
			// The whole work is done before the first byte is sent, so a failure is still a proper
			// HTTP error and not a stream that ends early.
			v, err := g.Consensus(r.Context(), Request{Prompt: strings.Join(turns, "\n\n"), System: strings.Join(system, "\n\n"), Within: voteWithin})
			if err != nil {
				apiError(w, http.StatusServiceUnavailable, err.Error())
				return
			}
			text = voteReply(v)
		} else {
			if _, _, err := g.resolve(req.Model); err != nil {
				apiError(w, http.StatusNotFound, err.Error()+` (or "coder" for a vote)`)
				return
			}
			ans, err := g.Ask(r.Context(), req.Model, strings.Join(system, "\n\n"), provider.Flatten(history), "")
			if err != nil {
				apiError(w, http.StatusServiceUnavailable, err.Error())
				return
			}
			text = ans.Text
		}

		id := fmt.Sprintf("aigw-%d", time.Now().UnixNano())
		usage := map[string]int{"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
		if !req.Stream {
			writeJSON(w, http.StatusOK, map[string]any{"id": id, "object": "chat.completion", "created": time.Now().Unix(), "model": req.Model, "usage": usage,
				"choices": []map[string]any{{"index": 0, "finish_reason": "stop", "message": map[string]string{"role": "assistant", "content": text}}}})
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		chunk := func(delta map[string]any, finish any, extra map[string]any) {
			c := map[string]any{"id": id, "object": "chat.completion.chunk", "created": time.Now().Unix(), "model": req.Model,
				"choices": []map[string]any{{"index": 0, "delta": delta, "finish_reason": finish}}}
			for k, v := range extra {
				c[k] = v
			}
			b, _ := json.Marshal(c)
			fmt.Fprintf(w, "data: %s\n\n", b)
		}
		chunk(map[string]any{"role": "assistant", "content": text}, nil, nil)
		chunk(map[string]any{}, "stop", map[string]any{"usage": usage})
		fmt.Fprint(w, "data: [DONE]\n\n")
		if f, ok := w.(http.Flusher); ok {
			f.Flush()
		}
	})
	return mux
}

// voteReply is a vote's outcome as a coder's answer: one line saying how it was chosen, then the
// code in a fence. With no majority the reply is still one piece of code, the largest group's, and
// says so: the caller builds and tests it either way.
func voteReply(v Verdict) string {
	how := v.How() + "; " + v.Method
	if !v.Majority {
		how = fmt.Sprintf("no majority among %d answers (%s); this is the version from %s", v.Answers, v.Method, strings.Join(v.Groups[0].By, ", "))
	}
	lang := ""
	if defRe.MatchString(v.Winner()) {
		lang = "python"
	}
	if note := v.PrivacyNote(); note != "" {
		how += ". Privacy: " + note
	}
	return fmt.Sprintf("[ai-gateway vote: %s]\n\n```%s\n%s\n```\n", how, lang, v.Winner())
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

func apiError(w http.ResponseWriter, status int, msg string) {
	writeJSON(w, status, map[string]any{"error": map[string]any{"message": msg, "type": "ai_gateway_error"}})
}

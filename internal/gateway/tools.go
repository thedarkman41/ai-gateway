package gateway

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"

	"github.com/thedarkman41/ai-gateway/internal/mcp"
)

const maxFanOut = 6

// Tools returns the gateway's MCP tools.
func (g *Gateway) Tools() []mcp.Tool {
	str := func(desc string) map[string]any { return map[string]any{"type": "string", "description": desc} }
	modelDesc := "\"auto\" (default), a provider id, or \"provider/model\". Call list_models for the ids."
	return []mcp.Tool{
		{
			Name:        "list_models",
			Description: "List the other AI models this gateway can reach, with each provider's current state and calls used today.",
			InputSchema: map[string]any{"type": "object", "properties": map[string]any{}},
			Annotations: map[string]any{"readOnlyHint": true, "openWorldHint": false},
			Handler: func(ctx context.Context, _ map[string]any) (string, error) {
				return g.Status(ctx), nil
			},
		},
		{
			Name: "ask",
			Description: "Ask another AI model a question and return its reply. The prompt is sent to a free third-party service, " +
				"so include everything the model needs and nothing private. Pass the same session id to continue a conversation.",
			InputSchema: map[string]any{"type": "object", "required": []string{"prompt"}, "properties": map[string]any{
				"prompt":  str("The question or task, self-contained."),
				"model":   str(modelDesc),
				"system":  str("Optional instructions for the model."),
				"session": str("Optional id; calls sharing it form one conversation."),
			}},
			Annotations: map[string]any{"readOnlyHint": false, "openWorldHint": true},
			Handler: func(ctx context.Context, a map[string]any) (string, error) {
				ans, err := g.Ask(ctx, arg(a, "model"), arg(a, "system"), arg(a, "prompt"), arg(a, "session"))
				if err != nil {
					return "", err
				}
				return fmt.Sprintf("[answered by %s]\n\n%s", ans.Used, ans.Text), nil
			},
		},
		{
			Name: "ask_many",
			Description: fmt.Sprintf("Ask the same question of several models at once (2 to %d) and return every reply, to compare them. "+
				"The prompt is sent to free third-party services.", maxFanOut),
			InputSchema: map[string]any{"type": "object", "required": []string{"prompt", "models"}, "properties": map[string]any{
				"prompt": str("The question or task, self-contained."),
				"models": map[string]any{"type": "array", "items": map[string]any{"type": "string"}, "description": "Provider ids or \"provider/model\" ids."},
				"system": str("Optional instructions for the models."),
			}},
			Annotations: map[string]any{"readOnlyHint": false, "openWorldHint": true},
			Handler: func(ctx context.Context, a map[string]any) (string, error) {
				var models []string
				if list, ok := a["models"].([]any); ok {
					for _, m := range list {
						if s, ok := m.(string); ok && strings.TrimSpace(s) != "" {
							models = append(models, s)
						}
					}
				}
				if len(models) < 2 || len(models) > maxFanOut {
					return "", fmt.Errorf("models must list 2 to %d ids", maxFanOut)
				}
				parts := make([]string, len(models))
				failed := 0
				var mu sync.Mutex
				var wg sync.WaitGroup
				for i, m := range models {
					wg.Add(1)
					go func() {
						defer wg.Done()
						ans, err := g.Ask(ctx, m, arg(a, "system"), arg(a, "prompt"), "")
						if err != nil {
							mu.Lock()
							failed++
							mu.Unlock()
							parts[i] = fmt.Sprintf("## %s (failed)\n\n%v", m, err)
							return
						}
						parts[i] = fmt.Sprintf("## %s\n\n%s", ans.Used, ans.Text)
					}()
				}
				wg.Wait()
				out := strings.Join(parts, "\n\n")
				if failed == len(models) {
					return "", errors.New(out)
				}
				return out, nil
			},
		},
	}
}

func arg(a map[string]any, k string) string {
	s, _ := a[k].(string)
	return s
}

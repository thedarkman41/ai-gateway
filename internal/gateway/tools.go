package gateway

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"sync"
	"time"

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
		{
			Name: "dispatch",
			Description: fmt.Sprintf("Fan a plan out: hand up to %d tasks to other AI models at once and get a job id back immediately. "+
				"The gateway spreads the tasks over whichever providers are free and retries a failed task on another. "+
				"Each prompt goes to a free third-party service and must be self-contained: the worker sees nothing but that prompt, "+
				"so include the code and context it needs. Then call collect with the job id.", maxTasks),
			InputSchema: map[string]any{"type": "object", "required": []string{"tasks"}, "properties": map[string]any{
				"tasks": map[string]any{"type": "array", "description": "The tasks, independent of each other.", "items": map[string]any{
					"type": "object", "required": []string{"id", "prompt"}, "properties": map[string]any{
						"id":     str("Short name for the task, unique in the job."),
						"prompt": str("The whole task, self-contained."),
						"system": str("Optional instructions for the worker."),
						"copies": map[string]any{"type": "integer", "description": fmt.Sprintf("How many different providers answer this task, 1 to %d (default 1). More than 1 gives rival answers to judge between; use it only for the hard tasks.", maxCopies)},
						"models": map[string]any{"type": "array", "items": map[string]any{"type": "string"}, "description": "Optional: name who answers (provider ids or provider/model). Leave out to let the gateway choose."},
					}}},
			}},
			Annotations: map[string]any{"readOnlyHint": false, "openWorldHint": true},
			Handler: func(_ context.Context, a map[string]any) (string, error) {
				raw, _ := json.Marshal(a["tasks"])
				var tasks []Task
				if err := json.Unmarshal(raw, &tasks); err != nil {
					return "", fmt.Errorf("tasks must be a list of {id, prompt}: %v", err)
				}
				_, plan, err := g.Dispatch(tasks)
				return plan, err
			},
		},
		{
			Name: "collect",
			Description: "Get results of a dispatched job as they finish. Returns the next result not yet collected (waiting up to wait_seconds for one) " +
				"and says how many are still running. Judge each result when it arrives, then call collect again until the job is complete.",
			InputSchema: map[string]any{"type": "object", "required": []string{"job"}, "properties": map[string]any{
				"job":          str("The job id dispatch returned."),
				"wait_seconds": map[string]any{"type": "integer", "description": "How long to wait for a result when none is ready (default 60, at most 240)."},
				"max_results":  map[string]any{"type": "integer", "description": "How many results to return at once (default 1, at most 4)."},
			}},
			Annotations: map[string]any{"readOnlyHint": true, "openWorldHint": false},
			Handler: func(ctx context.Context, a map[string]any) (string, error) {
				wait, _ := a["wait_seconds"].(float64)
				if wait <= 0 {
					wait = 60
				}
				n, _ := a["max_results"].(float64)
				return g.Collect(ctx, arg(a, "job"), time.Duration(min(wait, 240))*time.Second, int(min(max(n, 1), 4)))
			},
		},
	}
}

func arg(a map[string]any, k string) string {
	s, _ := a[k].(string)
	return s
}

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
			Name: "consensus",
			Description: fmt.Sprintf("For ONE function: have the %d strongest free agents each write it independently, and return the version most of them agree on. "+
				"Python answers are run on the same test calls and count as agreeing when they return the same results; other languages are compared as text. "+
				"With no majority it returns the distinct candidates for you to choose between. The prompt goes to third-party services and must be "+
				"self-contained: give the exact signature, the language, and any code the function must call.", defaultVoters),
			InputSchema: map[string]any{"type": "object", "required": []string{"prompt"}, "properties": map[string]any{
				"prompt":  str("The function to write, self-contained: signature, language, behaviour."),
				"system":  str("Optional instructions for the agents."),
				"agents":  map[string]any{"type": "integer", "description": fmt.Sprintf("How many agents vote (default %d, at most %d).", defaultVoters, maxVoters)},
				"tests":   map[string]any{"type": "array", "items": map[string]any{"type": "array"}, "description": "Optional, Python only: test calls to compare the answers on, each a list of arguments, e.g. [[2],[15],[97]]. Left out, a model writes some."},
				"context": str("Optional, Python only: code the function relies on (imports, helper functions), run before each answer is tested."),
			}},
			Annotations: map[string]any{"readOnlyHint": false, "openWorldHint": true},
			Handler: func(ctx context.Context, a map[string]any) (string, error) {
				n, _ := a["agents"].(float64)
				ctx, cancel := context.WithTimeout(ctx, 4*time.Minute)
				defer cancel()
				var tests [][]any
				if list, ok := a["tests"].([]any); ok {
					for _, t := range list {
						if args, ok := t.([]any); ok {
							tests = append(tests, args)
						}
					}
				}
				v, err := g.Consensus(ctx, Request{Prompt: arg(a, "prompt"), System: arg(a, "system"), Agents: int(n), Context: arg(a, "context"), Tests: tests})
				if err != nil {
					return "", err
				}
				return v.String(), nil
			},
		},
		{
			Name: "build",
			Description: fmt.Sprintf("For a program of SEVERAL functions: put up to %d tasks, one function each, in a queue. Every free agent takes a task and takes "+
				"another when it finishes, so faster agents write more. Returns a build id at once; call collect for the assembled program. "+
				"Each prompt goes to a third-party service and must be self-contained: the agent sees only that prompt. Decide the "+
				"signatures yourself and state them in every task that defines or calls them, so the parts fit.", maxTasks),
			InputSchema: map[string]any{"type": "object", "required": []string{"tasks"}, "properties": map[string]any{
				"tasks": map[string]any{"type": "array", "description": "One task per function, in the order the code should appear in the program.", "items": map[string]any{
					"type": "object", "required": []string{"id", "prompt"}, "properties": map[string]any{
						"id":     str("The function's name, unique in the build."),
						"prompt": str("The function to write, self-contained: signature, language, behaviour, and the signatures of anything it calls."),
					}}},
				"header": str("Optional code placed before the functions: imports, constants, shared types."),
				"footer": str("Optional code placed after the functions: the main block."),
				"system": str("Optional instructions for every agent, such as the language and style."),
			}},
			Annotations: map[string]any{"readOnlyHint": false, "openWorldHint": true},
			Handler: func(_ context.Context, a map[string]any) (string, error) {
				raw, _ := json.Marshal(a["tasks"])
				var tasks []Task
				if err := json.Unmarshal(raw, &tasks); err != nil {
					return "", fmt.Errorf("tasks must be a list of {id, prompt}: %v", err)
				}
				_, plan, err := g.Build(tasks, arg(a, "system"), arg(a, "header"), arg(a, "footer"))
				return plan, err
			},
		},
		{
			Name: "collect",
			Description: "Wait for a build (or its rework) to finish and get the whole program, assembled in task order, with who wrote each part. " +
				"While it is still running this returns the progress: call it again. Then check the program yourself: compile it, run its tests.",
			InputSchema: map[string]any{"type": "object", "required": []string{"build"}, "properties": map[string]any{
				"build":        str("The build id that build returned."),
				"wait_seconds": map[string]any{"type": "integer", "description": "How long to wait for it to finish (default 120, at most 240)."},
			}},
			Annotations: map[string]any{"readOnlyHint": true, "openWorldHint": false},
			Handler: func(ctx context.Context, a map[string]any) (string, error) {
				wait, _ := a["wait_seconds"].(float64)
				if wait <= 0 {
					wait = 120
				}
				return g.Collect(ctx, arg(a, "build"), time.Duration(min(wait, 240))*time.Second)
			},
		},
		{
			Name: "rework",
			Description: fmt.Sprintf("After checking a build: send back the functions that failed. Each is rewritten by a vote of the %d strongest free agents, "+
				"who see the error and the rejected code, and the majority's version replaces it. Then call collect for the program again.", defaultVoters),
			InputSchema: map[string]any{"type": "object", "required": []string{"build", "failed"}, "properties": map[string]any{
				"build": str("The build id."),
				"failed": map[string]any{"type": "array", "description": "The parts that failed your check.", "items": map[string]any{
					"type": "object", "required": []string{"id", "error"}, "properties": map[string]any{
						"id":    str("The task id of the function that failed."),
						"error": str("The compiler or test output that shows what is wrong."),
					}}},
			}},
			Annotations: map[string]any{"readOnlyHint": false, "openWorldHint": true},
			Handler: func(_ context.Context, a map[string]any) (string, error) {
				raw, _ := json.Marshal(a["failed"])
				var list []struct{ ID, Error string }
				if err := json.Unmarshal(raw, &list); err != nil || len(list) == 0 {
					return "", errors.New("failed must be a list of {id, error}")
				}
				failed := map[string]string{}
				for _, f := range list {
					failed[strings.TrimSpace(f.ID)] = f.Error
				}
				return g.Rework(arg(a, "build"), failed)
			},
		},
	}
}

func arg(a map[string]any, k string) string {
	s, _ := a[k].(string)
	return s
}

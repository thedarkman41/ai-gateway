package gateway

import (
	"context"
	_ "embed"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/thedarkman41/ai-gateway/internal/provider"
)

// Comparing answers as text finds no agreement between agents that all wrote a correct function in
// their own style. So for Python, answers are compared by what they do: every candidate is run on
// the same test calls, and candidates that return the same results are counted as agreeing.
//
// The candidates are code written by other models, so they run in a sandbox: no network, no writes
// outside a temporary directory, no reading the user's home directory.

//go:embed harness.py
var harness string

const (
	candidateTimeout = 25 * time.Second
	inputsTimeout    = 45 * time.Second
	maxTestCalls     = 40
)

var defRe = regexp.MustCompile(`(?m)^(?:async\s+)?def\s+([A-Za-z_]\w*)\s*\(([^)]*)\)`)

// isPython says whether most of the answers are Python functions.
func isPython(votes []Vote) bool {
	n := 0
	for _, v := range votes {
		if defRe.MatchString(v.Text) {
			n++
		}
	}
	return n*2 > len(votes)
}

// target picks the function the candidates were asked for: the public top-level name most of them
// define, preferring one the prompt mentions. It also returns how many arguments it takes.
func target(votes []Vote, prompt string) (name string, arity int) {
	count, args, order := map[string]int{}, map[string]int{}, []string{}
	for _, v := range votes {
		seen := map[string]bool{}
		for _, m := range defRe.FindAllStringSubmatch(v.Text, -1) {
			if strings.HasPrefix(m[1], "_") || seen[m[1]] {
				continue
			}
			seen[m[1]] = true
			if count[m[1]] == 0 {
				order = append(order, m[1])
				n := 0
				for _, a := range strings.Split(m[2], ",") {
					if a = strings.TrimSpace(a); a != "" && !strings.HasPrefix(a, "*") && !strings.Contains(a, "=") {
						n++
					}
				}
				args[m[1]] = n
			}
			count[m[1]]++
		}
	}
	sort.SliceStable(order, func(a, b int) bool {
		if count[order[a]] != count[order[b]] {
			return count[order[a]] > count[order[b]]
		}
		return strings.Contains(prompt, order[a]) && !strings.Contains(prompt, order[b])
	})
	if len(order) == 0 {
		return "", 0
	}
	return order[0], args[order[0]]
}

// defaultInputs are plain integer calls for a function of the given arity, used when no model
// supplied test calls. They suit numeric functions; for others every candidate raises, the calls
// carry no information, and the vote falls back to comparing text.
func defaultInputs(arity int) [][]any {
	pool := []any{-7, -1, 0, 1, 2, 3, 4, 5, 9, 10, 12, 17, 25, 97, 100, 7919}
	var out [][]any
	switch arity {
	case 0:
		out = [][]any{{}}
	case 1:
		for _, v := range pool {
			out = append(out, []any{v})
		}
	default:
		small := []any{-1, 0, 1, 2, 3, 10}
		for i, a := range small {
			for j, b := range small {
				call := []any{a, b}
				for k := 2; k < arity; k++ {
					call = append(call, small[(i+j+k)%len(small)])
				}
				out = append(out, call)
			}
		}
	}
	return out
}

// testInputs asks one fast model for test calls that fit the task. Any API provider will do; the
// first that answers with a usable list wins.
func (g *Gateway) testInputs(ctx context.Context, prompt string) [][]any {
	ctx, cancel := context.WithTimeout(ctx, inputsTimeout)
	defer cancel()
	ask := "Here is a task for a Python function:\n\n" + prompt +
		"\n\nWrite 12 test calls for it: typical cases and edge cases, with valid arguments only. " +
		"Reply with only a JSON array. Each element is the list of positional arguments for one call. " +
		"Example for a function f(a, b): [[1, 2], [0, 0], [-3, 7]]"
	tried := 0
	for _, p := range g.ranked(false) { // Ask itself keeps a private prompt to local models
		if p.Config().Kind != "openai" || tried >= 3 || ctx.Err() != nil {
			continue
		}
		tried++
		ans, err := g.Ask(ctx, p.Config().ID, "", ask, "")
		if err != nil {
			continue
		}
		text := ans.Text
		i, j := strings.Index(text, "["), strings.LastIndex(text, "]")
		if i < 0 || j < i {
			continue
		}
		var raw []any
		if json.Unmarshal([]byte(text[i:j+1]), &raw) != nil || len(raw) == 0 {
			continue
		}
		var calls [][]any
		for _, r := range raw {
			if args, ok := r.([]any); ok {
				calls = append(calls, args)
			} else {
				calls = append(calls, []any{r}) // a model that listed bare values for a one-argument function
			}
		}
		return calls[:min(len(calls), maxTestCalls)]
	}
	return nil
}

// outcome is what one candidate did on the test calls.
type outcome struct {
	Calls    []string `json:"calls"`
	Outcomes []string `json:"outcomes"`
	Error    string   `json:"error"` // it could not be loaded at all
}

// behaviour runs every candidate on the same calls. contextCode is run first in each (the imports
// and helper functions a part of a larger program relies on).
func behaviour(ctx context.Context, votes []Vote, name, contextCode string, inputs [][]any, exprs []string) ([]outcome, error) {
	dir, err := os.MkdirTemp("", "aigw-vote-")
	if err != nil {
		return nil, err
	}
	defer os.RemoveAll(dir)
	if real, err := filepath.EvalSymlinks(dir); err == nil {
		dir = real
	}
	if inputs == nil {
		inputs = [][]any{}
	}
	if exprs == nil {
		exprs = []string{}
	}
	want := len(inputs)
	if len(exprs) > 0 { // expressions replace argument lists
		want = len(exprs)
	}
	in, _ := json.Marshal(map[string]any{"args": inputs, "exprs": exprs})
	files := map[string]string{"harness.py": harness, "inputs.json": string(in), "context.py": contextCode}
	for i, v := range votes {
		files[fmt.Sprintf("cand%d.py", i)] = v.Text
	}
	for f, body := range files {
		if err := os.WriteFile(filepath.Join(dir, f), []byte(body), 0o600); err != nil {
			return nil, err
		}
	}
	run, err := sandbox(dir)
	if err != nil {
		return nil, err
	}
	outs := make([]outcome, len(votes))
	var wg sync.WaitGroup
	for i := range votes {
		wg.Add(1)
		go func() {
			defer wg.Done()
			cctx, cancel := context.WithTimeout(ctx, candidateTimeout)
			defer cancel()
			cmd := run(cctx, "harness.py", fmt.Sprintf("cand%d.py", i), "inputs.json", name, "context.py")
			cmd.Dir = dir
			cmd.Env = []string{"PATH=/usr/bin:/bin", "HOME=" + dir, "TMPDIR=" + dir, "PYTHONDONTWRITEBYTECODE=1"}
			var stderr strings.Builder
			cmd.Stderr = &stderr
			b, err := cmd.Output()
			if json.Unmarshal(lastLine(b), &outs[i]) != nil || (outs[i].Error == "" && len(outs[i].Outcomes) != want) {
				why := "produced no result"
				switch {
				case cctx.Err() != nil:
					why = "ran past the time limit"
				case strings.TrimSpace(stderr.String()) != "":
					why = provider.Short(stderr.String())
				case err != nil:
					why = err.Error()
				}
				outs[i] = outcome{Error: "could not be run: " + why}
			}
		}()
	}
	wg.Wait()
	// If nothing ran at all, the fault is the runner (no Python, a sandbox that will not start), not the candidates.
	for _, o := range outs {
		if !strings.HasPrefix(o.Error, "could not be run") {
			return outs, nil
		}
	}
	return nil, errors.New(outs[0].Error)
}

func lastLine(b []byte) []byte {
	lines := strings.Split(strings.TrimSpace(string(b)), "\n")
	return []byte(lines[len(lines)-1])
}

// sandbox returns a function that builds the command to run Python on files in dir, confined.
// On macOS that is sandbox-exec. Elsewhere there is no confinement to offer, so it refuses unless
// AI_GATEWAY_UNSANDBOXED=1 says the machine is disposable (a test container).
func sandbox(dir string) (func(ctx context.Context, args ...string) *exec.Cmd, error) {
	python := ""
	for _, p := range []string{"/usr/bin/python3", "/usr/local/bin/python3", "/opt/homebrew/bin/python3"} {
		if _, err := os.Stat(p); err == nil {
			python = p
			break
		}
	}
	if python == "" {
		p, err := exec.LookPath("python3")
		if err != nil {
			return nil, errors.New("python3 was not found")
		}
		python = p
	}
	base := []string{python, "-I", "-S", "-B"}
	if runtime.GOOS == "darwin" {
		const tool = "/usr/bin/sandbox-exec"
		if _, err := os.Stat(tool); err != nil {
			return nil, errors.New("sandbox-exec is missing, so candidates cannot be run safely")
		}
		home, _ := os.UserHomeDir()
		if real, err := filepath.EvalSymlinks(home); err == nil {
			home = real
		}
		profile := fmt.Sprintf(`(version 1)
(allow default)
(deny network*)
(deny file-write*)
(allow file-write* (subpath %q) (literal "/dev/null"))
(deny file-read* (subpath %q))
(allow file-read* (subpath %q))`, dir, home, dir)
		if real, err := filepath.EvalSymlinks(python); err == nil && strings.HasPrefix(real, home+"/") {
			// A Python that lives in the home directory (conda, pyenv): let it read its own installation.
			profile += fmt.Sprintf("\n(allow file-read* (subpath %q))", filepath.Dir(filepath.Dir(real)))
		}
		base = append([]string{tool, "-p", profile}, base...)
	} else if os.Getenv("AI_GATEWAY_UNSANDBOXED") != "1" {
		return nil, fmt.Errorf("no sandbox is available on %s, so candidates are not run (set AI_GATEWAY_UNSANDBOXED=1 on a disposable machine)", runtime.GOOS)
	}
	return func(ctx context.Context, args ...string) *exec.Cmd {
		all := append(append([]string{}, base[1:]...), args...)
		cmd := exec.CommandContext(ctx, base[0], all...)
		cmd.WaitDelay = 2 * time.Second
		return cmd
	}, nil
}

// behaviourKeys turns the outcomes into one key per candidate: candidates with equal keys behave the
// same. Only calls on which some candidate returned a value count; a call every candidate rejects
// says nothing about which is right. informative is how many calls counted.
func behaviourKeys(outs []outcome) (keys []string, informative []int) {
	n := 0
	for _, o := range outs {
		n = max(n, len(o.Outcomes))
	}
	for c := 0; c < n; c++ {
		for _, o := range outs {
			if c < len(o.Outcomes) && !strings.HasPrefix(o.Outcomes[c], "raises ") && o.Outcomes[c] != "timed out" {
				informative = append(informative, c)
				break
			}
		}
	}
	for _, o := range outs {
		if o.Error != "" {
			keys = append(keys, "!broken: "+o.Error)
			continue
		}
		var b strings.Builder
		for _, c := range informative {
			b.WriteString(o.Outcomes[c])
			b.WriteByte('\x00')
		}
		keys = append(keys, b.String())
	}
	return keys, informative
}

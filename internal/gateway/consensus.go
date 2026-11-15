package gateway

import (
	"context"
	"errors"
	"fmt"
	"os"
	"regexp"
	"slices"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/thedarkman41/ai-gateway/internal/provider"
)

// Consensus: one function, written independently by the strongest agents that are free, and the
// answer most of them agree on wins.

const (
	defaultVoters  = 5
	maxVoters      = 7
	slotRetries    = 3 // other agents a seat tries when its agent fails
	codeOnlySuffix = "\n\nReply with only the code: no explanation, no usage example, no tests."
)

// ranked lists the providers that can take work now, strongest first (file order breaks ties).
func (g *Gateway) ranked() []provider.Provider {
	var free []provider.Provider
	for _, p := range g.providers {
		cfg := p.Config()
		if p.Available() == nil && g.ledger.Blocked(cfg.ID, cfg.DailyLimit, g.now()) == "" {
			free = append(free, p)
		}
	}
	sort.SliceStable(free, func(a, b int) bool { return free[a].Config().Strength > free[b].Config().Strength })
	return free
}

// Vote is one agent's answer.
type Vote struct {
	Provider string
	Used     string
	Text     string // the code, with any prose and fences removed
	Seconds  float64
}

// Group is one distinct answer and who gave it.
type Group struct {
	Text string
	By   []string
}

// Verdict is the outcome of a vote.
type Verdict struct {
	Groups   []Group  // largest first; ties go to the group with the strongest agent
	Answers  int      // how many agents answered
	Majority bool     // Groups[0] is more than half the answers, and at least three answered
	Failed   []string // "agent: why" for those that did not answer
	Method   string   // how the answers were compared
	Diffs    []string // for each other group, a test call where it parts from the winner
}

// Winner is the answer to use: the majority's, else the largest group's.
func (v Verdict) Winner() string { return v.Groups[0].Text }

// How names the way the winner was chosen, for a build's record.
func (v Verdict) How() string {
	if v.Majority {
		return fmt.Sprintf("majority, %d of %d agree (%s)", len(v.Groups[0].By), v.Answers, strings.Join(v.Groups[0].By, ", "))
	}
	return fmt.Sprintf("no majority among %d answers; used the largest group, %d (%s)", v.Answers, len(v.Groups[0].By), strings.Join(v.Groups[0].By, ", "))
}

// Request is one vote.
type Request struct {
	Prompt  string
	System  string
	Agents  int           // how many vote (default 5)
	Context string        // code the function relies on: imports and helpers, run before each candidate
	Tests   [][]any       // test calls to compare the candidates on; empty = ask a model for some
	Within  time.Duration // stop waiting for agents after this long and decide among those that answered; 0 = wait for all
}

// Consensus asks the strongest free agents for the same function. Each seat moves on to the next
// strongest agent if its own fails, so the full number of answers comes back when enough are free.
func (g *Gateway) Consensus(ctx context.Context, r Request) (Verdict, error) {
	system, prompt, n := r.System, r.Prompt, r.Agents
	if strings.TrimSpace(prompt) == "" {
		return Verdict{}, errors.New("prompt is empty")
	}
	if n <= 0 {
		n = defaultVoters
	}
	n = min(n, maxVoters)
	pool := g.ranked()
	if len(pool) == 0 {
		return Verdict{}, errors.New("no agent is free right now; see list_models")
	}
	rank := map[string]int{}
	for i, p := range pool {
		rank[p.Config().ID] = i
	}

	var mu sync.Mutex
	next := 0
	var votes []Vote
	var v Verdict
	take := func() string {
		mu.Lock()
		defer mu.Unlock()
		if next >= len(pool) {
			return ""
		}
		next++
		return pool[next-1].Config().ID
	}
	// While the agents write, a fast model writes test calls to compare their answers on.
	inputs := make(chan [][]any, 1)
	switch {
	case len(r.Tests) > 0:
		inputs <- r.Tests[:min(len(r.Tests), maxTestCalls)]
	case strings.Contains(strings.ToLower(prompt+" "+system), "python"):
		go func() { inputs <- g.testInputs(ctx, prompt) }()
	default:
		inputs <- nil
	}

	askCtx := ctx // the agents' deadline; comparing their answers afterwards still runs under ctx
	if r.Within > 0 {
		var cancel context.CancelFunc
		askCtx, cancel = context.WithTimeout(ctx, r.Within)
		defer cancel()
	}
	var wg sync.WaitGroup
	for seat := 0; seat < min(n, len(pool)); seat++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for try := 0; try <= slotRetries && askCtx.Err() == nil; try++ {
				id := take()
				if id == "" {
					return
				}
				start := time.Now()
				ans, err := g.Ask(askCtx, id, system, prompt+codeOnlySuffix, "")
				mu.Lock()
				if err == nil {
					votes = append(votes, Vote{Provider: id, Used: ans.Used, Text: codeOnly(ans.Text), Seconds: time.Since(start).Seconds()})
					mu.Unlock()
					return
				}
				why := provider.Short(firstLine(err.Error()))
				if askCtx.Err() != nil && ctx.Err() == nil {
					why = fmt.Sprintf("had not answered after %s", r.Within)
				}
				v.Failed = append(v.Failed, id+": "+why)
				mu.Unlock()
			}
		}()
	}
	wg.Wait()
	if len(votes) == 0 {
		return v, fmt.Errorf("no agent answered:\n- %s", strings.Join(v.Failed, "\n- "))
	}

	// Group the answers. Python functions are grouped by what they do on the same test calls; anything
	// else, or Python that could not be run, by its text with comments, blank lines and spacing set aside.
	keys := make([]string, len(votes))
	for i, vt := range votes {
		keys[i] = normalize(vt.Text)
	}
	v.Method = "compared as text"
	var outs []outcome
	var informative []int
	if isPython(votes) && os.Getenv("AI_GATEWAY_VOTE") != "text" { // that setting turns running candidates off
		name, arity := target(votes, prompt)
		calls := <-inputs
		if len(calls) == 0 {
			calls = defaultInputs(arity)
		}
		var err error
		if outs, err = behaviour(ctx, votes, name, r.Context, calls); err != nil {
			v.Method = "compared as text (the candidates could not be run: " + provider.Short(err.Error()) + ")"
		} else if bk, inf := behaviourKeys(outs); len(inf) == 0 {
			v.Method = "compared as text (no test call gave a result to compare)"
			outs = nil
		} else {
			keys, informative = bk, inf
			v.Method = fmt.Sprintf("compared by behaviour on %d test calls to %s", len(inf), name)
		}
	}
	index := map[string]int{} // provider → position in votes, to find its outcomes later
	for i, vt := range votes {
		index[vt.Provider] = i
	}
	byKey := map[string]int{}
	for vi, vt := range votes {
		key := keys[vi]
		i, ok := byKey[key]
		if !ok {
			i = len(v.Groups)
			byKey[key] = i
			v.Groups = append(v.Groups, Group{Text: vt.Text})
		}
		v.Groups[i].By = append(v.Groups[i].By, vt.Provider)
	}
	best := func(gr Group) int {
		b := len(pool)
		for _, id := range gr.By {
			b = min(b, rank[id])
		}
		return b
	}
	text := map[string]string{}
	for _, vt := range votes {
		text[vt.Provider] = vt.Text
	}
	for i := range v.Groups { // a group's wording is its strongest member's, whoever answered first
		sort.SliceStable(v.Groups[i].By, func(a, b int) bool { return rank[v.Groups[i].By[a]] < rank[v.Groups[i].By[b]] })
		v.Groups[i].Text = text[v.Groups[i].By[0]]
	}
	// Broken: code that does not load, or that returned nothing on any call that counted (it raised
	// or ran out of time every time). It never wins over code that runs.
	broken := func(gr Group) bool {
		if outs == nil {
			return false
		}
		o := outs[index[gr.By[0]]]
		if o.Error != "" {
			return true
		}
		for _, c := range informative {
			if !strings.HasPrefix(o.Outcomes[c], "raises ") && o.Outcomes[c] != "timed out" {
				return false
			}
		}
		return true
	}
	sort.SliceStable(v.Groups, func(a, b int) bool {
		if broken(v.Groups[a]) != broken(v.Groups[b]) {
			return broken(v.Groups[b])
		}
		if len(v.Groups[a].By) != len(v.Groups[b].By) {
			return len(v.Groups[a].By) > len(v.Groups[b].By)
		}
		return best(v.Groups[a]) < best(v.Groups[b])
	})
	v.Answers = len(votes)
	v.Majority = v.Answers >= 3 && len(v.Groups[0].By)*2 > v.Answers && !broken(v.Groups[0])
	if outs != nil { // say where each other group parts from the winner
		win := outs[index[v.Groups[0].By[0]]]
		for i, gr := range v.Groups[1:] {
			o := outs[index[gr.By[0]]]
			if o.Error != "" {
				v.Diffs = append(v.Diffs, fmt.Sprintf("candidate %d does not run: %s", i+2, o.Error))
				continue
			}
			for _, c := range informative {
				if o.Outcomes[c] != win.Outcomes[c] {
					v.Diffs = append(v.Diffs, fmt.Sprintf("%s: candidate 1 gives %s, candidate %d gives %s", win.Calls[c], clip(win.Outcomes[c]), i+2, clip(o.Outcomes[c])))
					break
				}
			}
		}
	}
	if v.Majority { // the record of who sides with the majority shows which agents to trust
		for _, vt := range votes {
			g.ledger.Vote(vt.Provider, slices.Contains(v.Groups[0].By, vt.Provider), g.now())
		}
	}
	return v, nil
}

// String is the verdict as the planner reads it.
func (v Verdict) String() string {
	var b strings.Builder
	if v.Majority {
		fmt.Fprintf(&b, "MAJORITY: %d of %d agents agree (%s); %s.", len(v.Groups[0].By), v.Answers, strings.Join(v.Groups[0].By, ", "), v.Method)
		var others []string
		for _, gr := range v.Groups[1:] {
			others = append(others, gr.By...)
		}
		if len(others) > 0 {
			fmt.Fprintf(&b, " Differed: %s.", strings.Join(others, ", "))
		}
		for _, d := range v.Diffs {
			fmt.Fprintf(&b, "\nWhere they differ: %s", strings.Replace(d, "candidate 1", "the majority", 1))
		}
		fmt.Fprintf(&b, "\n\n%s\n", v.Groups[0].Text)
	} else {
		fmt.Fprintf(&b, "NO MAJORITY among %d answers (%s); the largest group is %d. Compare the candidates and choose: the one that compiles and passes a test, else the simplest.\n", v.Answers, v.Method, len(v.Groups[0].By))
		for _, d := range v.Diffs {
			fmt.Fprintf(&b, "Where they differ: %s\n", d)
		}
		for i, gr := range v.Groups {
			fmt.Fprintf(&b, "\n### candidate %d, from %s\n\n%s\n", i+1, strings.Join(gr.By, ", "), gr.Text)
		}
	}
	if len(v.Failed) > 0 {
		fmt.Fprintf(&b, "\nDid not answer: %s\n", strings.Join(v.Failed, "; "))
	}
	return b.String()
}

var fencedRe = regexp.MustCompile("(?ms)^[ \\t]*```[A-Za-z0-9_+.#-]*[ \\t]*\\r?\\n(.*?)\\r?\\n?^[ \\t]*```[ \\t]*$")

// codeOnly returns the code in a reply: the contents of its fenced blocks, in order, when it has
// any (the prose around them is dropped), else the whole reply.
func codeOnly(text string) string {
	var parts []string
	for _, m := range fencedRe.FindAllStringSubmatch(text, -1) {
		if strings.TrimSpace(m[1]) != "" {
			parts = append(parts, strings.TrimRight(m[1], " \t\r\n"))
		}
	}
	if len(parts) == 0 {
		return strings.TrimSpace(text)
	}
	return strings.Join(parts, "\n\n")
}

var (
	docstringRe = regexp.MustCompile(`(?s)""".*?"""|'''.*?'''`)
	spacesRe    = regexp.MustCompile(`[ \t]+`)
)

// normalize reduces code to what two agents must share to count as agreeing: the statements and
// their indentation. Comments, docstrings, blank lines and spacing inside a line are set aside.
func normalize(code string) string {
	code = docstringRe.ReplaceAllString(code, "")
	var out []string
	for _, line := range strings.Split(strings.ReplaceAll(code, "\t", "    "), "\n") {
		body := strings.TrimSpace(line)
		if body == "" || strings.HasPrefix(body, "#") || strings.HasPrefix(body, "//") {
			continue
		}
		indent := len(line) - len(strings.TrimLeft(line, " "))
		out = append(out, strings.Repeat(" ", indent)+spacesRe.ReplaceAllString(body, " "))
	}
	return strings.Join(out, "\n")
}

func clip(s string) string {
	if len(s) > 80 {
		return s[:80] + "…"
	}
	return s
}

func firstLine(s string) string {
	line, _, _ := strings.Cut(strings.TrimPrefix(s, "no provider answered:\n- "), "\n")
	return line
}

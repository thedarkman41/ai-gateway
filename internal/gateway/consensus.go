package gateway

import (
	"context"
	"errors"
	"fmt"
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

// Consensus asks n of the strongest free agents for the same function. Each seat moves on to the
// next strongest agent if its own fails, so n answers come back when enough agents are free.
func (g *Gateway) Consensus(ctx context.Context, system, prompt string, n int) (Verdict, error) {
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
	var wg sync.WaitGroup
	for seat := 0; seat < min(n, len(pool)); seat++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for try := 0; try <= slotRetries && ctx.Err() == nil; try++ {
				id := take()
				if id == "" {
					return
				}
				start := time.Now()
				ans, err := g.Ask(ctx, id, system, prompt+codeOnlySuffix, "")
				mu.Lock()
				if err == nil {
					votes = append(votes, Vote{Provider: id, Used: ans.Used, Text: codeOnly(ans.Text), Seconds: time.Since(start).Seconds()})
					mu.Unlock()
					return
				}
				v.Failed = append(v.Failed, id+": "+provider.Short(firstLine(err.Error())))
				mu.Unlock()
			}
		}()
	}
	wg.Wait()
	if len(votes) == 0 {
		return v, fmt.Errorf("no agent answered:\n- %s", strings.Join(v.Failed, "\n- "))
	}

	// Group answers that are the same code once comments, blank lines and spacing are set aside.
	byKey := map[string]int{}
	for _, vt := range votes {
		key := normalize(vt.Text)
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
	sort.SliceStable(v.Groups, func(a, b int) bool {
		if len(v.Groups[a].By) != len(v.Groups[b].By) {
			return len(v.Groups[a].By) > len(v.Groups[b].By)
		}
		return best(v.Groups[a]) < best(v.Groups[b])
	})
	v.Answers = len(votes)
	v.Majority = v.Answers >= 3 && len(v.Groups[0].By)*2 > v.Answers
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
		fmt.Fprintf(&b, "MAJORITY: %d of %d agents gave this answer (%s).", len(v.Groups[0].By), v.Answers, strings.Join(v.Groups[0].By, ", "))
		var others []string
		for _, gr := range v.Groups[1:] {
			others = append(others, gr.By...)
		}
		if len(others) > 0 {
			fmt.Fprintf(&b, " Differed: %s.", strings.Join(others, ", "))
		}
		fmt.Fprintf(&b, "\n\n%s\n", v.Groups[0].Text)
	} else {
		fmt.Fprintf(&b, "NO MAJORITY among %d answers; the largest group is %d. Compare the candidates and choose: the one that compiles and passes a test, else the simplest.\n", v.Answers, len(v.Groups[0].By))
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

func firstLine(s string) string {
	line, _, _ := strings.Cut(strings.TrimPrefix(s, "no provider answered:\n- "), "\n")
	return line
}

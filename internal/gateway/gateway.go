// Package gateway routes a question to a provider: it picks the provider and model, spaces calls
// out, falls back when one fails, and keeps the quota ledger and chat sessions.
package gateway

import (
	"context"
	"errors"
	"fmt"
	"regexp"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/thedarkman41/ai-gateway/internal/provider"
)

const (
	maxPrompt        = 400_000 // characters
	maxSessions      = 50
	maxSessionTurns  = 40 // messages kept per session
	modelsTriedUnpin = 3  // models tried per provider when the caller did not name one
	modelAttemptsMax = 8  // the same, counting models skipped at once for needing an API key

	keyedModelRest = 7 * 24 * time.Hour // a model that asked for an API key is left out this long
	busyModelRest  = 2 * time.Minute    // a model that said it was busy
	slowModelRest  = 30 * time.Minute   // a model that ran into the provider's timeout
)

type Gateway struct {
	providers []provider.Provider
	byID      map[string]provider.Provider
	ledger    *Ledger
	now       func() time.Time

	mu       sync.Mutex
	sessions map[string][]provider.Message
	recent   []string // session ids, oldest first
	gates    map[string]*gate
	solo     map[string]chan struct{} // one run at a time for a CLI or terminal app
	jobs     jobs
}

// gate spaces out calls to one provider.
type gate struct {
	mu   sync.Mutex
	last time.Time
}

func New(ps []provider.Provider, l *Ledger) *Gateway {
	g := &Gateway{providers: ps, byID: map[string]provider.Provider{}, ledger: l, now: time.Now,
		sessions: map[string][]provider.Message{}, gates: map[string]*gate{}, solo: map[string]chan struct{}{}}
	for _, p := range ps {
		g.byID[p.Config().ID] = p
		g.gates[p.Config().ID] = &gate{}
		if p.Config().Kind != "openai" {
			g.solo[p.Config().ID] = make(chan struct{}, 1)
		}
	}
	return g
}

// Answer is a reply and the "provider/model" that gave it.
type Answer struct {
	Text string
	Used string
}

// Ask sends prompt to model: "auto" (or empty) tries each provider in order, "provider" uses that
// provider's default model, "provider/model" pins both. A session id continues an earlier
// conversation held under that id.
func (g *Gateway) Ask(ctx context.Context, model, system, prompt, session string) (Answer, error) {
	if strings.TrimSpace(prompt) == "" {
		return Answer{}, errors.New("prompt is empty")
	}
	if len(prompt)+len(system) > maxPrompt {
		return Answer{}, fmt.Errorf("prompt is too long (limit %d characters)", maxPrompt)
	}
	candidates, pinned, err := g.resolve(model)
	if err != nil {
		return Answer{}, err
	}

	var msgs []provider.Message
	if system != "" {
		msgs = append(msgs, provider.Message{Role: "system", Content: system})
	}
	msgs = append(msgs, g.history(session)...)
	msgs = append(msgs, provider.Message{Role: "user", Content: prompt})

	var notes []string
	for _, p := range candidates {
		id := p.Config().ID
		if err := p.Available(); err != nil {
			notes = append(notes, id+": "+err.Error())
			continue
		}
		if why := g.ledger.Blocked(id, p.Config().DailyLimit, g.now()); why != "" {
			notes = append(notes, id+": "+why)
			continue
		}
		models := []string{pinned}
		if pinned == "" {
			// The model that answered last time goes first, so a provider whose default is
			// unavailable is not retried from the top on every call.
			models = p.Models(ctx)
			if last := g.ledger.Snapshot(id, g.now()).LastModel; last != "" && slices.Contains(models, last) {
				models = append([]string{last}, slices.DeleteFunc(slices.Clone(models), func(m string) bool { return m == last })...)
			}
			if len(models) == 0 {
				models = []string{""}
			}
		}
		var lastErr error
		tried, attempts, quotaHits, rested := 0, 0, 0, false
		for _, m := range models {
			name := strings.TrimSuffix(id+"/"+m, "/")
			if why := g.ledger.ModelBlocked(id, m, g.now()); why != "" {
				if pinned != "" {
					notes = append(notes, name+": "+why)
				}
				continue
			}
			if tried >= modelsTriedUnpin || attempts >= modelAttemptsMax {
				break
			}
			attempts++
			text, err := g.call(ctx, p, provider.Request{Model: m, Messages: msgs})
			if err == nil {
				g.ledger.Result(id, m, nil, g.now())
				g.remember(session, prompt, text)
				return Answer{Text: text, Used: name}, nil
			}
			lastErr = err
			notes = append(notes, fmt.Sprintf("%s: %v", name, err))
			if ctx.Err() != nil {
				break
			}
			// Learn from the failure, so the next ask does not repeat it.
			switch classify(err) {
			case needsKey: // not a keyless model: leave it out, and do not count it as a real try
				g.ledger.BlockModel(id, m, g.now().Add(keyedModelRest), "needs an API key", g.now())
				continue
			case quota:
				g.ledger.BlockModel(id, m, nextMidnight(g.now()), "free allowance used up", g.now())
				if quotaHits++; quotaHits >= 2 { // the allowance is the account's, not the model's
					g.ledger.Rest(id, nextMidnight(g.now()), "free allowance used up", g.now())
					rested = true
				}
			case busy:
				g.ledger.BlockModel(id, m, g.now().Add(busyModelRest), "busy", g.now())
			case slow: // do not sit through the same timeout on the next ask
				g.ledger.BlockModel(id, m, g.now().Add(slowModelRest), "timed out", g.now())
			}
			if tried++; rested {
				break
			}
		}
		switch {
		case lastErr == nil:
			notes = append(notes, id+": every model is resting; see list_models")
		case !rested:
			g.ledger.Result(id, "", lastErr, g.now())
		}
		if ctx.Err() != nil {
			break
		}
	}
	return Answer{}, fmt.Errorf("no provider answered:\n- %s", strings.Join(notes, "\n- "))
}

// call makes one attempt, after waiting out the provider's minimum interval.
func (g *Gateway) call(ctx context.Context, p provider.Provider, r provider.Request) (string, error) {
	cfg := p.Config()
	// A CLI or terminal app runs one ask at a time: two copies would share a scratch directory and
	// a login. HTTP endpoints take calls in parallel.
	if lane := g.solo[cfg.ID]; lane != nil {
		select {
		case lane <- struct{}{}:
			defer func() { <-lane }()
		case <-ctx.Done():
			return "", ctx.Err()
		}
	}
	gt := g.gates[cfg.ID]
	gt.mu.Lock()
	wait := time.Duration(cfg.MinIntervalSeconds*float64(time.Second)) - time.Since(gt.last)
	if wait > 0 {
		select {
		case <-time.After(wait):
		case <-ctx.Done():
			gt.mu.Unlock()
			return "", ctx.Err()
		}
	}
	gt.last = time.Now()
	gt.mu.Unlock()

	g.ledger.Attempt(cfg.ID, g.now())
	cctx, cancel := context.WithTimeout(ctx, cfg.Timeout())
	defer cancel()
	text, err := p.Ask(cctx, r)
	if err != nil && cctx.Err() == context.DeadlineExceeded && ctx.Err() == nil {
		err = fmt.Errorf("timed out after %s", cfg.Timeout())
	}
	return text, err
}

func (g *Gateway) resolve(model string) (ps []provider.Provider, pinned string, err error) {
	model = strings.TrimSpace(model)
	if model == "" || model == "auto" {
		return g.providers, "", nil
	}
	id, m, _ := strings.Cut(model, "/")
	p := g.byID[id]
	if p == nil {
		return nil, "", fmt.Errorf("unknown provider %q; use \"auto\" or one of: %s", id, strings.Join(g.ids(), ", "))
	}
	return []provider.Provider{p}, m, nil
}

func (g *Gateway) ids() []string {
	var ids []string
	for _, p := range g.providers {
		ids = append(ids, p.Config().ID)
	}
	return ids
}

func (g *Gateway) history(session string) []provider.Message {
	if session == "" {
		return nil
	}
	g.mu.Lock()
	defer g.mu.Unlock()
	return append([]provider.Message(nil), g.sessions[session]...)
}

func (g *Gateway) remember(session, prompt, reply string) {
	if session == "" {
		return
	}
	g.mu.Lock()
	defer g.mu.Unlock()
	if _, ok := g.sessions[session]; !ok {
		g.recent = append(g.recent, session)
		if len(g.recent) > maxSessions {
			delete(g.sessions, g.recent[0])
			g.recent = g.recent[1:]
		}
	}
	h := append(g.sessions[session], provider.Message{Role: "user", Content: prompt}, provider.Message{Role: "assistant", Content: reply})
	if len(h) > maxSessionTurns {
		h = h[len(h)-maxSessionTurns:]
	}
	g.sessions[session] = h
}

// Status describes every provider: whether it can be called now, today's calls and its models.
func (g *Gateway) Status(ctx context.Context) string {
	var b strings.Builder
	b.WriteString("Model ids: \"auto\" (first provider that answers, in the order below), \"<provider>\", or \"<provider>/<model>\".\n")
	for _, p := range g.providers {
		cfg := p.Config()
		e := g.ledger.Snapshot(cfg.ID, g.now())
		state := "ready"
		if err := p.Available(); err != nil {
			state = "unavailable: " + err.Error()
		} else if why := g.ledger.Blocked(cfg.ID, cfg.DailyLimit, g.now()); why != "" {
			state = why
		} else if e.LastOK.IsZero() {
			state = "ready (not yet used)"
		}
		quota := fmt.Sprintf("%d calls today", e.Calls)
		if cfg.DailyLimit > 0 {
			quota = fmt.Sprintf("%d of %d calls today", e.Calls, cfg.DailyLimit)
		}
		fmt.Fprintf(&b, "\n%s (tier %s): %s; %s\n", cfg.ID, cfg.Tier, state, quota)
		if rest := g.ledger.RestingModels(cfg.ID, g.now()); len(rest) > 0 {
			fmt.Fprintf(&b, "  resting models: %s\n", strings.Join(rest, "; "))
		}
		models := p.Models(ctx)
		shown := models
		if len(shown) > 20 {
			shown = shown[:20]
		}
		if len(models) > 0 {
			fmt.Fprintf(&b, "  models (default first): %s", strings.Join(shown, ", "))
			if len(models) > len(shown) {
				fmt.Fprintf(&b, ", and %d more", len(models)-len(shown))
			}
			b.WriteString("\n")
		}
	}
	return b.String()
}

// CheckResult is one provider's canary outcome.
type CheckResult struct {
	ID      string
	OK      bool
	Used    string
	Detail  string
	Seconds float64
}

// Check sends a canary prompt to every provider in turn.
func (g *Gateway) Check(ctx context.Context) []CheckResult {
	var out []CheckResult
	for _, p := range g.providers {
		id := p.Config().ID
		start := time.Now()
		a, err := g.Ask(ctx, id, "", "Reply with exactly one word: PONG", "")
		r := CheckResult{ID: id, Used: a.Used, Seconds: time.Since(start).Seconds()}
		switch {
		case err != nil:
			r.Detail = strings.TrimPrefix(strings.ReplaceAll(err.Error(), "\n", " "), "no provider answered: - ")
		case !strings.Contains(strings.ToUpper(a.Text), "PONG"):
			r.Detail = "unexpected reply: " + provider.Short(a.Text)
		default:
			r.OK = true
		}
		out = append(out, r)
	}
	return out
}

type failure int

const (
	other failure = iota
	needsKey
	quota
	busy
	slow
)

var (
	keyRe   = regexp.MustCompile(`(?i)api key|HTTP 401|HTTP 403|unauthorized|authentication`)
	quotaRe = regexp.MustCompile(`(?i)quota|usage limit|limit reached|exhausted|out of credits`)
	busyRe  = regexp.MustCompile(`(?i)HTTP 429|HTTP 503|temporarily|busy|overloaded|rate.?limit`)
)

// classify sorts a provider's error by what should be done about it. It reads the error text, since
// a CLI reports these conditions only in words.
func classify(err error) failure {
	msg := err.Error()
	switch {
	case keyRe.MatchString(msg):
		return needsKey
	case quotaRe.MatchString(msg):
		return quota
	case busyRe.MatchString(msg):
		return busy
	case strings.Contains(msg, "timed out after"):
		return slow
	}
	return other
}

// nextMidnight is when daily free allowances are assumed to reset: local midnight.
func nextMidnight(now time.Time) time.Time {
	y, m, d := now.Date()
	return time.Date(y, m, d+1, 0, 0, 0, 0, now.Location())
}

// Package gateway routes a question to a provider: it picks the provider and model, spaces calls
// out, falls back when one fails, and keeps the quota ledger and chat sessions.
package gateway

import (
	"context"
	"errors"
	"fmt"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/thedarkman41/ai-gateway/internal/privacy"
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

	privacy *privacy.Store                // what may leave this machine
	known   func() []string               // secret values the gateway itself holds (its API keys)
	Logf    func(format string, a ...any) // where privacy events are noted; nil = nowhere
}

// SetPrivacy gives the gateway the user's privacy rules and its own secret values to redact.
func (g *Gateway) SetPrivacy(store *privacy.Store, known func() []string) {
	g.privacy, g.known = store, known
}

func (g *Gateway) logf(format string, a ...any) {
	if g.Logf != nil {
		g.Logf(format, a...)
	}
}

// private says why a prompt must stay on this machine, or "". Rules that cannot be read count as a
// reason: when in doubt nothing leaves.
func (g *Gateway) private(ctx context.Context, texts ...string) string {
	if localOnly(ctx) {
		return "the caller asked for local models only"
	}
	pol, err := g.privacy.Policy()
	if err != nil {
		return "the privacy rules could not be read (" + err.Error() + ")"
	}
	return pol.Private(strings.Join(texts, "\n"))
}

// scrub returns the messages as they may be sent to an outside provider, secrets redacted, and a
// summary of what was redacted ("" for nothing).
func (g *Gateway) scrub(msgs []provider.Message) ([]provider.Message, string) {
	pol, _ := g.privacy.Policy()
	var known []string
	if g.known != nil {
		known = g.known()
	}
	total := map[string]int{}
	out := make([]provider.Message, len(msgs))
	for i, m := range msgs {
		text, found := pol.Scrub(m.Content, known)
		out[i] = provider.Message{Role: m.Role, Content: text}
		for k, n := range found {
			total[k] += n
		}
	}
	return out, privacy.Summary(total)
}

// gate spaces out calls to one provider.
type gate struct {
	mu   sync.Mutex
	last time.Time
}

func New(ps []provider.Provider, l *Ledger) *Gateway {
	g := &Gateway{providers: ps, byID: map[string]provider.Provider{}, ledger: l, now: time.Now,
		sessions: map[string][]provider.Message{}, gates: map[string]*gate{}, solo: map[string]chan struct{}{}, privacy: privacy.Open("")}
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
	Text     string
	Used     string
	Redacted string // what was redacted from the prompt before it was sent, e.g. "2 secrets (api-key)"; "" for nothing
	Private  string // why the prompt was kept to local models; "" when it was not
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

	// A prompt that carries private files goes to local models only.
	var texts []string
	for _, m := range msgs {
		texts = append(texts, m.Content)
	}
	private := g.private(ctx, texts...)
	if private != "" {
		var local []provider.Provider
		for _, p := range candidates {
			if p.Config().Local {
				local = append(local, p)
			}
		}
		if len(local) == 0 {
			return Answer{}, &Problem{Status: 403, Code: "no_local_provider", Message: fmt.Sprintf("kept on this machine: %s. Only a local model may see it, and none can take it here", private)}
		}
		candidates = local
	}

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
			send, redacted := msgs, ""
			if !p.Config().Local { // an outside provider never sees a secret
				if send, redacted = g.scrub(msgs); redacted != "" {
					g.logf("privacy: redacted %s before sending to %s", redacted, name)
				}
			}
			text, err := g.call(ctx, p, provider.Request{Model: m, Messages: send})
			if err == nil && cutOff(text) {
				err = fmt.Errorf("the reply was cut off inside a code block after %d characters", len(text))
			}
			if err == nil {
				text = unfence(text)
				g.ledger.Result(id, m, nil, g.now())
				g.remember(session, prompt, text)
				return Answer{Text: text, Used: name, Redacted: redacted, Private: private}, nil
			}
			lastErr = err
			notes = append(notes, fmt.Sprintf("%s: %v", name, err))
			if ctx.Err() != nil {
				break
			}
			// Learn from the failure, so the next ask does not repeat it.
			switch classify(err) {
			case needsKey: // not a keyless model: leave it out, and do not count it as a real try
				if p.Config().KeyEnv != "" && !p.Config().KeyOptional { // our key, not a keyless model, was refused
					g.ledger.Rest(id, g.now().Add(time.Hour), "the API key was refused ("+provider.Short(err.Error())+")", g.now())
					rested = true
					break
				}
				g.ledger.BlockModel(id, m, g.now().Add(keyedModelRest), "needs an API key", g.now())
				continue
			case paid: // not free after all: stop, and keep the whole provider out
				g.ledger.Rest(id, g.now().Add(keyedModelRest), "asks for payment ("+provider.Short(err.Error())+")", g.now())
				rested = true
			case quota:
				until := quotaReset(err.Error(), g.now())
				g.ledger.BlockModel(id, m, until, "free allowance used up", g.now())
				if quotaHits++; quotaHits >= 2 { // the allowance is the account's, not the model's
					g.ledger.Rest(id, until, "free allowance used up ("+provider.Short(err.Error())+")", g.now())
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
		case !rested && ctx.Err() == nil: // a caller that stopped waiting is not the provider's failure
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
		if cfg.Strength > 0 || e.Votes > 0 {
			fmt.Fprintf(&b, "  strength %d", cfg.Strength)
			if e.Votes > 0 {
				fmt.Fprintf(&b, "; sided with the majority in %d of %d votes", e.Agreed, e.Votes)
			}
			b.WriteString("\n")
		}
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

// cutOff reports a reply that stops inside a code block: an odd number of fence lines. That is what
// a provider's output limit or server timeout looks like (Pollinations on betelgeuse, cut at 60
// seconds), and half a file is worse than a failure the gateway can retry elsewhere.
func cutOff(text string) bool {
	fences := 0
	for _, line := range strings.Split(text, "\n") {
		if strings.HasPrefix(strings.TrimSpace(line), "```") {
			fences++
		}
	}
	return fences%2 == 1
}

var fenceRe = regexp.MustCompile("(?s)\\A```[A-Za-z0-9_+.#-]*[ \\t]*\\r?\\n(.*?)\\r?\\n?```\\z")

// unfence removes the Markdown code fence from a reply that is nothing but one fenced block, so
// every provider returns bare code for a "reply with only the code" task. A reply with prose
// around its code, or with several blocks, is left as it is: there the fences carry meaning.
func unfence(text string) string {
	t := strings.TrimSpace(text)
	if m := fenceRe.FindStringSubmatch(t); m != nil && !strings.Contains(m[1], "```") && strings.TrimSpace(m[1]) != "" {
		return m[1]
	}
	return text
}

type failure int

const (
	other failure = iota
	needsKey
	quota
	busy
	slow
	paid
)

var (
	keyRe   = regexp.MustCompile(`(?i)api key|HTTP 401|HTTP 403|unauthorized|authentication`)
	quotaRe = regexp.MustCompile(`(?i)quota|usage limit|limit reached|exhausted|out of credits`)
	paidRe  = regexp.MustCompile(`(?i)HTTP 402|payment method|payment required|billing`)
	rateRe  = regexp.MustCompile(`(?i)HTTP 429|rate.?limit|too many requests|per (second|minute|hour)|try again (in|shortly|later)`)
	dailyRe = regexp.MustCompile(`(?i)daily|per day|today|monthly|per month|free usage limit|out of credits`)
	busyRe  = regexp.MustCompile(`(?i)HTTP 429|HTTP 503|temporarily|busy|overloaded|rate.?limit`)
)

// Reset clears the rests on one provider, or on all of them when id is empty: for a rest the gateway
// got wrong, or an allowance the user knows has come back.
func (g *Gateway) Reset(id string) error {
	if id != "" && g.byID[id] == nil {
		return fmt.Errorf("unknown provider %q; one of: %s", id, strings.Join(g.ids(), ", "))
	}
	for _, p := range g.providers {
		if id == "" || p.Config().ID == id {
			g.ledger.Reset(p.Config().ID, g.now())
		}
	}
	return nil
}

// classify sorts a provider's error by what should be done about it. It reads the error text, since
// a CLI reports these conditions only in words.
func classify(err error) failure {
	msg := err.Error()
	switch {
	case keyRe.MatchString(msg):
		return needsKey
	case paidRe.MatchString(msg):
		return paid
	case rateRe.MatchString(msg) && !dailyRe.MatchString(msg):
		return busy // a short-term rate limit, even when it is worded as a limit or quota
	case quotaRe.MatchString(msg):
		return quota
	case busyRe.MatchString(msg):
		return busy
	case strings.Contains(msg, "timed out after"):
		return slow
	}
	return other
}

var resetsInRe = regexp.MustCompile(`(?i)resets? in\s+((?:\d+\s*d\s*)?(?:\d+\s*h\s*)?(?:\d+\s*m\s*)?(?:\d+(?:\.\d+)?\s*s)?)`)

// quotaReset is when a spent allowance comes back. A provider that says so is believed: Antigravity
// reports "Resets in 113h53m34s" for a weekly allowance, and asking it again every midnight until
// then only wastes calls. Otherwise it is assumed to be daily and to reset at local midnight.
func quotaReset(msg string, now time.Time) time.Time {
	if m := resetsInRe.FindStringSubmatch(msg); m != nil {
		var total time.Duration
		for _, part := range regexp.MustCompile(`(\d+(?:\.\d+)?)\s*([dhms])`).FindAllStringSubmatch(strings.ToLower(m[1]), -1) {
			n, _ := strconv.ParseFloat(part[1], 64)
			total += time.Duration(n * float64(map[string]time.Duration{"d": 24 * time.Hour, "h": time.Hour, "m": time.Minute, "s": time.Second}[part[2]]))
		}
		if total > 0 && total <= 31*24*time.Hour {
			return now.Add(total + time.Minute) // a minute's margin, so the first call after is not early
		}
	}
	return nextMidnight(now)
}

// nextMidnight is when daily free allowances are assumed to reset: local midnight.
func nextMidnight(now time.Time) time.Time {
	y, m, d := now.Date()
	return time.Date(y, m, d+1, 0, 0, 0, 0, now.Location())
}

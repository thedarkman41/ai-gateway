package gateway

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"
)

const (
	breakerFailures = 3                // consecutive failed asks before a provider is rested
	breakerRest     = 10 * time.Minute // how long it rests
)

// Ledger tracks each provider's calls today and its recent health, and persists them so a restart
// (Bill starts a fresh gateway per session) does not forget a quota or a tripped breaker.
type Ledger struct {
	path string
	mu   sync.Mutex
	data ledgerData
}

type ledgerData struct {
	Day       string            `json:"day"`
	Providers map[string]*Entry `json:"providers"`
}

// Block rests one model of a provider.
type Block struct {
	Until time.Time `json:"until"`
	Why   string    `json:"why"`
}

type Entry struct {
	Calls     int              `json:"calls"`               // attempts today
	Fails     int              `json:"fails"`               // consecutive failed asks
	DownUntil time.Time        `json:"down_until,omitzero"` // breaker: resting until then
	LastError string           `json:"last_error,omitempty"`
	LastOK    time.Time        `json:"last_ok,omitzero"`
	LastModel string           `json:"last_model,omitempty"` // the model that last answered
	RestWhy   string           `json:"rest_why,omitempty"`   // why it is resting, when not the breaker
	Models    map[string]Block `json:"resting_models,omitempty"`
}

// OpenLedger loads the ledger at path; a missing or unreadable file starts empty. An empty path
// keeps the ledger in memory only.
func OpenLedger(path string) *Ledger {
	l := &Ledger{path: path, data: ledgerData{Providers: map[string]*Entry{}}}
	if b, err := os.ReadFile(path); err == nil {
		var d ledgerData
		if json.Unmarshal(b, &d) == nil && d.Providers != nil {
			l.data = d
		}
	}
	return l
}

func (l *Ledger) entry(id string, now time.Time) *Entry {
	if day := now.Format("2006-01-02"); l.data.Day != day {
		l.data.Day = day
		for _, e := range l.data.Providers {
			e.Calls = 0
		}
	}
	e := l.data.Providers[id]
	if e == nil {
		e = &Entry{}
		l.data.Providers[id] = e
	}
	return e
}

// Snapshot returns a copy of a provider's entry.
func (l *Ledger) Snapshot(id string, now time.Time) Entry {
	l.mu.Lock()
	defer l.mu.Unlock()
	return *l.entry(id, now)
}

// Blocked says why a provider should not be called now, or "".
func (l *Ledger) Blocked(id string, dailyLimit int, now time.Time) string {
	l.mu.Lock()
	defer l.mu.Unlock()
	e := l.entry(id, now)
	if now.Before(e.DownUntil) {
		if e.RestWhy != "" {
			return fmt.Sprintf("resting until %s: %s", clock(e.DownUntil, now), e.RestWhy)
		}
		return fmt.Sprintf("resting until %s after %d failures in a row (last: %s)", clock(e.DownUntil, now), breakerFailures, e.LastError)
	}
	if dailyLimit > 0 && e.Calls >= dailyLimit {
		return fmt.Sprintf("daily limit reached (%d calls)", dailyLimit)
	}
	return ""
}

// Attempt counts one call against today's quota.
func (l *Ledger) Attempt(id string, now time.Time) {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.entry(id, now).Calls++
	l.save()
}

// Result records how one ask to a provider ended, and on success the model that answered.
func (l *Ledger) Result(id, model string, err error, now time.Time) {
	l.mu.Lock()
	defer l.mu.Unlock()
	e := l.entry(id, now)
	if err == nil {
		e.Fails, e.LastError, e.LastOK, e.DownUntil, e.LastModel, e.RestWhy = 0, "", now, time.Time{}, model, ""
	} else {
		e.Fails++
		e.LastError = err.Error()
		if e.Fails >= breakerFailures {
			e.DownUntil, e.RestWhy = now.Add(breakerRest), ""
			e.Fails = 0
		}
	}
	l.save()
}

// Rest takes a whole provider out of use until a time, for a stated reason.
func (l *Ledger) Rest(id string, until time.Time, why string, now time.Time) {
	l.mu.Lock()
	defer l.mu.Unlock()
	e := l.entry(id, now)
	e.DownUntil, e.RestWhy, e.Fails = until, why, 0
	l.save()
}

// BlockModel takes one model of a provider out of use until a time.
func (l *Ledger) BlockModel(id, model string, until time.Time, why string, now time.Time) {
	l.mu.Lock()
	defer l.mu.Unlock()
	e := l.entry(id, now)
	if e.Models == nil {
		e.Models = map[string]Block{}
	}
	e.Models[model] = Block{Until: until, Why: why}
	l.save()
}

// ModelBlocked says why a model should not be called now, or "".
func (l *Ledger) ModelBlocked(id, model string, now time.Time) string {
	l.mu.Lock()
	defer l.mu.Unlock()
	b, ok := l.entry(id, now).Models[model]
	if !ok || !now.Before(b.Until) {
		return ""
	}
	return fmt.Sprintf("resting until %s: %s", clock(b.Until, now), b.Why)
}

// RestingModels lists a provider's resting models with reasons, grouped by reason.
func (l *Ledger) RestingModels(id string, now time.Time) []string {
	l.mu.Lock()
	defer l.mu.Unlock()
	by := map[string][]string{}
	for m, b := range l.entry(id, now).Models {
		if now.Before(b.Until) {
			by[b.Why] = append(by[b.Why], m)
		}
	}
	var out []string
	for why, ms := range by {
		sort.Strings(ms)
		if len(ms) > 6 {
			ms = append(ms[:6], fmt.Sprintf("and %d more", len(ms)-6))
		}
		out = append(out, why+": "+strings.Join(ms, ", "))
	}
	sort.Strings(out)
	return out
}

// clock formats a time of day, with the date when it is not today.
func clock(t, now time.Time) string {
	if t.Format("2006-01-02") == now.Format("2006-01-02") {
		return t.Format("15:04")
	}
	return t.Format("Jan 2 15:04")
}

func (l *Ledger) save() {
	if l.path == "" {
		return
	}
	b, _ := json.MarshalIndent(l.data, "", "  ")
	tmp := l.path + ".tmp"
	if os.MkdirAll(filepath.Dir(l.path), 0o755) != nil || os.WriteFile(tmp, b, 0o644) != nil {
		return
	}
	os.Rename(tmp, l.path)
}

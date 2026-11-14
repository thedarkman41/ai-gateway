package gateway

import (
	"context"
	"crypto/rand"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

// A job is one fan-out: a list of tasks handed to providers at once, whose results are collected as
// they finish. The planner (Bill) splits the work into tasks, dispatches them, and judges each
// result as it comes back.

const (
	maxTasks       = 12
	maxCopies      = 4
	maxAssignments = 24
	maxJobs        = 20
	jobTimeout     = 30 * time.Minute
	taskRetries    = 2      // other providers tried when the first one fails
	maxResultChars = 60_000 // a longer result is cut, so one reply cannot flood the planner
	jobKeep        = 48 * time.Hour
)

// Task is one unit of work in a job.
type Task struct {
	ID     string   `json:"id"`
	Prompt string   `json:"prompt"`
	System string   `json:"system,omitempty"`
	Models []string `json:"models,omitempty"` // who answers; empty = the gateway chooses
	Copies int      `json:"copies,omitempty"` // how many different providers answer it (default 1)
}

// Result is one provider's answer to one task.
type Result struct {
	Task    string  `json:"task"`
	Used    string  `json:"used,omitempty"`
	OK      bool    `json:"ok"`
	Text    string  `json:"text"`
	Seconds float64 `json:"seconds"`
}

type job struct {
	ID        string    `json:"id"`
	Created   time.Time `json:"created"`
	Total     int       `json:"total"`     // assignments
	Results   []Result  `json:"results"`   // in the order they finished
	Delivered int       `json:"delivered"` // how many of Results collect has handed out
	Running   []string  `json:"running"`   // "task → provider" still in flight

	mu     sync.Mutex
	change chan struct{} // closed and replaced whenever a result arrives
	path   string
	lost   bool // loaded from disk after a restart: whatever was still running is gone
}

type jobs struct {
	mu   sync.Mutex
	byID map[string]*job
	dir  string
	next int // round-robin position, so load spreads across dispatches
}

// SetJobsDir makes finished results survive a restart of the gateway (Bill stops an idle server).
func (g *Gateway) SetJobsDir(dir string) {
	g.jobs.dir = dir
	if entries, err := os.ReadDir(dir); err == nil {
		for _, e := range entries {
			if info, err := e.Info(); err == nil && time.Since(info.ModTime()) > jobKeep {
				os.Remove(filepath.Join(dir, e.Name()))
			}
		}
	}
}

// Free lists the providers that can take work right now, in file order.
func (g *Gateway) Free() []string {
	var free []string
	for _, p := range g.providers {
		cfg := p.Config()
		if p.Available() == nil && g.ledger.Blocked(cfg.ID, cfg.DailyLimit, g.now()) == "" {
			free = append(free, cfg.ID)
		}
	}
	return free
}

// Dispatch starts a job and returns its id and who got which task.
func (g *Gateway) Dispatch(tasks []Task) (id, plan string, err error) {
	if len(tasks) == 0 || len(tasks) > maxTasks {
		return "", "", fmt.Errorf("a job takes 1 to %d tasks", maxTasks)
	}
	// Who can take work right now, in file order.
	var free []string
	var resting []string
	for _, p := range g.providers {
		cfg := p.Config()
		if err := p.Available(); err != nil {
			continue
		}
		if why := g.ledger.Blocked(cfg.ID, cfg.DailyLimit, g.now()); why != "" {
			resting = append(resting, cfg.ID+" ("+why+")")
			continue
		}
		free = append(free, cfg.ID)
	}

	type assignment struct {
		task   Task
		target string
		spare  []string // providers to fall back to; nil when the caller named the target
	}
	var as []assignment
	seen := map[string]bool{}
	g.jobs.mu.Lock()
	for _, t := range tasks {
		t.ID = strings.TrimSpace(t.ID)
		if t.ID == "" || seen[t.ID] || strings.TrimSpace(t.Prompt) == "" {
			g.jobs.mu.Unlock()
			return "", "", fmt.Errorf("every task needs its own id and a prompt (task %q)", t.ID)
		}
		seen[t.ID] = true
		if len(t.Models) > 0 {
			for _, m := range t.Models {
				if _, _, err := g.resolve(m); err != nil {
					g.jobs.mu.Unlock()
					return "", "", fmt.Errorf("task %s: %w", t.ID, err)
				}
				as = append(as, assignment{task: t, target: m})
			}
			continue
		}
		if len(free) == 0 {
			g.jobs.mu.Unlock()
			return "", "", fmt.Errorf("no provider is free right now; resting: %s", strings.Join(resting, "; "))
		}
		copies := min(max(t.Copies, 1), maxCopies, len(free))
		for c := 0; c < copies; c++ {
			at := (g.jobs.next + c) % len(free)
			var spare []string
			for i := copies; i < len(free); i++ { // the providers this task is not already going to
				spare = append(spare, free[(g.jobs.next+i)%len(free)])
			}
			as = append(as, assignment{task: t, target: free[at], spare: spare})
		}
		g.jobs.next = (g.jobs.next + copies) % len(free)
	}
	if len(as) > maxAssignments {
		g.jobs.mu.Unlock()
		return "", "", fmt.Errorf("that is %d assignments; the limit is %d (fewer tasks or copies)", len(as), maxAssignments)
	}

	var b [4]byte
	rand.Read(b[:])
	j := &job{ID: fmt.Sprintf("j%x", b), Created: time.Now(), Total: len(as), change: make(chan struct{})}
	if g.jobs.dir != "" {
		j.path = filepath.Join(g.jobs.dir, j.ID+".json")
	}
	if g.jobs.byID == nil {
		g.jobs.byID = map[string]*job{}
	}
	if len(g.jobs.byID) >= maxJobs { // drop the oldest
		var oldest *job
		for _, o := range g.jobs.byID {
			if oldest == nil || o.Created.Before(oldest.Created) {
				oldest = o
			}
		}
		delete(g.jobs.byID, oldest.ID)
	}
	g.jobs.byID[j.ID] = j
	g.jobs.mu.Unlock()

	var lines []string
	for _, a := range as {
		j.Running = append(j.Running, a.task.ID+" → "+a.target)
		lines = append(lines, a.task.ID+" → "+a.target)
	}
	j.save()

	ctx, cancel := context.WithTimeout(context.Background(), jobTimeout)
	var wg sync.WaitGroup
	for _, a := range as {
		wg.Add(1)
		go func() {
			defer wg.Done()
			start := time.Now()
			ans, err := g.Ask(ctx, a.target, a.task.System, a.task.Prompt, "")
			for i := 0; err != nil && i < taskRetries && i < len(a.spare) && ctx.Err() == nil; i++ {
				ans, err = g.Ask(ctx, a.spare[i], a.task.System, a.task.Prompt, "")
			}
			r := Result{Task: a.task.ID, Used: ans.Used, OK: err == nil, Text: ans.Text, Seconds: time.Since(start).Seconds()}
			if err != nil {
				r.Text = err.Error()
			}
			if len(r.Text) > maxResultChars {
				r.Text = r.Text[:maxResultChars] + fmt.Sprintf("\n\n[cut: the reply was %d characters; ask for it in smaller parts]", len(r.Text))
			}
			j.finish(a.task.ID+" → "+a.target, r)
		}()
	}
	go func() { wg.Wait(); cancel() }()

	plan = fmt.Sprintf("job %s started: %d assignments\n- %s", j.ID, len(as), strings.Join(lines, "\n- "))
	if len(resting) > 0 {
		plan += "\nnot used (resting): " + strings.Join(resting, "; ")
	}
	return j.ID, plan, nil
}

func (j *job) finish(key string, r Result) {
	j.mu.Lock()
	defer j.mu.Unlock()
	j.Results = append(j.Results, r)
	for i, k := range j.Running {
		if k == key {
			j.Running = append(j.Running[:i], j.Running[i+1:]...)
			break
		}
	}
	j.saveLocked()
	close(j.change)
	j.change = make(chan struct{})
}

func (j *job) save() { j.mu.Lock(); defer j.mu.Unlock(); j.saveLocked() }

func (j *job) saveLocked() {
	if j.path == "" {
		return
	}
	b, _ := json.MarshalIndent(j, "", "  ")
	if os.MkdirAll(filepath.Dir(j.path), 0o700) == nil && os.WriteFile(j.path+".tmp", b, 0o600) == nil {
		os.Rename(j.path+".tmp", j.path)
	}
}

// Collect hands out results not yet delivered, at most max at a time. With none ready it waits up
// to wait for one. The text says how much is still to come.
func (g *Gateway) Collect(ctx context.Context, id string, wait time.Duration, max int) (string, error) {
	g.jobs.mu.Lock()
	j := g.jobs.byID[id]
	g.jobs.mu.Unlock()
	if j == nil { // a job from before this gateway process started: its finished results are on disk
		if g.jobs.dir == "" || strings.ContainsAny(id, `/\.`) {
			return "", fmt.Errorf("no job %q", id)
		}
		b, err := os.ReadFile(filepath.Join(g.jobs.dir, id+".json"))
		if err != nil {
			return "", fmt.Errorf("no job %q", id)
		}
		j = &job{change: make(chan struct{}), path: filepath.Join(g.jobs.dir, id+".json")}
		if json.Unmarshal(b, j) != nil {
			return "", fmt.Errorf("job %q could not be read", id)
		}
		j.lost = len(j.Running) > 0
		g.jobs.mu.Lock()
		if g.jobs.byID == nil {
			g.jobs.byID = map[string]*job{}
		}
		g.jobs.byID[id] = j
		g.jobs.mu.Unlock()
	}
	if max <= 0 {
		max = 1
	}
	deadline := time.After(wait)
	for {
		j.mu.Lock()
		if n := len(j.Results) - j.Delivered; n > 0 {
			n = min(n, max)
			out := j.Results[j.Delivered : j.Delivered+n]
			j.Delivered += n
			var b strings.Builder
			for _, r := range out {
				if r.OK {
					fmt.Fprintf(&b, "### task %s, answered by %s in %.0fs\n\n%s\n\n", r.Task, r.Used, r.Seconds, r.Text)
				} else {
					fmt.Fprintf(&b, "### task %s FAILED\n\n%s\n\n", r.Task, r.Text)
				}
			}
			b.WriteString(j.statusLocked())
			j.saveLocked()
			j.mu.Unlock()
			return b.String(), nil
		}
		status, running, change, lost := j.statusLocked(), len(j.Running), j.change, j.lost
		j.mu.Unlock()
		if running == 0 || lost {
			return status, nil
		}
		select {
		case <-change:
		case <-deadline:
			return "nothing new yet. " + status, nil
		case <-ctx.Done():
			return "", errors.New("collect was cancelled; the job keeps running")
		}
	}
}

func (j *job) statusLocked() string {
	left := len(j.Results) - j.Delivered
	switch {
	case len(j.Running) == 0 && left == 0:
		return fmt.Sprintf("job %s is complete: all %d results delivered.", j.ID, j.Total)
	case len(j.Running) == 0:
		return fmt.Sprintf("job %s: all %d results are in, %d not yet collected. Call collect again.", j.ID, j.Total, left)
	case j.lost:
		return fmt.Sprintf("job %s: %d of %d results in, %d not yet collected. Lost when the gateway restarted: %s. Dispatch those again.",
			j.ID, len(j.Results), j.Total, left, strings.Join(j.Running, "; "))
	}
	return fmt.Sprintf("job %s: %d of %d results in, %d not yet collected, still running: %s. Call collect again.",
		j.ID, len(j.Results), j.Total, left, strings.Join(j.Running, "; "))
}

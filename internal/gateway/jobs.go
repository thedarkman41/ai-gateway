package gateway

import (
	"context"
	"crypto/rand"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
	"time"
)

// A build is one program written by many agents. Its tasks (one function each) go into a queue.
// Every free agent takes a task, and takes another as soon as it finishes, so the faster agents
// write more of the program. The parts are assembled in task order into one piece of code for the
// planner to compile. Parts that fail there are reworked: each one is rewritten by a vote of the
// strongest agents (Consensus) and the build is assembled again.

const (
	maxTasks      = 40
	maxWorkers    = 12
	maxJobs       = 20
	taskAttempts  = 3 // agents that may try one task before it is given up
	workerStrikes = 2 // failures in a row before an agent leaves the queue
	buildTimeout  = 40 * time.Minute
	jobKeep       = 48 * time.Hour
	maxBuildChars = 2_000_000 // all prompts, context, header and footer of one build
)

// Task is one function to write.
type Task struct {
	ID     string   `json:"id"`
	Prompt string   `json:"prompt"`
	Tests  []string `json:"tests,omitempty"` // Python expressions that call the function, e.g. "slugify('A b')": a rework vote compares candidates on them
}

// BuildSpec is what a build is made from.
type BuildSpec struct {
	Tasks     []Task
	System    string // instructions for every agent
	Language  string // e.g. "python"; names the language to the agents and decides how the program is assembled
	Header    string // code placed before the parts
	Footer    string // code placed after them
	Context   string // shared text every agent sees before its task: existing code, signatures, attached files
	LocalOnly bool   // only local models may see any of it
}

// Part is the code written for a task.
type Part struct {
	Text     string  `json:"text"`
	By       string  `json:"by"`     // "provider/model", or how a vote chose it
	Author   string  `json:"author"` // the agent whose code it is
	Seconds  float64 `json:"seconds"`
	Attempts int     `json:"attempts"`
	Reworked bool    `json:"reworked,omitempty"`
}

type buildJob struct {
	ID        string            `json:"id"`
	Created   time.Time         `json:"created"`
	System    string            `json:"system,omitempty"`
	Language  string            `json:"language,omitempty"`
	Header    string            `json:"header,omitempty"`
	Footer    string            `json:"footer,omitempty"`
	Context   string            `json:"context,omitempty"`
	LocalOnly bool              `json:"local_only,omitempty"`
	Tasks     []Task            `json:"tasks"`
	Parts     map[string]Part   `json:"parts"`
	Missing   map[string]string `json:"missing"`  // task id → why nobody produced it
	Tried     map[string]int    `json:"tried"`    // task id → agents that have attempted it
	Private   map[string]string `json:"private"`  // task id → why only a local agent may take it
	Redacted  string            `json:"redacted"` // what was redacted from prompts before they left
	Log       []string          `json:"log"`
	Running   bool              `json:"running"`
	Cancelled bool              `json:"cancelled,omitempty"`

	mu       sync.Mutex
	cond     *sync.Cond
	queue    []string
	inflight map[string]string   // task id → agent
	tried    map[string][]string // task id → agents that failed it
	done     chan struct{}       // closed when the current phase ends
	cancel   context.CancelFunc  // stops the current phase's agent calls
	path     string
}

type jobs struct {
	mu   sync.Mutex
	byID map[string]*buildJob
	dir  string
}

// SetJobsDir makes builds survive a restart of the gateway (Bill stops an idle server).
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

// noAgent is the error for "nobody can take this now": a quota problem with a time to retry when
// the agents are only resting, else no provider at all.
func (g *Gateway) noAgent(localOnly bool) *Problem {
	if localOnly {
		return &Problem{Status: 403, Code: "no_local_provider", Message: "only a local model may see this, and none is free"}
	}
	soonest := time.Time{}
	for _, p := range g.providers {
		if p.Available() != nil {
			continue
		}
		if until := g.ledger.Snapshot(p.Config().ID, g.now()).DownUntil; until.After(g.now()) && (soonest.IsZero() || until.Before(soonest)) {
			soonest = until
		}
	}
	if !soonest.IsZero() {
		return &Problem{Status: 429, Code: "quota", Message: "every agent is resting or out of its free allowance", RetryAfter: int(soonest.Sub(g.now()).Seconds()) + 1}
	}
	return &Problem{Status: 503, Code: "no_provider", Message: "no agent is free right now; see list_models"}
}

// Build starts a build and returns its id and who is working on it.
func (g *Gateway) Build(spec BuildSpec) (id, plan string, err error) {
	tasks := spec.Tasks
	if len(tasks) == 0 || len(tasks) > maxTasks {
		return "", "", &Problem{Status: 400, Code: "bad_request", Message: fmt.Sprintf("tasks: a build takes 1 to %d tasks", maxTasks)}
	}
	seen := map[string]bool{}
	total := len(spec.Header) + len(spec.Footer) + len(spec.Context) + len(spec.System)
	for i := range tasks {
		tasks[i].ID = strings.TrimSpace(tasks[i].ID)
		if tasks[i].ID == "" || seen[tasks[i].ID] || strings.TrimSpace(tasks[i].Prompt) == "" {
			return "", "", &Problem{Status: 400, Code: "bad_request", Message: fmt.Sprintf("tasks: every task needs its own id and a prompt (task %q)", tasks[i].ID)}
		}
		seen[tasks[i].ID] = true
		total += len(tasks[i].Prompt)
	}
	if total > maxBuildChars {
		return "", "", &Problem{Status: 413, Code: "too_large", Message: fmt.Sprintf("the build is %d characters; the limit is %d", total, maxBuildChars)}
	}
	ctx := context.Background()
	if spec.LocalOnly {
		ctx = WithLocalOnly(ctx)
	}
	agents := g.ranked(spec.LocalOnly)
	if len(agents) == 0 {
		return "", "", g.noAgent(spec.LocalOnly)
	}
	private := map[string]string{} // tasks that only a local agent may take, and why
	for _, t := range tasks {
		if why := g.private(ctx, spec.System, spec.Context, t.Prompt); why != "" {
			private[t.ID] = why
		}
	}
	agents = agents[:min(len(agents), maxWorkers)]

	var rnd [4]byte
	rand.Read(rnd[:])
	b := &buildJob{ID: fmt.Sprintf("b_%x", rnd), Created: time.Now(), System: spec.System, Language: strings.ToLower(strings.TrimSpace(spec.Language)),
		Header: spec.Header, Footer: spec.Footer, Context: spec.Context, LocalOnly: spec.LocalOnly,
		Tasks: tasks, Parts: map[string]Part{}, Missing: map[string]string{}, Tried: map[string]int{}, Private: private, Running: true,
		inflight: map[string]string{}, tried: map[string][]string{}, done: make(chan struct{})}
	b.cond = sync.NewCond(&b.mu)
	for _, t := range tasks {
		b.queue = append(b.queue, t.ID)
	}
	g.jobs.mu.Lock()
	if g.jobs.dir != "" {
		b.path = filepath.Join(g.jobs.dir, b.ID+".json")
	}
	if g.jobs.byID == nil {
		g.jobs.byID = map[string]*buildJob{}
	}
	if len(g.jobs.byID) >= maxJobs { // drop the oldest that is not running; it stays on disk
		var oldest *buildJob
		for _, o := range g.jobs.byID {
			if !o.isRunning() && (oldest == nil || o.Created.Before(oldest.Created)) {
				oldest = o
			}
		}
		if oldest != nil {
			delete(g.jobs.byID, oldest.ID)
		}
	}
	g.jobs.byID[b.ID] = b
	g.jobs.mu.Unlock()

	ctx, cancel := context.WithTimeout(ctx, buildTimeout)
	b.mu.Lock()
	b.cancel = cancel
	b.saveLocked()
	b.mu.Unlock()
	var wg sync.WaitGroup
	var names []string
	for _, p := range agents { // strongest first, so the strongest take the first tasks
		id := p.Config().ID
		names = append(names, id)
		wg.Add(1)
		local := p.Config().Local
		go func() {
			defer wg.Done()
			g.work(ctx, b, id, local)
		}()
		time.Sleep(time.Millisecond) // keep the start order
	}
	go func() {
		wg.Wait()
		cancel()
		b.mu.Lock()
		for _, id := range b.queue { // whatever is left had no agent to take it
			b.Missing[id] = "no agent was left to write it"
			if why := b.Private[id]; why != "" {
				b.Missing[id] = "kept on this machine (" + why + ") and no local agent could write it"
			}
			if b.Cancelled {
				b.Missing[id] = "the build was cancelled"
			}
		}
		b.queue = nil
		b.finishLocked()
		b.mu.Unlock()
	}()
	return b.ID, fmt.Sprintf("build %s started: %d tasks in the queue, %d agents free to take them, strongest first: %s",
		b.ID, len(tasks), len(names), strings.Join(names, ", ")), nil
}

// taskPrompt is what an agent is sent for a task: the shared context, then the task.
func (b *buildJob) taskPrompt(t *Task) string {
	if strings.TrimSpace(b.Context) == "" {
		return t.Prompt
	}
	return strings.TrimRight(b.Context, "\n") + "\n\n" + t.Prompt
}

func (b *buildJob) system() string {
	if b.Language == "" {
		return b.System
	}
	return strings.TrimSpace(b.System + "\nThe code must be written in " + b.Language + ".")
}

// work is one agent's loop: take a task, write it, take the next. It leaves when the queue is
// empty and nothing in flight could come back to it, or after failing twice in a row.
func (g *Gateway) work(ctx context.Context, b *buildJob, agent string, local bool) {
	strikes := 0
	for {
		b.mu.Lock()
		var task *Task
		for task == nil {
			for i, id := range b.queue {
				if !contains(b.tried[id], agent) && (b.Private[id] == "" || local) { // a private task waits for a local agent
					task = b.task(id)
					b.queue = append(b.queue[:i], b.queue[i+1:]...)
					break
				}
			}
			if task != nil {
				break
			}
			if len(b.inflight) == 0 || ctx.Err() != nil { // nothing can come back to the queue
				b.mu.Unlock()
				return
			}
			b.cond.Wait() // a task in flight elsewhere may fail and return
		}
		b.inflight[task.ID] = agent
		b.Tried[task.ID]++
		prompt, system := b.taskPrompt(task), b.system()
		b.mu.Unlock()

		start := time.Now()
		ans, err := g.Ask(ctx, agent, system, prompt+codeOnlySuffix, "")
		secs := time.Since(start).Seconds()

		b.mu.Lock()
		delete(b.inflight, task.ID)
		switch {
		case err == nil:
			strikes = 0
			if ans.Redacted != "" {
				b.Redacted = ans.Redacted
			}
			b.Parts[task.ID] = Part{Text: codeOnly(ans.Text), By: ans.Used, Author: agent, Seconds: secs, Attempts: b.Tried[task.ID]}
			b.Log = append(b.Log, fmt.Sprintf("%s written by %s in %.0fs", task.ID, ans.Used, secs))
		case ctx.Err() != nil: // cancelled or out of time: not the agent's failure
			b.queue = append([]string{task.ID}, b.queue...)
		default:
			strikes++
			b.tried[task.ID] = append(b.tried[task.ID], agent)
			why := firstLine(err.Error())
			b.Log = append(b.Log, fmt.Sprintf("%s failed on %s: %s", task.ID, agent, why))
			if len(b.tried[task.ID]) >= taskAttempts {
				b.Missing[task.ID] = fmt.Sprintf("%d agents failed it; last: %s", taskAttempts, why)
			} else {
				b.queue = append([]string{task.ID}, b.queue...) // back to the front for another agent
			}
		}
		b.saveLocked()
		b.cond.Broadcast()
		b.mu.Unlock()
		if strikes >= workerStrikes || ctx.Err() != nil {
			b.mu.Lock()
			b.cond.Broadcast()
			b.mu.Unlock()
			return
		}
	}
}

// Rework rewrites the parts that failed the planner's check. Each goes to a vote of the strongest
// agents, with the failure and the rejected code in the prompt; the winner replaces the part.
func (g *Gateway) Rework(id string, failed map[string]string) (string, error) {
	b, err := g.find(id)
	if err != nil {
		return "", err
	}
	b.mu.Lock()
	if b.Running {
		b.mu.Unlock()
		return "", &Problem{Status: 409, Code: "build_running", Message: fmt.Sprintf("build %s is still running; collect it first", id)}
	}
	var ids []string
	for _, t := range b.Tasks { // in task order
		if _, ok := failed[t.ID]; ok {
			ids = append(ids, t.ID)
		}
	}
	if len(ids) != len(failed) || len(ids) == 0 {
		b.mu.Unlock()
		return "", &Problem{Status: 400, Code: "bad_request", Message: fmt.Sprintf("failures must name tasks of build %s; its tasks are: %s", id, strings.Join(b.taskIDs(), ", "))}
	}
	ctx := context.Background()
	if b.LocalOnly { // a build keeps the scope it was created with
		ctx = WithLocalOnly(ctx)
	}
	ctx, cancel := context.WithTimeout(ctx, buildTimeout)
	b.Running, b.Cancelled = true, false
	b.done = make(chan struct{})
	b.cancel = cancel
	b.saveLocked()
	b.mu.Unlock()

	go func() {
		defer cancel()
		for _, tid := range ids {
			if ctx.Err() != nil {
				break
			}
			b.mu.Lock()
			task, old := b.task(tid), b.Parts[tid]
			others := []string{b.Header} // what the function is tested alongside: the rest of the program
			for _, t := range b.Tasks {
				if t.ID != tid {
					others = append(others, b.Parts[t.ID].Text)
				}
			}
			surrounding := strings.Join(others, "\n\n")
			prompt, system := b.taskPrompt(task), b.system()
			b.inflight[tid] = "vote"
			b.mu.Unlock()
			prompt += "\n\nAn earlier attempt was rejected. The problem reported:\n" + clipChars(strings.TrimSpace(failed[tid]), 4000)
			if old.Text != "" {
				prompt += "\n\nThe rejected code:\n" + old.Text
			}
			start := time.Now()
			v, err := g.Consensus(ctx, Request{Prompt: prompt, System: system, Context: surrounding, Exprs: task.Tests})
			b.mu.Lock()
			delete(b.inflight, tid)
			if err != nil {
				b.Log = append(b.Log, fmt.Sprintf("%s rework failed: %s", tid, firstLine(err.Error())))
				b.Missing[tid] = "rework: " + firstLine(err.Error())
			} else {
				delete(b.Missing, tid)
				if v.Redacted != "" {
					b.Redacted = v.Redacted
				}
				b.Parts[tid] = Part{Text: v.Winner(), By: v.How(), Author: v.Groups[0].By[0], Seconds: time.Since(start).Seconds(), Attempts: old.Attempts + 1, Reworked: true}
				b.Log = append(b.Log, fmt.Sprintf("%s reworked: %s", tid, v.How()))
			}
			b.saveLocked()
			b.mu.Unlock()
		}
		b.mu.Lock()
		b.finishLocked()
		b.mu.Unlock()
	}()
	return fmt.Sprintf("build %s: reworking %d parts by vote of the %d strongest free agents: %s. Call collect.", id, len(ids), defaultVoters, strings.Join(ids, ", ")), nil
}

// Cancel stops a build's outstanding agent calls. What was written is kept.
func (g *Gateway) Cancel(id string) error {
	b, err := g.find(id)
	if err != nil {
		return err
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.Running && b.cancel != nil {
		b.Cancelled = true
		b.cancel()
		b.cond.Broadcast()
	}
	return nil
}

// wait blocks until the build's current phase ends, or for at most d.
func (b *buildJob) wait(ctx context.Context, d time.Duration) error {
	b.mu.Lock()
	done := b.done
	b.mu.Unlock()
	select {
	case <-done:
	case <-time.After(d):
	case <-ctx.Done():
		return ctx.Err()
	}
	return nil
}

// PartView is one task of a build as a caller sees it.
type PartView struct {
	ID       string  `json:"id"`
	Status   string  `json:"status"` // pending | running | done | failed | missing
	Author   string  `json:"author"`
	Code     string  `json:"code"`
	Error    *string `json:"error"`
	Attempts int     `json:"attempts"`
	Reworked bool    `json:"reworked"`
}

// BuildView is a build as a caller sees it.
type BuildView struct {
	ID        string     `json:"build_id"`
	Status    string     `json:"status"` // running | done | partial | failed
	Language  string     `json:"language"`
	Program   string     `json:"program"`
	Parts     []PartView `json:"parts"`
	Notes     []string   `json:"notes"`
	Elapsed   int        `json:"elapsed_seconds"`
	LocalOnly bool       `json:"-"`
}

// View waits up to wait for the build's current phase to end and reports where it stands.
func (g *Gateway) View(ctx context.Context, id string, wait time.Duration) (BuildView, error) {
	b, err := g.find(id)
	if err != nil {
		return BuildView{}, err
	}
	if wait > 0 {
		if err := b.wait(ctx, wait); err != nil {
			return BuildView{}, err
		}
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	v := BuildView{ID: b.ID, Language: b.Language, Notes: []string{}, Elapsed: int(time.Since(b.Created).Seconds()), LocalOnly: b.LocalOnly}
	done := 0
	var home []string
	for _, t := range b.Tasks {
		pv := PartView{ID: t.ID, Attempts: b.Tried[t.ID]}
		why, gone := b.Missing[t.ID]
		p, written := b.Parts[t.ID]
		_, flying := b.inflight[t.ID]
		switch {
		case flying:
			pv.Status = "running"
			pv.Author, pv.Code, pv.Reworked = p.Author, p.Text, p.Reworked // a part being reworked still has its old code
		case gone && strings.Contains(why, "failed"):
			pv.Status, pv.Error = "failed", &why
		case gone:
			pv.Status, pv.Error = "missing", &why
		case written:
			pv.Status, pv.Author, pv.Code, pv.Reworked, pv.Attempts = "done", p.Author, p.Text, p.Reworked, max(p.Attempts, b.Tried[t.ID])
			done++
			if p.Reworked {
				v.Notes = append(v.Notes, "reworked "+t.ID+": "+p.By)
			}
		default:
			pv.Status = "pending"
		}
		if b.Private[t.ID] != "" {
			home = append(home, t.ID)
		}
		v.Parts = append(v.Parts, pv)
	}
	if b.Redacted != "" {
		v.Notes = append(v.Notes, "redacted: "+b.Redacted)
	}
	if len(home) > 0 {
		v.Notes = append(v.Notes, "kept home: "+strings.Join(home, ", "))
	}
	if b.Cancelled {
		v.Notes = append(v.Notes, "cancelled")
	}
	switch {
	case b.Running:
		v.Status = "running"
	case done == len(b.Tasks):
		v.Status = "done"
	case done == 0:
		v.Status = "failed"
	default:
		v.Status = "partial"
	}
	if !b.Running {
		v.Program = b.assembleLocked()
	}
	return v, nil
}

// assembleLocked joins the header, the parts in task order and the footer. A part that was not
// written is one comment line saying so, so the program still shows where it belongs.
func (b *buildJob) assembleLocked() string {
	var code, bodies []string
	if strings.TrimSpace(b.Header) != "" {
		code = append(code, strings.TrimRight(b.Header, "\n"))
	}
	for _, t := range b.Tasks {
		if p, ok := b.Parts[t.ID]; ok && b.Missing[t.ID] == "" {
			bodies = append(bodies, p.Text)
			continue
		}
		state := "missing"
		if strings.Contains(b.Missing[t.ID], "failed") {
			state = "failed"
		}
		bodies = append(bodies, fmt.Sprintf("%s [ai-gateway: %s %s]", commentMark(b.Language), t.ID, state))
	}
	// Each agent writes the imports its own function needs. In Python they are gathered at the
	// top, once each, instead of being scattered through the file.
	if b.Language == "" || b.Language == "python" {
		if imports, rest := hoistImports(b.Header, bodies); len(imports) > 0 {
			code = append(code, strings.Join(imports, "\n"))
			bodies = rest
		}
	}
	// In Rust the same must happen or the program does not compile: two parts that both say
	// `use std::collections::HashMap;` are a duplicate-import error.
	if b.Language == "rust" {
		if uses, rest := hoistRustUses(b.Header, bodies); len(uses) > 0 {
			code = append(code, strings.Join(uses, "\n"))
			bodies = rest
		}
	}
	code = append(code, bodies...)
	if strings.TrimSpace(b.Footer) != "" {
		code = append(code, strings.TrimRight(b.Footer, "\n"))
	}
	return strings.Join(code, "\n\n")
}

// commentMark is how a one-line comment starts in a language.
func commentMark(language string) string {
	switch language {
	case "", "python", "ruby", "shell", "bash", "sh", "zsh", "perl", "r", "yaml", "toml", "elixir", "julia", "powershell", "makefile":
		return "#"
	case "sql", "lua", "haskell", "ada", "elm":
		return "--"
	case "lisp", "clojure", "scheme":
		return ";;"
	}
	return "//"
}

// Collect waits up to wait for the build's current phase to finish. Finished, it returns the whole
// program assembled in task order and a record of who wrote what; unfinished, the progress.
func (g *Gateway) Collect(ctx context.Context, id string, wait time.Duration) (string, error) {
	v, err := g.View(ctx, id, wait)
	if err != nil {
		if ctx.Err() != nil {
			return "", fmt.Errorf("collect was cancelled; the build keeps running")
		}
		return "", err
	}
	b, _ := g.find(id)
	if v.Status == "running" {
		written, queued := 0, 0
		var doing []string
		for _, p := range v.Parts {
			switch p.Status {
			case "done":
				written++
			case "pending":
				queued++
			case "running":
				doing = append(doing, p.ID)
			}
		}
		return fmt.Sprintf("build %s is still running: %d of %d parts written, %d in the queue, in progress: %s. Call collect again.",
			v.ID, written, len(v.Parts), queued, strings.Join(doing, ", ")), nil
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	var record, missing []string
	for _, t := range b.Tasks {
		if why := b.Missing[t.ID]; why != "" {
			missing = append(missing, t.ID+" ("+why+")")
			continue
		}
		record = append(record, fmt.Sprintf("- %s: %s", t.ID, b.Parts[t.ID].By))
	}
	var out strings.Builder
	fmt.Fprintf(&out, "build %s is complete: %d of %d parts written.\n", b.ID, len(b.Tasks)-len(missing), len(b.Tasks))
	if len(missing) > 0 {
		fmt.Fprintf(&out, "MISSING (each is one comment line in the program below; rework them): %s\n", strings.Join(missing, "; "))
	}
	for _, n := range v.Notes {
		if strings.HasPrefix(n, "redacted") || strings.HasPrefix(n, "kept home") {
			fmt.Fprintf(&out, "Privacy: %s\n", n)
		}
	}
	fmt.Fprintf(&out, "Who wrote what:\n%s\n\nCheck the program (compile it, run its tests). For each part that fails, call rework with the task id and the error.\n\n===== program =====\n%s\n",
		strings.Join(record, "\n"), v.Program)
	return out.String(), nil
}

var importRe = regexp.MustCompile(`^(import\s+\S|from\s+\S+\s+import\s)`)

// hoistImports takes the top-level import lines out of Python parts and returns them once each, in
// first-seen order, leaving out any the header already has. Parts that are not Python are returned
// untouched with no imports.
func hoistImports(header string, parts []string) (imports []string, rest []string) {
	python := 0
	for _, p := range parts {
		if defRe.MatchString(p) {
			python++
		}
	}
	if python*2 <= len(parts) {
		return nil, parts
	}
	seen := map[string]bool{}
	for _, l := range strings.Split(header, "\n") {
		seen[strings.TrimSpace(l)] = true
	}
	for _, p := range parts {
		var kept []string
		for _, l := range strings.Split(p, "\n") {
			if importRe.MatchString(l) { // column 0 only: an import inside a function stays where it is
				if key := strings.TrimSpace(l); !seen[key] {
					seen[key] = true
					imports = append(imports, key)
				}
				continue
			}
			kept = append(kept, l)
		}
		rest = append(rest, strings.Trim(strings.Join(kept, "\n"), "\n"))
	}
	return imports, rest
}

var rustUseRe = regexp.MustCompile(`^use\s+[^;]+;\s*$`)

// hoistRustUses takes the one-line `use` declarations at the left margin out of Rust parts and
// returns them once each, in first-seen order, leaving out any the header already has. A `use`
// inside a function or module body is indented and stays where it is; so do `pub use` and a `use`
// that spans several lines.
func hoistRustUses(header string, parts []string) (uses []string, rest []string) {
	norm := func(l string) string { return strings.Join(strings.Fields(l), " ") }
	seen := map[string]bool{}
	for _, l := range strings.Split(header, "\n") {
		seen[norm(l)] = true
	}
	for _, p := range parts {
		var kept []string
		for _, l := range strings.Split(p, "\n") {
			if rustUseRe.MatchString(l) {
				if key := norm(l); !seen[key] {
					seen[key] = true
					uses = append(uses, key)
				}
				continue
			}
			kept = append(kept, l)
		}
		rest = append(rest, strings.Trim(strings.Join(kept, "\n"), "\n"))
	}
	return uses, rest
}

func (g *Gateway) find(id string) (*buildJob, error) {
	g.jobs.mu.Lock()
	defer g.jobs.mu.Unlock()
	if b := g.jobs.byID[id]; b != nil {
		return b, nil
	}
	unknown := &Problem{Status: 404, Code: "unknown_build", Message: fmt.Sprintf("no build %q", id)}
	if g.jobs.dir == "" || strings.ContainsAny(id, `/\.`) {
		return nil, unknown
	}
	// A build from before this gateway process started: what it had finished is on disk.
	path := filepath.Join(g.jobs.dir, id+".json")
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, unknown
	}
	b := &buildJob{path: path, inflight: map[string]string{}, tried: map[string][]string{}, done: make(chan struct{})}
	b.cond = sync.NewCond(&b.mu)
	if json.Unmarshal(data, b) != nil || b.Parts == nil {
		return nil, unknown
	}
	for _, m := range []*map[string]string{&b.Missing, &b.Private} {
		if *m == nil {
			*m = map[string]string{}
		}
	}
	if b.Tried == nil {
		b.Tried = map[string]int{}
	}
	if b.Running { // it was cut short: what was not written is missing, and can be reworked
		for _, t := range b.Tasks {
			if _, ok := b.Parts[t.ID]; !ok {
				b.Missing[t.ID] = "the gateway restarted before it was written"
			}
		}
	}
	b.Running = false
	close(b.done)
	if g.jobs.byID == nil {
		g.jobs.byID = map[string]*buildJob{}
	}
	g.jobs.byID[id] = b
	return b, nil
}

func (b *buildJob) isRunning() bool { b.mu.Lock(); defer b.mu.Unlock(); return b.Running }

func (b *buildJob) finishLocked() {
	b.Running = false
	b.saveLocked()
	close(b.done)
	b.cond.Broadcast()
}

func (b *buildJob) task(id string) *Task {
	for i := range b.Tasks {
		if b.Tasks[i].ID == id {
			return &b.Tasks[i]
		}
	}
	return nil
}

func (b *buildJob) taskIDs() []string {
	var ids []string
	for _, t := range b.Tasks {
		ids = append(ids, t.ID)
	}
	return ids
}

func (b *buildJob) saveLocked() {
	if b.path == "" {
		return
	}
	data, _ := json.MarshalIndent(b, "", "  ")
	if os.MkdirAll(filepath.Dir(b.path), 0o700) == nil && os.WriteFile(b.path+".tmp", data, 0o600) == nil {
		os.Rename(b.path+".tmp", b.path)
	}
}

func contains(list []string, s string) bool {
	for _, x := range list {
		if x == s {
			return true
		}
	}
	return false
}

func clipChars(s string, n int) string {
	if r := []rune(s); len(r) > n {
		return string(r[:n])
	}
	return s
}

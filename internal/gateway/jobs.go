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
)

// Task is one function to write.
type Task struct {
	ID     string `json:"id"`
	Prompt string `json:"prompt"`
}

// Part is the code written for a task.
type Part struct {
	Text    string  `json:"text"`
	By      string  `json:"by"` // the agent, or how a vote chose it
	Seconds float64 `json:"seconds"`
}

type buildJob struct {
	ID      string            `json:"id"`
	Created time.Time         `json:"created"`
	System  string            `json:"system,omitempty"`
	Header  string            `json:"header,omitempty"`
	Footer  string            `json:"footer,omitempty"`
	Tasks   []Task            `json:"tasks"`
	Parts   map[string]Part   `json:"parts"`
	Missing map[string]string `json:"missing"` // task id → why nobody produced it
	Log     []string          `json:"log"`
	Running bool              `json:"running"`

	mu       sync.Mutex
	cond     *sync.Cond
	queue    []string
	inflight map[string]string   // task id → agent
	tried    map[string][]string // task id → agents that failed it
	done     chan struct{}       // closed when the current phase ends
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

// Build starts a build and returns its id and who is working on it.
func (g *Gateway) Build(tasks []Task, system, header, footer string) (id, plan string, err error) {
	if len(tasks) == 0 || len(tasks) > maxTasks {
		return "", "", fmt.Errorf("a build takes 1 to %d tasks", maxTasks)
	}
	seen := map[string]bool{}
	for i := range tasks {
		tasks[i].ID = strings.TrimSpace(tasks[i].ID)
		if tasks[i].ID == "" || seen[tasks[i].ID] || strings.TrimSpace(tasks[i].Prompt) == "" {
			return "", "", fmt.Errorf("every task needs its own id and a prompt (task %q)", tasks[i].ID)
		}
		seen[tasks[i].ID] = true
	}
	agents := g.ranked()
	if len(agents) == 0 {
		return "", "", errors.New("no agent is free right now; see list_models")
	}
	agents = agents[:min(len(agents), maxWorkers)]

	var rnd [4]byte
	rand.Read(rnd[:])
	b := &buildJob{ID: fmt.Sprintf("b%x", rnd), Created: time.Now(), System: system, Header: header, Footer: footer,
		Tasks: tasks, Parts: map[string]Part{}, Missing: map[string]string{}, Running: true,
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
	if len(g.jobs.byID) >= maxJobs {
		var oldest *buildJob
		for _, o := range g.jobs.byID {
			if oldest == nil || o.Created.Before(oldest.Created) {
				oldest = o
			}
		}
		delete(g.jobs.byID, oldest.ID)
	}
	g.jobs.byID[b.ID] = b
	g.jobs.mu.Unlock()
	b.save()

	ctx, cancel := context.WithTimeout(context.Background(), buildTimeout)
	var wg sync.WaitGroup
	var names []string
	for _, p := range agents { // strongest first, so the strongest take the first tasks
		id := p.Config().ID
		names = append(names, id)
		wg.Add(1)
		go func() {
			defer wg.Done()
			g.work(ctx, b, id)
		}()
		time.Sleep(time.Millisecond) // keep the start order
	}
	go func() {
		wg.Wait()
		cancel()
		b.mu.Lock()
		for _, id := range b.queue { // whatever is left had no agent to take it
			b.Missing[id] = "no agent was left to write it"
		}
		b.queue = nil
		b.finishLocked()
		b.mu.Unlock()
	}()
	return b.ID, fmt.Sprintf("build %s started: %d tasks in the queue, %d agents taking them, strongest first: %s",
		b.ID, len(tasks), len(names), strings.Join(names, ", ")), nil
}

// work is one agent's loop: take a task, write it, take the next. It leaves when the queue is
// empty and nothing in flight could come back to it, or after failing twice in a row.
func (g *Gateway) work(ctx context.Context, b *buildJob, agent string) {
	strikes := 0
	for {
		b.mu.Lock()
		var task *Task
		for task == nil {
			for i, id := range b.queue {
				if !contains(b.tried[id], agent) {
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
		b.mu.Unlock()

		start := time.Now()
		ans, err := g.Ask(ctx, agent, b.System, task.Prompt+codeOnlySuffix, "")
		secs := time.Since(start).Seconds()

		b.mu.Lock()
		delete(b.inflight, task.ID)
		if err == nil {
			strikes = 0
			b.Parts[task.ID] = Part{Text: codeOnly(ans.Text), By: ans.Used, Seconds: secs}
			b.Log = append(b.Log, fmt.Sprintf("%s written by %s in %.0fs", task.ID, ans.Used, secs))
		} else {
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
		return "", fmt.Errorf("build %s is still running; collect it first", id)
	}
	var ids []string
	for _, t := range b.Tasks { // in task order
		if _, ok := failed[t.ID]; ok {
			ids = append(ids, t.ID)
		}
	}
	if len(ids) != len(failed) || len(ids) == 0 {
		b.mu.Unlock()
		return "", fmt.Errorf("failed must name tasks of build %s; its tasks are: %s", id, strings.Join(b.taskIDs(), ", "))
	}
	b.Running = true
	b.done = make(chan struct{})
	b.saveLocked()
	b.mu.Unlock()

	go func() {
		ctx, cancel := context.WithTimeout(context.Background(), buildTimeout)
		defer cancel()
		for _, tid := range ids {
			b.mu.Lock()
			task, old := b.task(tid), b.Parts[tid]
			b.mu.Unlock()
			prompt := task.Prompt + "\n\nAn earlier attempt was rejected. The problem reported:\n" + strings.TrimSpace(failed[tid])
			if old.Text != "" {
				prompt += "\n\nThe rejected code:\n" + old.Text
			}
			start := time.Now()
			v, err := g.Consensus(ctx, b.System, prompt, defaultVoters)
			b.mu.Lock()
			if err != nil {
				b.Log = append(b.Log, fmt.Sprintf("%s rework failed: %s", tid, firstLine(err.Error())))
				b.Missing[tid] = "rework: " + firstLine(err.Error())
			} else {
				delete(b.Missing, tid)
				b.Parts[tid] = Part{Text: v.Winner(), By: v.How(), Seconds: time.Since(start).Seconds()}
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

// Collect waits up to wait for the build's current phase to finish. Finished, it returns the whole
// program assembled in task order and a record of who wrote what; unfinished, the progress.
func (g *Gateway) Collect(ctx context.Context, id string, wait time.Duration) (string, error) {
	b, err := g.find(id)
	if err != nil {
		return "", err
	}
	b.mu.Lock()
	done := b.done
	b.mu.Unlock()
	select {
	case <-done:
	case <-time.After(wait):
	case <-ctx.Done():
		return "", errors.New("collect was cancelled; the build keeps running")
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.Running {
		var doing []string
		for tid, agent := range b.inflight {
			doing = append(doing, tid+" ("+agent+")")
		}
		return fmt.Sprintf("build %s is still running: %d of %d parts written, %d in the queue, in progress: %s. Call collect again.",
			b.ID, len(b.Parts), len(b.Tasks), len(b.queue), strings.Join(doing, ", ")), nil
	}
	var code []string
	if strings.TrimSpace(b.Header) != "" {
		code = append(code, strings.TrimRight(b.Header, "\n"))
	}
	var record, missing []string
	for _, t := range b.Tasks {
		if why, gone := b.Missing[t.ID]; gone {
			missing = append(missing, t.ID+" ("+why+")")
			continue
		}
		p := b.Parts[t.ID]
		code = append(code, p.Text)
		record = append(record, fmt.Sprintf("- %s: %s", t.ID, p.By))
	}
	if strings.TrimSpace(b.Footer) != "" {
		code = append(code, strings.TrimRight(b.Footer, "\n"))
	}
	var out strings.Builder
	fmt.Fprintf(&out, "build %s is complete: %d of %d parts written.\n", b.ID, len(b.Tasks)-len(missing), len(b.Tasks))
	if len(missing) > 0 {
		fmt.Fprintf(&out, "MISSING (not in the program below; rework them): %s\n", strings.Join(missing, "; "))
	}
	fmt.Fprintf(&out, "Who wrote what:\n%s\n\nCheck the program (compile it, run its tests). For each part that fails, call rework with the task id and the error.\n\n===== program =====\n%s\n",
		strings.Join(record, "\n"), strings.Join(code, "\n\n"))
	return out.String(), nil
}

func (g *Gateway) find(id string) (*buildJob, error) {
	g.jobs.mu.Lock()
	defer g.jobs.mu.Unlock()
	if b := g.jobs.byID[id]; b != nil {
		return b, nil
	}
	if g.jobs.dir == "" || strings.ContainsAny(id, `/\.`) {
		return nil, fmt.Errorf("no build %q", id)
	}
	// A build from before this gateway process started: what it had finished is on disk.
	path := filepath.Join(g.jobs.dir, id+".json")
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("no build %q", id)
	}
	b := &buildJob{path: path, inflight: map[string]string{}, tried: map[string][]string{}, done: make(chan struct{})}
	b.cond = sync.NewCond(&b.mu)
	if json.Unmarshal(data, b) != nil || b.Parts == nil {
		return nil, fmt.Errorf("build %q could not be read", id)
	}
	if b.Missing == nil {
		b.Missing = map[string]string{}
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

func (b *buildJob) save() { b.mu.Lock(); defer b.mu.Unlock(); b.saveLocked() }

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

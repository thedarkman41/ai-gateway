package provider

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
	"time"

	"github.com/creack/pty"
	"github.com/hinshun/vt10x"
)

// tty drives an interactive terminal app that has no way to take a prompt on its command line. It
// starts the app in a pseudo-terminal, waits for it to settle, types the prompt, waits for the
// reply to finish and reads it off the rendered screen. Each ask is a fresh session.
//
// Everything app-specific (when it is ready, when it is done, which screen lines are the reply) is
// in the provider's config entry, so a UI change is a config edit.
type tty struct {
	cfg        Config
	doneRe     *regexp.Regexp
	startRe    *regexp.Regexp
	endRe      *regexp.Regexp
	dropRes    []*regexp.Regexp
	cols, rows int
}

func newTTY(c Config) (*tty, error) {
	p := &tty{cfg: c, cols: c.Cols, rows: c.Rows}
	if p.cols <= 0 {
		p.cols = 120
	}
	if p.rows <= 0 {
		p.rows = 200 // tall, so a long reply stays on screen
	}
	var err error
	compile := func(s string) *regexp.Regexp {
		if s == "" || err != nil {
			return nil
		}
		var re *regexp.Regexp
		re, err = regexp.Compile(s)
		return re
	}
	p.doneRe, p.startRe, p.endRe = compile(c.DoneRegex), compile(c.ReplyStart), compile(c.ReplyEnd)
	for _, d := range c.Drop {
		p.dropRes = append(p.dropRes, compile(d))
	}
	if err != nil {
		return nil, fmt.Errorf("%s: %w", c.ID, err)
	}
	return p, nil
}

func (p *tty) Config() Config { return p.cfg }

func (p *tty) Available() error {
	if _, err := exec.LookPath(p.cfg.Bin); err != nil {
		return fmt.Errorf("%s is not installed (or not on PATH)", p.cfg.Bin)
	}
	return nil
}

func (p *tty) Models(context.Context) []string {
	if ms := order(p.cfg.DefaultModel, p.cfg.StaticModels); len(ms) > 0 {
		return ms
	}
	return []string{"default"}
}

func (p *tty) Ask(ctx context.Context, r Request) (string, error) {
	res, err := p.Session(ctx, r)
	return res.Reply, err
}

// TTYResult is what one terminal session produced: the reply, and the whole final screen for
// tuning the entry's patterns.
type TTYResult struct {
	Reply  string
	Screen string
}

// Screener is a provider that reads its reply off a terminal screen and can show that screen.
type Screener interface {
	Session(ctx context.Context, r Request) (TTYResult, error)
}

// screen is the emulated terminal plus the bookkeeping the wait loops need.
type screen struct {
	mu    sync.Mutex
	term  vt10x.Terminal
	bytes int
	last  time.Time
	done  bool // the app exited
}

func (s *screen) snapshot() (text string, n int, idle time.Duration, done bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	lines := strings.Split(s.term.String(), "\n")
	for i := range lines {
		lines[i] = strings.TrimRight(lines[i], " ")
	}
	return strings.TrimRight(strings.Join(lines, "\n"), "\n"), s.bytes, time.Since(s.last), s.done
}

// Session runs one ask and returns the reply with the final screen.
func (p *tty) Session(ctx context.Context, r Request) (TTYResult, error) {
	path, err := exec.LookPath(p.cfg.Bin)
	if err != nil {
		return TTYResult{}, p.Available()
	}
	scratch, cleanup, err := newScratch(p.cfg)
	if err != nil {
		return TTYResult{}, err
	}
	defer cleanup() // runs after the app has been closed, below
	var args []string
	for _, a := range p.cfg.Args {
		args = append(args, strings.ReplaceAll(a, "{scratch}", scratch))
	}
	if r.Model != "" && r.Model != "default" {
		for _, a := range p.cfg.ModelArgs {
			args = append(args, strings.ReplaceAll(a, "{model}", r.Model))
		}
	}
	started := time.Now()
	cmd := exec.Command(path, args...)
	cmd.Dir = scratch
	cmd.Env = append(childEnv(p.cfg.Env), "TERM=xterm-256color")
	f, err := pty.StartWithSize(cmd, &pty.Winsize{Cols: uint16(p.cols), Rows: uint16(p.rows)})
	if err != nil {
		return TTYResult{}, err
	}
	sc := &screen{term: vt10x.New(vt10x.WithSize(p.cols, p.rows)), last: time.Now()}
	go func() {
		buf := make([]byte, 32<<10)
		for {
			n, err := f.Read(buf)
			if n > 0 {
				sc.mu.Lock()
				sc.term.Write(buf[:n])
				sc.bytes += n
				sc.last = time.Now()
				sc.mu.Unlock()
				answerQueries(f, buf[:n])
			}
			if err != nil {
				sc.mu.Lock()
				sc.done = true
				sc.mu.Unlock()
				return
			}
		}
	}()
	defer func() {
		f.Write([]byte(firstNonEmpty(p.cfg.Quit, "\x03\x03")))
		exited := make(chan struct{})
		go func() { cmd.Wait(); close(exited) }()
		select {
		case <-exited:
		case <-time.After(2 * time.Second):
			cmd.Process.Kill()
			<-exited
		}
		f.Close()
	}()

	// 1. Wait for the app to draw its UI and go quiet.
	if err := p.wait(ctx, sc, 60*time.Second, func(text string, n int, idle time.Duration) bool {
		return n > 0 && idle >= seconds(p.cfg.ReadyIdleSeconds, 2)
	}); err != nil {
		text, _, _, _ := sc.snapshot()
		return TTYResult{Screen: text}, fmt.Errorf("did not become ready: %w", err)
	}

	// 2. Type the prompt. Bracketed paste keeps a multi-line prompt from being submitted line by line.
	prompt := p.cfg.withPreamble(Flatten(r.Messages))
	if p.cfg.NoPaste {
		f.Write([]byte(strings.Join(strings.Fields(prompt), " ")))
	} else {
		f.Write([]byte("\x1b[200~" + prompt + "\x1b[201~"))
	}
	time.Sleep(300 * time.Millisecond)
	_, typed, _, _ := sc.snapshot()
	f.Write([]byte(firstNonEmpty(p.cfg.Submit, "\r")))
	submitted := time.Now()

	// 3. Wait for the reply to finish: the done pattern on screen, or output gone quiet.
	err = p.wait(ctx, sc, 0, func(text string, n int, idle time.Duration) bool {
		if n <= typed || time.Since(submitted) < seconds(p.cfg.MinReplySeconds, 2) {
			return false
		}
		if p.doneRe != nil { // the pattern, then a short quiet spell so a still-drawing screen is not read
			below, _ := p.afterPrompt(text, prompt)
			return p.doneRe.MatchString(below) && idle >= seconds(p.cfg.DoneIdleSeconds, 1)
		}
		return idle >= seconds(p.cfg.DoneIdleSeconds, 6)
	})
	text, _, _, _ := sc.snapshot()
	res := TTYResult{Screen: text}
	if err != nil && !errors.Is(err, errExited) {
		return res, fmt.Errorf("reply did not finish: %w", err)
	}
	// The app's own saved conversation is the better source: it has the whole reply however long,
	// and does not depend on the screen layout. The screen is the fallback.
	if reply := p.fromTranscript(ctx, scratch, prompt, started); reply != "" {
		res.Reply = reply
		return res, nil
	}
	var found bool
	res.Reply, found = p.extract(text, prompt)
	if !found {
		return res, errors.New("the start of the reply is not on the screen: it was too long and scrolled off, or the app's layout changed")
	}
	if res.Reply == "" {
		return res, errors.New("no reply found on the screen (the entry's reply patterns may need tuning; see `ai-gateway try`)")
	}
	return res, nil
}

var errExited = errors.New("the app exited")

// wait polls the screen until ok reports true. max 0 means only the context bounds it.
func (p *tty) wait(ctx context.Context, sc *screen, max time.Duration, ok func(text string, n int, idle time.Duration) bool) error {
	start := time.Now()
	for {
		text, n, idle, done := sc.snapshot()
		if ok(text, n, idle) {
			return nil
		}
		if done {
			return errExited
		}
		if max > 0 && time.Since(start) > max {
			return fmt.Errorf("still busy after %s", max)
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(100 * time.Millisecond):
		}
	}
}

// afterPrompt returns the screen below the line where the typed prompt (or reply_start) last
// appears, and whether that line was found.
func (p *tty) afterPrompt(text, prompt string) (string, bool) {
	lines := strings.Split(text, "\n")
	start := -1
	if p.startRe != nil {
		for i, l := range lines {
			if p.startRe.MatchString(l) {
				start = i
			}
		}
	} else {
		head := strings.Join(strings.Fields(prompt), " ")
		if len(head) > 24 {
			head = head[:24]
		}
		for i, l := range lines {
			if strings.Contains(strings.Join(strings.Fields(l), " "), head) {
				start = i
			}
		}
	}
	return strings.Join(lines[start+1:], "\n"), start >= 0
}

// extract returns the reply lines: below the prompt, up to reply_end, without dropped lines, and
// with the indent the app adds to every line removed.
func (p *tty) extract(text, prompt string) (string, bool) {
	below, found := p.afterPrompt(text, prompt)
	var out []string
	for _, l := range strings.Split(below, "\n") {
		if p.endRe != nil && p.endRe.MatchString(l) {
			break
		}
		drop := false
		for _, re := range p.dropRes {
			drop = drop || re.MatchString(l)
		}
		if !drop {
			out = append(out, l)
		}
	}
	indent := -1
	for _, l := range out {
		if t := strings.TrimLeft(l, " "); t != "" && (indent < 0 || len(l)-len(t) < indent) {
			indent = len(l) - len(t)
		}
	}
	for i, l := range out {
		if len(l) >= indent && indent > 0 {
			out[i] = l[indent:]
		}
	}
	return strings.TrimSpace(strings.Join(out, "\n")), found
}

// fromTranscript reads the reply from the conversation file the app saved for this session, in the
// format freebuff and codebuff write (chat-messages.json): a list of messages, where the reply is
// the text blocks of the "ai" messages after our "user" message. It waits a few seconds for the
// file to show a finished reply, and returns "" when there is none.
func (p *tty) fromTranscript(ctx context.Context, scratch, prompt string, started time.Time) string {
	if p.cfg.Transcript == "" {
		return ""
	}
	pattern := expandHome(strings.ReplaceAll(p.cfg.Transcript, "{scratch_base}", filepath.Base(scratch)))
	want := strings.Join(strings.Fields(prompt), " ")
	for deadline := time.Now().Add(4 * time.Second); ; {
		files, _ := filepath.Glob(pattern)
		for _, file := range files {
			st, err := os.Stat(file)
			if err != nil || st.ModTime().Before(started.Add(-2*time.Second)) {
				continue // an older session's conversation
			}
			b, err := os.ReadFile(file)
			if err != nil {
				continue
			}
			if reply, ok := transcriptReply(b, want); ok {
				if !p.cfg.TranscriptKeep {
					os.RemoveAll(filepath.Dir(file))
				}
				return reply
			}
		}
		if time.Now().After(deadline) || ctx.Err() != nil {
			return ""
		}
		time.Sleep(250 * time.Millisecond)
	}
}

// transcriptReply returns the finished reply to the user message equal to want (whitespace collapsed).
func transcriptReply(data []byte, want string) (string, bool) {
	var msgs []struct {
		Variant    string `json:"variant"`
		Content    string `json:"content"`
		IsComplete bool   `json:"isComplete"`
		Blocks     []struct {
			Type     string `json:"type"`
			Content  string `json:"content"`
			TextType string `json:"textType"`
		} `json:"blocks"`
	}
	if json.Unmarshal(data, &msgs) != nil {
		return "", false
	}
	at := -1
	for i, m := range msgs {
		if m.Variant == "user" && strings.Join(strings.Fields(m.Content), " ") == want {
			at = i
		}
	}
	if at < 0 {
		return "", false
	}
	var parts []string
	complete := false
	for _, m := range msgs[at+1:] {
		if m.Variant != "ai" {
			continue
		}
		complete = m.IsComplete
		if strings.TrimSpace(m.Content) != "" {
			parts = append(parts, strings.TrimSpace(m.Content))
		}
		for _, b := range m.Blocks {
			if b.Type == "text" && (b.TextType == "" || b.TextType == "text") && strings.TrimSpace(b.Content) != "" {
				parts = append(parts, strings.TrimSpace(b.Content))
			}
		}
	}
	reply := strings.Join(parts, "\n\n")
	return reply, complete && reply != ""
}

// answerQueries replies to the terminal queries a full-screen app sends at startup (cursor position,
// device attributes, colours). An app that gets no answer may hang or draw nothing.
func answerQueries(f *os.File, chunk []byte) {
	for q, a := range map[string]string{
		"\x1b[6n":     "\x1b[1;1R",
		"\x1b[c":      "\x1b[?1;2c",
		"\x1b[0c":     "\x1b[?1;2c",
		"\x1b]10;?":   "\x1b]10;rgb:ffff/ffff/ffff\x07",
		"\x1b]11;?":   "\x1b]11;rgb:0000/0000/0000\x07",
		"\x1b[?u":     "\x1b[?0u",
		"\x1b[?2026$": "\x1b[?2026;0$y",
	} {
		if bytes.Contains(chunk, []byte(q)) {
			f.Write([]byte(a))
		}
	}
}

func seconds(v, def float64) time.Duration {
	if v <= 0 {
		v = def
	}
	return time.Duration(v * float64(time.Second))
}

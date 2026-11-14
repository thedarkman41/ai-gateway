package provider

import (
	"context"
	"os"
	"strings"
	"testing"
	"time"
)

// A small full-screen app: clears the screen, asks the terminal where the cursor is (and waits for
// the answer), draws a prompt, reads a line, shows a spinner, then draws the reply and a status bar
// using cursor movement.
const fakeTUI = `
printf '\033[2J\033[H\033[1mFake TUI\033[0m v1\r\n'
stty -echo -icanon min 1 time 0; printf '\033[6n'; dd bs=1 count=6 >/dev/null 2>&1; stty sane
printf '> '
read line
printf 'thinking...'; sleep 0.4; printf '\r\033[K'
printf '\033[32mPONG\033[0m from the fake\r\n'
printf '  second line\r\n'
printf '\033[12;1H-- status: ready --'
sleep 30
`

func TestTTYSession(t *testing.T) {
	ps, err := ParseAll([]byte(`{"providers":[{"id":"fake","tier":"B","kind":"tty","bin":"sh","args":["-c",` + jsonString(fakeTUI) + `],
		"no_paste":true,"ready_idle_seconds":0.4,"done_idle_seconds":0.8,"min_reply_seconds":0.1,
		"reply_end":"^-- status","drop":["^\\s*$"],"rows":24,"cols":80,"disabled":true}]}`))
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	start := time.Now()
	res, err := ps[0].(*tty).Session(ctx, Request{Messages: []Message{{Role: "user", Content: "say\npong please"}}})
	if err != nil {
		t.Fatalf("%v\nscreen:\n%s", err, res.Screen)
	}
	if res.Reply != "PONG from the fake\n  second line" {
		t.Errorf("reply %q\nscreen:\n%s", res.Reply, res.Screen)
	}
	if !strings.Contains(res.Screen, "> say pong please") || !strings.Contains(res.Screen, "-- status: ready --") {
		t.Errorf("screen:\n%s", res.Screen)
	}
	if d := time.Since(start); d > 10*time.Second {
		t.Errorf("session took %s: the app was not closed promptly", d)
	}

	// Disabled entries are left out of Parse.
	if _, err := Parse([]byte(`{"providers":[{"id":"x","kind":"tty","bin":"sh","disabled":true}]}`)); err == nil {
		t.Error("Parse returned a disabled provider")
	}
}

func jsonString(s string) string {
	r := strings.NewReplacer(`\`, `\\`, `"`, `\"`, "\n", `\n`)
	return `"` + r.Replace(s) + `"`
}

func TestModelIDsKeepsKeylessChatModels(t *testing.T) {
	got := ModelIDs([]byte(`{"data":[
		{"id":"flaky","model_type":"chat","usage_based_only":false,"availability_last_hour_percent":61.5},
		{"id":"paid","model_type":"chat","usage_based_only":true,"availability_last_hour_percent":100},
		{"id":"painter","model_type":"image","usage_based_only":false,"availability_last_hour_percent":99},
		{"id":"steady","model_type":"chat","usage_based_only":false,"availability_last_hour_percent":98.2}]}`))
	if strings.Join(got, ",") != "steady,flaky" {
		t.Errorf("got %v, want [steady flaky]", got)
	}
	// A plain list without those fields is returned whole, in order.
	if got := ModelIDs([]byte(`[{"name":"b"},{"id":"a"}]`)); strings.Join(got, ",") != "b,a" {
		t.Errorf("plain list: %v", got)
	}
}

// The freebuff 0.2.36 screen, as captured on betelgeuse by `ai-gateway try freebuff`.
const freebuffScreen = ` ███ ██▀    ██▀    ▀█████   ▀████▀  ██████▀   ▀████▀   ██    ██

 ┌─────────────────────────────────┐
 │ May use data for AI training    │
 │ Session active                  │
 └─────────────────────────────────┘

   [12:11 PM]
   Summarise this, then
   reply with one word ⎘

  PONG, since you asked.

    - an indented point
                                                              ⎘ • 4s • △▽



 14K (1%)                                                ✕ End session
╭────────────────────────────────────────────────────────────────────╮
│  ▍Enter a coding task or / for commands                            │
╰────────────────────────────────────────────────────────────────────╯
 DeepSeek V4.1 Flash Fast • high · /tmp/ai-gateway-scratch · /model
 to change · Chat: Summarise this, then reply with one word
 ← for history · ? for help`

func TestFreebuffEntryReadsItsScreen(t *testing.T) {
	ps, err := Parse(providersJSON(t))
	if err != nil {
		t.Fatal(err)
	}
	var fb *tty
	for _, p := range ps {
		if p.Config().ID == "freebuff" {
			fb = p.(*tty)
		}
	}
	if fb == nil {
		t.Fatal("no enabled freebuff entry")
	}
	prompt := "Summarise this, then\nreply with one word"
	reply, found := fb.extract(freebuffScreen, prompt)
	if !found || reply != "PONG, since you asked.\n\n  - an indented point" {
		t.Errorf("reply %q found=%v", reply, found)
	}
	below, _ := fb.afterPrompt(freebuffScreen, prompt)
	if !fb.doneRe.MatchString(below) {
		t.Error("finished reply not recognised")
	}
	// Mid-reply (no duration line yet) is not finished; a screen without the prompt is an error.
	mid := freebuffScreen[:strings.Index(freebuffScreen, "    - an indented")]
	if below, _ := fb.afterPrompt(mid, prompt); fb.doneRe.MatchString(below) {
		t.Error("unfinished reply taken as finished")
	}
	if _, found := fb.extract("  tail of a long reply\n  ⎘ • 9s • △▽", prompt); found {
		t.Error("a reply whose start scrolled off was accepted")
	}
}

func providersJSON(t *testing.T) []byte {
	t.Helper()
	b, err := os.ReadFile("../../providers/providers.json")
	if err != nil {
		t.Fatal(err)
	}
	return b
}

// With a transcript configured, the reply comes from the app's saved conversation (whole, however
// long), not from the screen, and the saved conversation is removed afterwards.
func TestTTYReadsTranscript(t *testing.T) {
	dir := t.TempDir()
	app := `
printf 'App\r\n> '
read line
mkdir -p "$CHATS/one"
printf '[{"variant":"ai","blocks":[{"type":"mode-divider"}]},{"variant":"user","content":"%s"},{"variant":"ai","content":"","blocks":[{"type":"text","content":"thinking aloud","textType":"reasoning"},{"type":"text","content":"The whole reply.","textType":"text"},{"type":"tool","content":"ignored"}],"isComplete":true}]' "$line" > "$CHATS/one/chat-messages.json"
printf 'only part of the reply fits here\r\n'
sleep 30
`
	ps, err := Parse([]byte(`{"providers":[{"id":"fake","tier":"B","kind":"tty","bin":"sh","args":["-c",` + jsonString(app) + `],
		"env":{"CHATS":"` + dir + `"},"transcript":"` + dir + `/*/chat-messages.json",
		"no_paste":true,"ready_idle_seconds":0.4,"done_idle_seconds":0.8,"min_reply_seconds":0.1,"rows":24,"cols":80}]}`))
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	res, err := ps[0].(*tty).Session(ctx, Request{Messages: []Message{{Role: "user", Content: "what is  it"}}})
	if err != nil || res.Reply != "The whole reply." {
		t.Errorf("reply %q, %v\nscreen:\n%s", res.Reply, err, res.Screen)
	}
	if _, err := os.Stat(dir + "/one"); !os.IsNotExist(err) {
		t.Errorf("saved conversation was not removed: %v", err)
	}

	// The real file from betelgeuse, and an unfinished one.
	real := `[{"id":"divider-1","variant":"ai","content":"","blocks":[{"type":"mode-divider","mode":"LITE"}],"timestamp":"12:11 PM"},{"id":"user-1","variant":"user","content":"Reply with exactly one word: PONG","timestamp":"12:11 PM"},{"id":"ai-1","variant":"ai","content":"","blocks":[{"type":"text","content":"PONG","textType":"text"}],"timestamp":"12:11 PM","metadata":{"allowInlineAds":true},"isComplete":true,"completionTime":"4s","credits":0}]`
	if r, ok := transcriptReply([]byte(real), "Reply with exactly one word: PONG"); !ok || r != "PONG" {
		t.Errorf("real transcript: %q %v", r, ok)
	}
	if _, ok := transcriptReply([]byte(strings.Replace(real, `"isComplete":true`, `"isComplete":false`, 1)), "Reply with exactly one word: PONG"); ok {
		t.Error("an unfinished reply was accepted")
	}
	if _, ok := transcriptReply([]byte(real), "another prompt"); ok {
		t.Error("another session's conversation was accepted")
	}
}

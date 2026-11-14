package provider

import (
	"context"
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

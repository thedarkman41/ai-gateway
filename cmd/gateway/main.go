// Command gateway is the keyless AI gateway: an MCP server that lets a client ask other AI models.
package main

import (
	"bufio"
	"context"
	"encoding/json"
	"encoding/xml"
	"flag"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	"github.com/thedarkman41/ai-gateway/internal/gateway"
	"github.com/thedarkman41/ai-gateway/internal/mcp"
	"github.com/thedarkman41/ai-gateway/internal/privacy"
	"github.com/thedarkman41/ai-gateway/internal/probe"
	"github.com/thedarkman41/ai-gateway/internal/provider"
	"github.com/thedarkman41/ai-gateway/providers"
	"golang.org/x/term"
)

const version = "0.2.0"

const usage = `usage: gateway <command> [flags]

  serve    run the MCP server on stdio, or on HTTP with -http 127.0.0.1:8765
  check    send a test prompt through every configured provider
  models   print the providers, their state and their models
  privacy  privacy show | check [file] | block <dir> | allow <dir> | remove <dir>: what may leave this
           machine. check shows what would be redacted from a file (or standard input) and whether
           it would be kept home
  service  service install | uninstall | status [-addr 127.0.0.1:8765]: run the gateway in the background
           on this Mac (launchd), serving MCP and the OpenAI-compatible endpoint on that address
  key      key list | key set <provider> | key remove <provider>: free API keys, kept in a private
           file in the state directory, never in the config or the repository
  reset    reset [provider]: clear the rests on one provider, or on all of them
  consensus  consensus [-agents n] <prompt>: one function written by the strongest free agents;
             prints the version the majority agrees on
  build      build [-fail ids] <prompt>...: a program of several functions through the greedy queue;
             prints the assembled program
  try      try <provider> [prompt]: one ask straight to a provider, disabled ones included;
           for a terminal-driven provider it also prints the screen the reply was read from
  probe    survey the candidate providers (providers/candidates.json)

serve, check and models take -config <file> and -state <dir>.`

func main() {
	log.SetFlags(0)
	log.SetPrefix("gateway: ")
	if len(os.Args) < 2 {
		fmt.Fprintln(os.Stderr, usage)
		os.Exit(2)
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	switch os.Args[1] {
	case "serve":
		fs := flag.NewFlagSet("serve", flag.ExitOnError)
		addr := fs.String("http", "", "serve HTTP on this address (127.0.0.1 only) instead of stdio")
		g := open(fs, os.Args[2:])
		srv := &mcp.Server{Name: "ai-gateway", Version: version, Tools: g.Tools()}
		if *addr == "" {
			if err := srv.ServeStdio(ctx, os.Stdin, os.Stdout); err != nil {
				log.Fatal(err)
			}
			return
		}
		host, _, err := net.SplitHostPort(*addr)
		if err != nil || (host != "127.0.0.1" && host != "localhost" && host != "::1") {
			log.Fatalf("-http must be a loopback address such as 127.0.0.1:8765, not %q", *addr)
		}
		mux := http.NewServeMux()
		mux.Handle("/mcp", srv.HTTPHandler())
		mux.Handle("/v1/", mcp.LocalOnly(g.OpenAIHandler()))
		mux.Handle("/healthz", gateway.HealthHandler())
		hs := &http.Server{Addr: *addr, Handler: mux}
		go func() { <-ctx.Done(); hs.Close() }()
		log.Printf("serving on http://%s: MCP at /mcp, OpenAI-compatible chat at /v1/chat/completions, health at /healthz", *addr)
		if err := hs.ListenAndServe(); err != http.ErrServerClosed {
			log.Fatal(err)
		}
	case "check":
		g := open(flag.NewFlagSet("check", flag.ExitOnError), os.Args[2:])
		passed := 0
		fmt.Println("| Provider | Result | Time | Detail |\n|---|---|---|---|")
		for _, r := range g.Check(ctx) {
			result, detail := "failed", r.Detail
			if r.OK {
				result, detail = "answered", r.Used
				passed++
			}
			fmt.Printf("| %s | %s | %.1fs | %s |\n", r.ID, result, r.Seconds, detail)
		}
		if passed == 0 {
			os.Exit(1)
		}
	case "models":
		g := open(flag.NewFlagSet("models", flag.ExitOnError), os.Args[2:])
		fmt.Print(g.Status(ctx))
	case "try":
		runTry(ctx, os.Args[2:])
	case "reset":
		fs := flag.NewFlagSet("reset", flag.ExitOnError)
		g := open(fs, os.Args[2:])
		if err := g.Reset(fs.Arg(0)); err != nil {
			log.Fatal(err)
		}
		fmt.Print(g.Status(ctx))
	case "ask":
		fs := flag.NewFlagSet("ask", flag.ExitOnError)
		g := open(fs, os.Args[2:])
		if fs.NArg() < 1 {
			log.Fatal("usage: gateway ask <model> [prompt]   (model: auto, a provider, or provider/model)")
		}
		prompt := "Reply with exactly one word: PONG"
		if fs.NArg() > 1 {
			prompt = fs.Arg(1)
		}
		ans, err := g.Ask(ctx, fs.Arg(0), "", prompt, "")
		if err != nil {
			log.Fatal(err)
		}
		fmt.Printf("[answered by %s]\n\n%s\n", ans.Used, ans.Text)
	case "key":
		runKey(os.Args[2:])
	case "service":
		runService(os.Args[2:])
	case "privacy":
		runPrivacy(os.Args[2:])
	case "consensus":
		runConsensus(ctx, os.Args[2:])
	case "build":
		runBuild(ctx, os.Args[2:])
	case "probe":
		runProbe(ctx, os.Args[2:])
	default:
		fmt.Fprintln(os.Stderr, usage)
		os.Exit(2)
	}
}

// open parses the shared flags and builds the gateway. The provider list is -config if given, else
// <state>/providers.json if it exists, else the list built into the binary.
func open(fs *flag.FlagSet, args []string) *gateway.Gateway {
	data, from, state := config(fs, args)
	ps, err := provider.Parse(data)
	if err != nil {
		log.Fatalf("%s providers: %v", from, err)
	}
	g := gateway.New(ps, gateway.OpenLedger(filepath.Join(state, "ledger.json")))
	g.SetJobsDir(filepath.Join(state, "jobs"))
	// The privacy rules, and the gateway's own API keys as values to redact wherever they turn up.
	g.SetPrivacy(privacy.Open(filepath.Join(state, "privacy.json")), func() []string {
		var known []string
		for _, v := range readKeys(filepath.Join(state, "keys.json")) {
			known = append(known, v)
		}
		for _, p := range ps {
			if env := p.Config().KeyEnv; env != "" && os.Getenv(env) != "" {
				known = append(known, os.Getenv(env))
			}
		}
		return known
	})
	g.Logf = log.Printf
	return g
}

// config parses the shared flags and returns the providers document, where it came from and the
// state directory.
func config(fs *flag.FlagSet, args []string) (data []byte, from, state string) {
	file := fs.String("config", "", "providers file (default: <state>/providers.json, else built in)")
	dir := fs.String("state", defaultState(), "directory for the quota ledger")
	fs.Parse(args)

	data, from = providers.Active, "built-in"
	path := *file
	if path == "" {
		path = filepath.Join(*dir, "providers.json")
	}
	if b, err := os.ReadFile(path); err == nil {
		data, from = b, path
	} else if *file != "" {
		log.Fatal(err)
	}
	// A provider's free API key comes from its environment variable, else from the private key file.
	keys := filepath.Join(*dir, "keys.json")
	provider.KeyLookup = func(id, env string) string {
		if v := os.Getenv(env); v != "" {
			return v
		}
		return readKeys(keys)[id]
	}
	return data, from, *dir
}

// runTry sends one prompt straight to one provider, bypassing the ledger, so an entry can be tuned
// before it is turned on.
func runTry(ctx context.Context, args []string) {
	fs := flag.NewFlagSet("try", flag.ExitOnError)
	model := fs.String("model", "", "model to pass (default: the provider's own)")
	data, from, _ := config(fs, args)
	if fs.NArg() < 1 {
		log.Fatal("usage: gateway try [-model m] <provider> [prompt]")
	}
	ps, err := provider.ParseAll(data)
	if err != nil {
		log.Fatalf("%s providers: %v", from, err)
	}
	prompt := "Reply with exactly one word: PONG"
	if fs.NArg() > 1 {
		prompt = fs.Arg(1)
	}
	for _, p := range ps {
		if p.Config().ID != fs.Arg(0) {
			continue
		}
		if *model == "" {
			*model = p.Models(ctx)[0]
		}
		ctx, cancel := context.WithTimeout(ctx, p.Config().Timeout())
		defer cancel()
		req := provider.Request{Model: *model, Messages: []provider.Message{{Role: "user", Content: prompt}}}
		var reply string
		if s, ok := p.(provider.Screener); ok {
			var res provider.TTYResult
			res, err = s.Session(ctx, req)
			reply = res.Reply
			fmt.Printf("===== screen =====\n%s\n", res.Screen)
		} else {
			reply, err = p.Ask(ctx, req)
		}
		fmt.Printf("===== reply (%s) =====\n%s\n", *model, reply)
		if err != nil {
			log.Fatal(err)
		}
		return
	}
	log.Fatalf("no provider %q in the %s list", fs.Arg(0), from)
}

// runConsensus runs one vote from the terminal: the strongest free agents each write the function.
func runConsensus(ctx context.Context, args []string) {
	fs := flag.NewFlagSet("consensus", flag.ExitOnError)
	n := fs.Int("agents", 0, "how many agents vote (default 5)")
	g := open(fs, args)
	if fs.NArg() != 1 {
		log.Fatal("usage: gateway consensus [-agents n] <prompt>")
	}
	start := time.Now()
	v, err := g.Consensus(ctx, gateway.Request{Prompt: fs.Arg(0), Agents: *n})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("%s\n[%.0fs]\n", v, time.Since(start).Seconds())
}

// runBuild runs a build from the terminal the way Bill would: each prompt is a task in the queue;
// when the program comes back, -fail sends named parts through rework to show that path.
func runBuild(ctx context.Context, args []string) {
	fs := flag.NewFlagSet("build", flag.ExitOnError)
	header := fs.String("header", "", "code placed before the functions")
	fail := fs.String("fail", "", "comma-separated task ids to send through rework afterwards, as if they had failed to compile")
	g := open(fs, args)
	if fs.NArg() == 0 {
		log.Fatal("usage: gateway build [-header code] [-fail task1,task3] <prompt>...   (tasks are named task1, task2, ...)")
	}
	var tasks []gateway.Task
	for i, p := range fs.Args() {
		tasks = append(tasks, gateway.Task{ID: fmt.Sprintf("task%d", i+1), Prompt: p})
	}
	start := time.Now()
	id, plan, err := g.Build(tasks, "", *header, "")
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("%s\n\n", plan)
	wait := func() {
		for {
			out, err := g.Collect(ctx, id, 20*time.Second)
			if err != nil {
				log.Fatal(err)
			}
			fmt.Printf("[%3.0fs] %s\n\n", time.Since(start).Seconds(), out)
			if !strings.Contains(out, "is still running") {
				return
			}
		}
	}
	wait()
	if *fail != "" {
		failed := map[string]string{}
		for _, t := range strings.Split(*fail, ",") {
			failed[strings.TrimSpace(t)] = "SyntaxError: invalid syntax (simulated by -fail)"
		}
		out, err := g.Rework(id, failed)
		if err != nil {
			log.Fatal(err)
		}
		fmt.Printf("%s\n\n", out)
		wait()
	}
}

// hint describes a key without revealing it: its length and first four characters, enough to see
// that the right provider's key was pasted (Groq's start gsk_, OpenRouter's sk-o, NVIDIA's nvap).
func hint(key string) string {
	key = strings.TrimSpace(key)
	return fmt.Sprintf("(%d characters, starts %s…)", len(key), key[:min(4, len(key))])
}

// runPrivacy shows and edits the privacy rules, and tests text against them.
func runPrivacy(args []string) {
	fs := flag.NewFlagSet("privacy", flag.ExitOnError)
	if len(args) == 0 {
		log.Fatal("usage: gateway privacy show | check [file] | block <dir> | allow <dir> | remove <dir>")
	}
	_, _, state := config(fs, args[1:])
	store := privacy.Open(filepath.Join(state, "privacy.json"))
	pol, err := store.Policy()
	if err != nil {
		log.Fatalf("%v\nUntil this is fixed the gateway sends nothing to outside providers.", err)
	}
	dir := func() string {
		if fs.NArg() != 1 {
			log.Fatalf("usage: gateway privacy %s <dir>", args[0])
		}
		return privacy.Expand(fs.Arg(0))
	}
	without := func(list []string, d string) []string {
		var out []string
		for _, x := range list {
			if x != d {
				out = append(out, x)
			}
		}
		return out
	}
	show := func() {
		pol, _ := store.Policy()
		fmt.Println("Always: API keys, tokens, private keys, passwords and credentials in URLs are redacted before a prompt")
		fmt.Println("leaves this machine, and a prompt carrying a credentials file (.env, *.pem, ~/.ssh/...) is kept home.")
		fmt.Printf("\nBlocked directories (files under these never leave): %s\n", orNone(pol.BlockedPaths))
		fmt.Printf("Allowed directories (if any are listed, only files under these may leave): %s\n", orNone(pol.AllowedPaths))
		fmt.Printf("Extra patterns to redact: %s\n", orNone(pol.ExtraSecrets))
	}
	switch args[0] {
	case "show":
		show()
	case "block":
		d := dir()
		if err := store.Edit(func(p *privacy.Policy) { p.BlockedPaths = append(without(p.BlockedPaths, d), d) }); err != nil {
			log.Fatal(err)
		}
		show()
	case "allow":
		d := dir()
		if err := store.Edit(func(p *privacy.Policy) { p.AllowedPaths = append(without(p.AllowedPaths, d), d) }); err != nil {
			log.Fatal(err)
		}
		show()
	case "remove":
		d := dir()
		if err := store.Edit(func(p *privacy.Policy) {
			p.BlockedPaths, p.AllowedPaths = without(p.BlockedPaths, d), without(p.AllowedPaths, d)
		}); err != nil {
			log.Fatal(err)
		}
		show()
	case "check":
		var data []byte
		name := "standard input"
		if fs.NArg() == 1 {
			name = privacy.Expand(fs.Arg(0))
			data, err = os.ReadFile(name)
		} else {
			data, err = io.ReadAll(os.Stdin)
		}
		if err != nil {
			log.Fatal(err)
		}
		text := string(data)
		// A file is checked as Bill would attach it to a hand-off, so the path rules apply to it.
		probe := text
		if fs.NArg() == 1 {
			probe = "FILE " + name + " (its full current text, supplied by the harness):\n" + text
		}
		if why := pol.Private(probe); why != "" {
			fmt.Printf("KEPT HOME: %s.\nA prompt carrying this goes to local models only.\n", why)
			return
		}
		clean, found := pol.Scrub(text, nil)
		if s := privacy.Summary(found); s != "" {
			fmt.Printf("WOULD LEAVE with %s redacted. As it would be sent:\n\n%s\n", s, clean)
			return
		}
		fmt.Println("WOULD LEAVE unchanged: nothing in it matches a secret pattern.")
	default:
		log.Fatal("usage: gateway privacy show | check [file] | block <dir> | allow <dir> | remove <dir>")
	}
}

func orNone(list []string) string {
	if len(list) == 0 {
		return "none"
	}
	return strings.Join(list, ", ")
}

const serviceLabel = "local.ai-gateway"

// servicePlist is the launchd job: the gateway serving HTTP on addr, started at login and kept
// running. launchd gives a job almost no PATH, so the PATH of the shell that installs it is written
// in: that is where the agents' CLIs are found.
func servicePlist(binary, addr, path, home, logFile string) string {
	esc := func(s string) string {
		var b strings.Builder
		xml.EscapeText(&b, []byte(s))
		return b.String()
	}
	return fmt.Sprintf(`<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
	<key>Label</key><string>%s</string>
	<key>ProgramArguments</key>
	<array><string>%s</string><string>serve</string><string>-http</string><string>%s</string></array>
	<key>EnvironmentVariables</key>
	<dict><key>PATH</key><string>%s</string><key>HOME</key><string>%s</string></dict>
	<key>RunAtLoad</key><true/>
	<key>KeepAlive</key><true/>
	<key>StandardOutPath</key><string>%s</string>
	<key>StandardErrorPath</key><string>%s</string>
</dict>
</plist>
`, serviceLabel, esc(binary), esc(addr), esc(path), esc(home), esc(logFile), esc(logFile))
}

func healthy(addr string) bool {
	c := http.Client{Timeout: 2 * time.Second}
	resp, err := c.Get("http://" + addr + "/healthz")
	if err != nil {
		return false
	}
	resp.Body.Close()
	return resp.StatusCode == 200
}

// runService installs, removes or checks the background gateway.
func runService(args []string) {
	fs := flag.NewFlagSet("service", flag.ExitOnError)
	addr := fs.String("addr", "127.0.0.1:8765", "address to serve on (loopback only)")
	show := fs.Bool("print", false, "with install: print the launchd job instead of installing it")
	if len(args) == 0 {
		log.Fatal("usage: gateway service install | uninstall | status [-addr 127.0.0.1:8765]")
	}
	fs.Parse(args[1:])
	home, _ := os.UserHomeDir()
	plist := filepath.Join(home, "Library", "LaunchAgents", serviceLabel+".plist")
	domain := fmt.Sprintf("gui/%d", os.Getuid())
	launchctl := func(a ...string) error { return exec.Command("/bin/launchctl", a...).Run() }
	switch args[0] {
	case "status":
		if healthy(*addr) {
			fmt.Printf("the gateway is running on http://%s\n", *addr)
			return
		}
		fmt.Printf("nothing is answering on http://%s\n", *addr)
		os.Exit(1)
	case "install":
		binary, err := os.Executable()
		if err != nil {
			log.Fatal(err)
		}
		if real, err := filepath.EvalSymlinks(binary); err == nil {
			binary = real
		}
		logFile := filepath.Join(defaultState(), "gateway.log")
		job := servicePlist(binary, *addr, os.Getenv("PATH"), home, logFile)
		if *show {
			fmt.Print(job)
			return
		}
		if runtime.GOOS != "darwin" {
			log.Fatal("service install is for macOS (launchd); elsewhere run: gateway serve -http " + *addr)
		}
		if err := os.MkdirAll(filepath.Dir(plist), 0o755); err != nil {
			log.Fatal(err)
		}
		os.MkdirAll(defaultState(), 0o700)
		if err := os.WriteFile(plist, []byte(job), 0o644); err != nil {
			log.Fatal(err)
		}
		launchctl("bootout", domain+"/"+serviceLabel) // replace a running copy; fails harmlessly when there is none
		if err := launchctl("bootstrap", domain, plist); err != nil {
			log.Fatalf("launchctl could not start the job (%v); the job file is %s", err, plist)
		}
		for i := 0; i < 20 && !healthy(*addr); i++ {
			time.Sleep(250 * time.Millisecond)
		}
		if !healthy(*addr) {
			log.Fatalf("the job was installed but nothing answers on http://%s; see %s", *addr, logFile)
		}
		fmt.Printf("the gateway is running on http://%s and will start at login\n  log:  %s\n  job:  %s\n", *addr, logFile, plist)
		fmt.Printf("After installing or signing in to another agent's CLI, or rebuilding the gateway, run this again.\n")
	case "uninstall":
		launchctl("bootout", domain+"/"+serviceLabel)
		os.Remove(plist)
		fmt.Println("the background gateway is stopped and removed")
	default:
		log.Fatal("usage: gateway service install | uninstall | status [-addr 127.0.0.1:8765]")
	}
}

func readKeys(path string) map[string]string {
	keys := map[string]string{}
	if b, err := os.ReadFile(path); err == nil {
		json.Unmarshal(b, &keys)
	}
	return keys
}

// runKey manages the free API keys some providers need. A key is typed into the terminal with the
// input hidden and stored in <state>/keys.json, readable only by the user.
func runKey(args []string) {
	fs := flag.NewFlagSet("key", flag.ExitOnError)
	data, from, state := config(fs, args)
	ps, err := provider.ParseAll(data)
	if err != nil {
		log.Fatalf("%s providers: %v", from, err)
	}
	path := filepath.Join(state, "keys.json")
	keys := readKeys(path)
	find := func(id string) provider.Config {
		for _, p := range ps {
			if c := p.Config(); c.ID == id && c.KeyEnv != "" {
				return c
			}
		}
		log.Fatalf("%q is not a provider that takes a key; see `gateway key list`", id)
		return provider.Config{}
	}
	save := func() {
		b, _ := json.MarshalIndent(keys, "", "  ")
		if err := os.MkdirAll(state, 0o700); err != nil {
			log.Fatal(err)
		}
		if err := os.WriteFile(path, b, 0o600); err != nil {
			log.Fatal(err)
		}
		os.Chmod(path, 0o600)
	}
	switch {
	case fs.NArg() == 1 && fs.Arg(0) == "list":
		fmt.Println("| Provider | Key | Where to create one |\n|---|---|---|")
		for _, p := range ps {
			c := p.Config()
			if c.KeyEnv == "" {
				continue
			}
			state := "not set"
			switch {
			case c.AccountEnv != "" && os.Getenv(c.AccountEnv) == "" && keys[c.ID+".account"] == "":
				state = "not set (needs an account id and a key)"
			case os.Getenv(c.KeyEnv) != "":
				state = "set in $" + c.KeyEnv + " " + hint(os.Getenv(c.KeyEnv))
				if keys[c.ID] != "" {
					state += "; the stored key is not used while that variable is set"
				}
			case keys[c.ID] != "":
				state = "set " + hint(keys[c.ID])
			case c.KeyOptional:
				state = "not set (optional: works without, higher limits with)"
			}
			fmt.Printf("| %s | %s | %s |\n", c.ID, state, c.KeyURL)
		}
	case fs.NArg() == 2 && fs.Arg(0) == "set":
		c := find(fs.Arg(1))
		var key string
		in := bufio.NewReader(os.Stdin)
		if c.AccountEnv != "" { // not a secret, but the provider's address needs it
			fmt.Printf("%s account id: ", c.ID)
			line, _ := in.ReadString('\n')
			if line = strings.TrimSpace(line); line == "" {
				log.Fatal("no account id entered; nothing changed")
			}
			keys[c.ID+".account"] = line
		}
		if term.IsTerminal(int(os.Stdin.Fd())) {
			fmt.Printf("Create a free key at %s\nPaste the %s key (input is hidden): ", c.KeyURL, c.ID)
			b, err := term.ReadPassword(int(os.Stdin.Fd()))
			fmt.Println()
			if err != nil {
				log.Fatal(err)
			}
			key = string(b)
		} else {
			b, _ := io.ReadAll(io.LimitReader(in, 4096))
			key = string(b)
		}
		if key = strings.TrimSpace(key); key == "" {
			log.Fatal("no key entered; nothing changed")
		}
		keys[c.ID] = key
		save()
		fmt.Printf("saved the %s key (%d characters) to %s\ntest it with: ai-gateway try %s\n", c.ID, len(key), path, c.ID)
	case fs.NArg() == 2 && fs.Arg(0) == "remove":
		c := find(fs.Arg(1))
		delete(keys, c.ID)
		delete(keys, c.ID+".account")
		save()
		fmt.Printf("removed the %s key\n", c.ID)
	default:
		log.Fatal("usage: gateway key list | key set <provider> | key remove <provider>")
	}
}

func defaultState() string {
	if d := os.Getenv("AI_GATEWAY_HOME"); d != "" {
		return d
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return ".ai-gateway"
	}
	return filepath.Join(home, ".ai-gateway")
}

func runProbe(ctx context.Context, args []string) {
	fs := flag.NewFlagSet("probe", flag.ExitOnError)
	file := fs.String("candidates", "providers/candidates.json", "candidates file")
	out := fs.String("out", "survey", "directory for survey.md and survey.json")
	timeout := fs.Duration("timeout", 45*time.Second, "per-provider timeout")
	askCLI := fs.Bool("ask-cli", false, "also send the prompt through installed CLIs (spends a little free quota)")
	fs.Parse(args)

	f, err := probe.Load(*file)
	if err != nil {
		log.Fatal(err)
	}
	now := time.Now()
	rs := probe.Run(ctx, f, probe.Options{Timeout: *timeout, AskCLI: *askCLI})
	md := probe.Markdown(rs, now)
	fmt.Print(md)

	if err := os.MkdirAll(*out, 0o755); err != nil {
		log.Fatal(err)
	}
	js, _ := json.MarshalIndent(rs, "", "  ")
	for name, data := range map[string][]byte{"survey.md": []byte(md), "survey.json": js} {
		if err := os.WriteFile(filepath.Join(*out, name), data, 0o644); err != nil {
			log.Fatal(err)
		}
	}
}

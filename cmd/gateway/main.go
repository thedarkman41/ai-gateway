// Command gateway is the keyless AI gateway: an MCP server that lets a client ask other AI models.
package main

import (
	"bufio"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"time"

	"github.com/thedarkman41/ai-gateway/internal/gateway"
	"github.com/thedarkman41/ai-gateway/internal/mcp"
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
  key      key list | key set <provider> | key remove <provider>: free API keys, kept in a private
           file in the state directory, never in the config or the repository
  reset    reset [provider]: clear the rests on one provider, or on all of them
  fanout   fanout [-all] <prompt>...: run the dispatch/collect loop from the terminal. Each prompt is
           a task spread over the free providers; -all sends one prompt to every free provider
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
		hs := &http.Server{Addr: *addr, Handler: mux}
		go func() { <-ctx.Done(); hs.Close() }()
		log.Printf("serving MCP at http://%s/mcp", *addr)
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
	case "key":
		runKey(os.Args[2:])
	case "fanout":
		runFanout(ctx, os.Args[2:])
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

// runFanout drives a real job from the terminal the way Bill would: dispatch, then collect results
// one at a time as they finish.
func runFanout(ctx context.Context, args []string) {
	fs := flag.NewFlagSet("fanout", flag.ExitOnError)
	all := fs.Bool("all", false, "send the one prompt to every free provider, to compare them")
	g := open(fs, args)
	prompts := fs.Args()
	if len(prompts) == 0 || (*all && len(prompts) != 1) {
		log.Fatal("usage: gateway fanout <prompt>...   or   gateway fanout -all <prompt>")
	}
	var tasks []gateway.Task
	if *all {
		for _, id := range g.Free() {
			tasks = append(tasks, gateway.Task{ID: id, Prompt: prompts[0], Models: []string{id}})
		}
		if len(tasks) == 0 {
			log.Fatal("no provider is free right now; see `gateway models`")
		}
	} else {
		for i, p := range prompts {
			tasks = append(tasks, gateway.Task{ID: fmt.Sprintf("task%d", i+1), Prompt: p})
		}
	}
	start := time.Now()
	id, plan, err := g.Dispatch(tasks)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("%s\n\n", plan)
	for {
		out, err := g.Collect(ctx, id, 4*time.Minute, 1)
		if err != nil {
			log.Fatal(err)
		}
		fmt.Printf("[%3.0fs] %s\n\n", time.Since(start).Seconds(), out)
		if strings.Contains(out, "is complete") || strings.Contains(out, "Lost when") {
			return
		}
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
				state = "set (in $" + c.KeyEnv + ")"
			case keys[c.ID] != "":
				state = "set"
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

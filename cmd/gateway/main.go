// Command gateway is the keyless AI gateway: an MCP server that lets a client ask other AI models.
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
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
)

const version = "0.2.0"

const usage = `usage: gateway <command> [flags]

  serve    run the MCP server on stdio, or on HTTP with -http 127.0.0.1:8765
  check    send a test prompt through every configured provider
  models   print the providers, their state and their models
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

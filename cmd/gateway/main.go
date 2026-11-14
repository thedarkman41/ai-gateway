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
	config := fs.String("config", "", "providers file (default: <state>/providers.json, else built in)")
	state := fs.String("state", defaultState(), "directory for the quota ledger")
	fs.Parse(args)

	data, from := providers.Active, "built-in"
	path := *config
	if path == "" {
		path = filepath.Join(*state, "providers.json")
	}
	if b, err := os.ReadFile(path); err == nil {
		data, from = b, path
	} else if *config != "" {
		log.Fatal(err)
	}
	ps, err := provider.Parse(data)
	if err != nil {
		log.Fatalf("%s providers: %v", from, err)
	}
	return gateway.New(ps, gateway.OpenLedger(filepath.Join(*state, "ledger.json")))
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

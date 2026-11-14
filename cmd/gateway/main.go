// Command gateway is the keyless AI gateway. Phase 1 has one subcommand: probe.
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/thedarkman41/ai-gateway/internal/probe"
)

func main() {
	if len(os.Args) < 2 || os.Args[1] != "probe" {
		fmt.Fprintln(os.Stderr, "usage: gateway probe [-candidates file] [-out dir] [-timeout 45s] [-ask-cli]")
		os.Exit(2)
	}
	fs := flag.NewFlagSet("probe", flag.ExitOnError)
	file := fs.String("candidates", "providers/candidates.json", "candidates file")
	out := fs.String("out", "survey", "directory for survey.md and survey.json")
	timeout := fs.Duration("timeout", 45*time.Second, "per-provider timeout")
	askCLI := fs.Bool("ask-cli", false, "also send the prompt through installed CLIs (spends a little free quota)")
	fs.Parse(os.Args[2:])

	f, err := probe.Load(*file)
	if err != nil {
		fmt.Fprintln(os.Stderr, "gateway:", err)
		os.Exit(1)
	}
	now := time.Now()
	rs := probe.Run(context.Background(), f, probe.Options{Timeout: *timeout, AskCLI: *askCLI})
	md := probe.Markdown(rs, now)
	fmt.Print(md)

	if err := os.MkdirAll(*out, 0o755); err != nil {
		fmt.Fprintln(os.Stderr, "gateway:", err)
		os.Exit(1)
	}
	js, _ := json.MarshalIndent(rs, "", "  ")
	for name, data := range map[string][]byte{"survey.md": []byte(md), "survey.json": js} {
		if err := os.WriteFile(filepath.Join(*out, name), data, 0o644); err != nil {
			fmt.Fprintln(os.Stderr, "gateway:", err)
			os.Exit(1)
		}
	}
}

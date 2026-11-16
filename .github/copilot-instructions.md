# Copilot instructions

## Build, test, and lint

- Build the gateway: `go build -o /usr/local/bin/ai-gateway ./cmd/gateway` (the output path can be changed as needed).
- Run all tests: `go test ./...`.
- Run one test: `go test ./internal/gateway -run '^TestAskRoutingAndFallback$' -count=1`. Replace the package and test name to target another test.
- Check Go formatting with `gofmt -w <changed-go-files>` and static analysis with `go vet ./...`. There is no separate linter configured in the repository.
- `ai-gateway check` and `ai-gateway probe -ask-cli` contact external providers and may consume free usage quota; they are live checks, not unit tests.

## Architecture

- `cmd/gateway` is the executable entry point. It loads provider configuration, opens the persistent ledger, and exposes the gateway through MCP stdio or loopback-only HTTP. It also implements the `check`, `models`, `try`, and `probe` commands.
- `internal/provider` defines the common `Provider` interface and constructs adapters from JSON configuration: OpenAI-compatible HTTP (`openai`), headless vendor CLI (`cli`), and pseudo-terminal-driven interactive apps (`tty`). `providers/providers.json` is embedded into the binary by `providers/embed.go`.
- `internal/gateway` routes model requests, handles fallback and model selection, manages in-memory conversation sessions, spaces calls, and records quota/health state in `ledger.go`. `internal/gateway/tools.go` exposes these operations as MCP tools.
- `internal/mcp` implements the JSON-RPC methods the gateway needs (`initialize`, `ping`, `tools/list`, and `tools/call`) over newline-delimited stdio and HTTP.
- `internal/probe` is a distinct survey path for `providers/candidates.json`; its results are written to `survey/`.

## Repository conventions

- Keep provider-specific behavior in config where possible. Provider order in `providers/providers.json` is the fallback order for `auto`; a user's `<state>/providers.json` replaces the embedded list, while `-config` selects an explicit file.
- Provider IDs cannot contain `/` or spaces: routing uses `provider/model` to pin a model. `provider.Parse` omits `disabled` entries; `provider.ParseAll` includes them so `try` can test an entry before enabling it.
- Preserve the keyless-provider constraint in `README.md`: provider adapters do not take API keys. CLI adapters run in an empty scratch directory, and the `tty` adapter's readiness, completion, transcript, and reply parsing are controlled by each provider entry.
- Gateway fallback relies on classifying provider error text as authentication/key, quota, or busy failures; those classifications update persistent model/provider rests. Keep error reporting informative when changing adapters.
- The HTTP MCP transport is intentionally limited to loopback hosts and rejects non-local origins. Preserve that boundary when changing transport behavior.
- Tests use local fakes and `httptest` rather than live provider calls; gateway tests use temporary ledger files. Keep routine tests independent of provider credentials, installed CLIs, and external services.

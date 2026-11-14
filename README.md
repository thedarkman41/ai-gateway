# ai-gateway

A local MCP server that gives Bill the Cat (or any MCP client) access to other AI models.

**Golden rule: free and keyless.** No provider here may need an API key. Access comes through four tiers:

| Tier | How it connects |
|---|---|
| A | Open HTTP endpoints, no account |
| B | The vendor's own CLI, logged in with a free account, run headless |
| C | Mailbox: a chat app connects to the gateway as an MCP host and pulls queued questions |
| D | Browser session: the gateway drives the web UI in your own logged-in Chrome profile |

Limits on tier D: your own accounts, at a human pace. No CAPTCHA or bot-detection defeat, no account
rotation, no forged client tokens. A challenged provider marks itself down until you clear it by hand.

## Status

Phases 1 and 2 are done: the survey, and the MCP server with the first four providers. Tiers C and D are
not written yet.

| Provider | Tier | Route | On betelgeuse |
|---|---|---|---|
| gemini-cli | B | `gemini -p` | failing: past the trust gate, now "Error generating content via API"; cause not yet known |
| pollinations | A | OpenAI-compatible endpoint | answered |
| llm7 | A | OpenAI-compatible endpoint | answered (a DeepSeek model) |
| opencode | B | `opencode run -m` | enabled with its free hosted models only; not yet checked |

## Build and check

```bash
go build -o /usr/local/bin/ai-gateway ./cmd/gateway
ai-gateway check     # one test prompt through every provider
ai-gateway models    # providers, state, calls today, model ids
```

## Tools

| Tool | What it does |
|---|---|
| `list_models` | Providers, whether each can be called now, calls used today, model ids |
| `ask` | One question to one model. `model` is `auto` (default: first provider that answers), a provider id, or `provider/model`. `session` continues a conversation |
| `ask_many` | The same question to 2 to 6 models at once, for comparison |

## How it stays up

- **Fallback.** `auto` walks the providers in file order and skips any that is not installed, resting or over quota.
- **Model retry.** When no model is named, the model that answered last time is tried first, then the provider's listed models, three attempts in all.
- **Breaker.** Three failed asks in a row rest a provider for 10 minutes.
- **Spacing and quota.** `min_interval_seconds` spaces calls out; `daily_limit` caps them. Both live in `~/.ai-gateway/ledger.json`, so they survive restarts.
- **CLIs are sandboxed to a scratch directory.** The vendor CLIs are coding agents, so they start in an empty temp directory, never in a project.
- **Fix without rebuilding.** Copy `providers/providers.json` to `~/.ai-gateway/providers.json` and edit it; that copy replaces the built-in list.

## Connecting Bill the Cat

1. Build the binary to `/usr/local/bin/ai-gateway` (above).
2. Add the object in [`bill/catalog-entry.json`](bill/catalog-entry.json) to the `catalog` array of
   `~/Library/Application Support/BillCat/mcp.json`. Check its `pathDirs` against `which gemini opencode`.
3. In the app, ask Bill to use it. Bill shows the entry for your approval the first time, and asks before each
   `ask` or `ask_many` because they send data to other services.

To run it as a local HTTP server instead: `ai-gateway serve -http 127.0.0.1:8765`, with `"transport": "http"` and
`"url": "http://127.0.0.1:8765/mcp"` in the entry. It refuses any request that does not come from this machine.

## Survey

`providers/candidates.json` lists candidate providers that are not adapters yet. Results depend on the machine's
network and accounts, so run it on betelgeuse.

```bash
ai-gateway probe            # tier A prompts, CLI presence, tier D reachability
ai-gateway probe -ask-cli   # also send one prompt through each installed CLI
```

It prints the provider matrix and writes `survey/survey.md` and `survey/survey.json`.

## Phases

1. Survey (done)
2. Core (done): MCP server (stdio + localhost HTTP), adapter interface, quota ledger, tier A adapters, Bill registry entry
3. Tier B CLI adapters
4. Tier D browser engine; Muse and Perplexity first
5. Tier C mailbox and tunnel
6. Nightly canaries, self-repair loop, OpenAI-compatible endpoint

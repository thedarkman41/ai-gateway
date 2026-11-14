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

Phase 1 (survey) only. The MCP server itself is not written yet.

## Survey

`providers/candidates.json` lists candidate providers. Every entry is a guess until the probe has passed
it on the machine the gateway will run on (betelgeuse): results depend on that network and those accounts.

```bash
go build -o gateway ./cmd/gateway
./gateway probe            # tier A prompts, CLI presence, tier D reachability
./gateway probe -ask-cli   # also send one prompt through each installed CLI
```

It prints the provider matrix and writes `survey/survey.md` and `survey/survey.json`.

## Phases

1. Survey (this)
2. Core: MCP server (stdio + localhost HTTP), adapter interface, quota ledger, tier A adapters, Bill registry entry
3. Tier B CLI adapters
4. Tier D browser engine; Muse and Perplexity first
5. Tier C mailbox and tunnel
6. Nightly canaries, self-repair loop, OpenAI-compatible endpoint

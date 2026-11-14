# ai-gateway

A local MCP server that gives Bill the Cat (or any MCP client) access to other AI models.

**Golden rule: no cost per use.** No provider may charge per call or need a payment card. Most need no API key; a free API key is allowed; and a flat subscription you already pay for may be used through its own CLI. Access comes through these tiers:

| Tier | How it connects |
|---|---|
| A | Open HTTP endpoints, no account |
| B | The vendor's own CLI, logged in with a free account, run headless |
| C | Mailbox: a chat app connects to the gateway as an MCP host and pulls queued questions |
| S | A subscription you already pay for, through the vendor's CLI |
| K | A free API key on an OpenAI-compatible endpoint |
| L | A model on your own machines |
| D | Browser session: the gateway drives the web UI in your own logged-in Chrome profile |

Limits on tier D: your own accounts, at a human pace. No CAPTCHA or bot-detection defeat, no account
rotation, no forged client tokens. A challenged provider marks itself down until you clear it by hand.

## Status

Phases 1 and 2 are done: the survey, and the MCP server with the first four providers. Tiers C and D are
not written yet.

| Provider | Tier | Route | On betelgeuse |
|---|---|---|---|
| gemini-cli | S | `gemini -p`, signed in with Google | answered earlier on a free API key; now set to use the signed-in account (Google AI Pro), not yet re-run |
| pollinations | A | OpenAI-compatible endpoint | answered |
| llm7 | A | OpenAI-compatible endpoint | answered; only its keyless chat models are used |
| opencode | B | `opencode run -m` | free allowance spent each time it was checked |
| freebuff | B | terminal session (it has no prompt flag) | answered, reply read correctly |
| qwen-code | B | `qwen -p` | disabled: its sign-in offers no free keyless option |
| copilot-cli | B | `copilot -p` | answered, on the Copilot Free plan |
| codex-cli | B | `codex exec` | answered |
| cursor-agent | B | `cursor-agent -p --trust` | answered |
| amp | B | `amp -x` | disabled: execute mode uses paid credits only |
| rigel-coder | L | Loadout endpoint on rigel | answered |
| kilo, ovh | A | keyless endpoints | both answered |
| groq | K | free API key | answered |
| cloudflare | K | free API token and account id | answered |
| cohere, zai | K | free API key | answered |
| gemini-api | K | free API key | model names updated after the old one was refused as retired; not yet re-run |
| openrouter | K | free API key | the key stored on betelgeuse was another provider's; not yet called with its own |
| nvidia | K | free API key | answered |
| sambanova | K | free API key | disabled: every model answered that a payment method is required |
| mistral | K | free API key | disabled: its console requires a paid upgrade before a key can be created |

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
| `dispatch` | Fan-out: up to 12 independent tasks handed to the free providers at once; returns a job id immediately |
| `collect` | The next finished result of a job, with what is still running |

## One interface for every provider

Bill never sees how a provider is reached. Every one is a model id under the same `ask` tool, whether the
gateway calls an HTTP endpoint, runs a CLI with a prompt flag, or types into a terminal app:

| Kind | Used for | How the reply comes back |
|---|---|---|
| `openai` | OpenAI-compatible endpoints | JSON response |
| `cli` | CLIs with a prompt flag (`gemini -p`, `opencode run`) | the command's output |
| `tty` | interactive terminal apps with no prompt flag (freebuff) | the app's saved conversation file, else the rendered screen |

A `tty` provider is the most fragile: the screen tells the gateway when a reply is finished, so a UI change can break that. Tune an entry with:

```bash
ai-gateway try freebuff      # prints the final screen, then the reply extracted from it
```

### Adding another tool

Add one entry to `providers.json`; nothing in Bill changes.

- The tool takes a prompt on its command line: `kind: cli` with `bin` and `ask_args` containing `{prompt}`.
- It is interactive only: `kind: tty`, tuned with `ai-gateway try <id>` as freebuff was.

Then `ai-gateway try <id>` to see it answer, and `ai-gateway check` to run them all.

## Plan, fan out, judge

For a large coding task, Bill plans and judges and the other models do the parts:

1. **Plan.** Bill splits the work into independent, self-contained tasks.
2. **Fan out.** One `dispatch` call hands them all over. The gateway spreads tasks across whichever providers are
   free (not resting, not out of quota), runs a CLI one task at a time, and retries a failed task on up to two other
   providers. `copies: 2` on a task gets rival answers from two providers.
3. **Judge.** `collect` returns results one at a time as they finish, so each is judged (and built or tested) before
   the next arrives and none floods Bill's context.

Workers only ever see the prompt text: they do not touch the repository, and Bill applies what it accepts.
[`bill/fan-out.md`](bill/fan-out.md) is the skill that teaches Bill this loop; copy it to `billcat-skills/agent/`.

To run the same loop from a terminal, without Bill:

```bash
ai-gateway fanout "Write a Python function that parses ISO dates." "Write pytest tests for an ISO date parser."
ai-gateway fanout -all "Write a Python function is_prime(n). Reply with only the code."   # every free provider, to compare
```

Finished results are saved under `~/.ai-gateway/jobs/`, so they survive Bill stopping an idle gateway (after five
minutes without a call). Tasks still running at that moment are lost and reported as such; running the gateway as an
HTTP service (`serve -http`) avoids that.

## Replies are normalised

A reply that is nothing but one fenced code block (` ```python ... ``` `) comes back as the bare code, so every
provider answers a "reply with only the code" task the same way. A reply with prose around its code, or several
blocks, keeps its fences. A reply that stops inside a code block, or that the provider marks as cut at its output limit, counts as a failure and is retried on the next model or provider.

## Free API keys

Some providers give a free tier through an API key. The gateway skips them until their key is set.

```bash
ai-gateway key list            # which providers take a key, which are set, where to create one
ai-gateway key set groq        # paste the key; input is hidden
ai-gateway try groq
```

Keys are stored in `~/.ai-gateway/keys.json`, readable only by you, or read from the variable named in the entry's
`key_env`. They are never in `providers.json` or this repository, and a provider's error text is scrubbed of the key.

## How it stays up

- **Fallback.** `auto` walks the providers in file order and skips any that is not installed, resting or over quota.
- **Model retry.** When no model is named, the model that answered last time is tried first, then the provider's listed models, three attempts in all.
- **Learns from failures.** A model that asks for an API key is left out for a week. A busy model rests two minutes; one that times out rests thirty. A model whose free allowance is spent rests until midnight, and two such models in one ask rest the whole provider until midnight. A rate limit counts as busy, not as a spent allowance. A provider that asks for payment is rested for a week. `ai-gateway models` shows what is resting and why; `ai-gateway reset [provider]` clears a rest.
- **Breaker.** Three failed asks in a row rest a provider for 10 minutes.
- **Spacing and quota.** `min_interval_seconds` spaces calls out; `daily_limit` caps them. Both live in `~/.ai-gateway/ledger.json`, so they survive restarts.
- **Each CLI run is isolated.** The vendor CLIs are coding agents: they write files where they run and read whatever is there as "the project". So every run gets its own empty temp directory, deleted afterwards along with the per-run state the tool keeps elsewhere, and a standing instruction (the `preamble` in `providers.json`) to answer in the reply instead of creating files.
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

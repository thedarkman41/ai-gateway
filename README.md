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
| gemini-cli | S | `gemini -p -m`, signed in with Google | answers on `gemini-flash-lite-latest` only; other models hung until killed |
| antigravity-opus, antigravity-sonnet, antigravity-argon | S | `agy -p --model`, signed in with Google | answered, in votes and in a build |
| antigravity-hydrogen | S | same | not yet called |
| pollinations | A | OpenAI-compatible endpoint | answered |
| llm7 | A | OpenAI-compatible endpoint | answered; only its keyless chat models are used |
| opencode | B | `opencode run -m` | free allowance spent each time it was checked |
| freebuff | B | terminal session (it has no prompt flag) | answered, reply read correctly |
| qwen-code | B | `qwen -p` | disabled: its sign-in offers no free keyless option |
| copilot-cli | B | `copilot -p` | answered, on the Copilot Free plan |
| codex-cli | B | `codex exec` | answered |
| cursor-agent | B | `cursor-agent -p --trust` | answered |
| amp | B | `amp -x` | disabled: execute mode uses paid credits only |
| rigel-coder | L | Loadout endpoint on rigel, profile `coder-16k` (16K context) | answered |
| kilo, ovh | A | keyless endpoints | both answered |
| groq | K | free API key | answered |
| cloudflare | K | free API token and account id | answered |
| cohere, zai | K | free API key | answered |
| gemini-api | K | free API key | answered, on Gemini 4 Hydrogen |
| openrouter | K | free API key | answered, on one of its free models |
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
| `consensus` | One function, written independently by the five strongest free agents; returns the version the majority agrees on |
| `build` | A program of several functions: one task each, in a queue the free agents pull from; returns a build id |
| `collect` | Waits for a build and returns the whole program, assembled in task order, with who wrote each part |
| `rework` | Sends the functions that failed the planner's check back through a majority vote, with the error |

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

## How code gets written

There is no fixed order of providers for coding work. Bill plans and checks; the gateway decides who writes.

**One function: a vote.** `consensus` gives the function to the five strongest agents that are free. Each writes it
independently, and a seat whose agent fails moves on to the next strongest, so five answers still come back. If more
than half agree, that version is returned; if not, every distinct candidate comes back, with a test call on which
each differs, and Bill chooses.

What "agree" means depends on the language:

- **Python: same behaviour.** Agents that all write a correct function rarely write the same text, so the gateway
  runs every candidate on the same test calls and groups those that return the same results. The test calls come
  from Bill (`tests`), else from a fast model asked to write some, else plain integers. A call every candidate
  rejects is ignored. Code that does not load, or never returns, cannot win.
- **Anything else: same text**, with comments, docstrings, blank lines, spacing and Markdown wrapping set aside.

Candidates are code written by other models, so they run in a sandbox (macOS `sandbox-exec`): no network, no writes
outside a temporary directory, no reading your home directory, two seconds a call. Where no sandbox exists the
gateway does not run them and compares text instead. `AI_GATEWAY_VOTE=text` turns running candidates off.

**A program: a greedy queue.** `build` takes one task per function. Every free agent takes a task from the queue,
strongest first, and takes another the moment it finishes, so faster agents write more of the program. A task that
fails goes back to the front of the queue for a different agent; an agent that fails twice running leaves. `collect`
returns the parts as one program, in task order, between the header and footer Bill supplied. For Python and Rust, the import lines each agent wrote (`import`, `use`) are gathered once at the top; in Rust a repeated `use` would not compile.

**Check, then rework.** Bill compiles and tests the program. Each function that fails goes to `rework` with its
error: it is rewritten by the same five-agent vote, with the error and the rejected code in the prompt, and the
winner replaces it. Then Bill collects and checks again.

**Strength.** `strength` in `providers.json` (1 to 10) ranks the agents. It is a starting judgment, meant to be
edited: `ai-gateway models` shows how often each agent has sided with the majority.

Agents only ever see prompt text; they never touch the repository. [`bill/fan-out.md`](bill/fan-out.md) is the skill
that teaches Bill this; copy it to `billcat-skills/agent/`.

From a terminal, without Bill:

```bash
ai-gateway consensus "Write a Python function is_prime(n)."
ai-gateway build -header "import re" "Write a Python function slugify(s)." "Write a Python function word_count(s)." "Write a Python function is_palindrome(s)."
ai-gateway build -fail task2 "..." "..."      # also sends task2 through rework, as if it had failed to compile
```

For a full end-to-end run, [`examples/textkit_test.py`](examples/textkit_test.py) builds a 36-function Python
library through the queue over HTTP, checks all 91 expected results, and sends whatever fails back through the vote:

```bash
python3 examples/textkit_test.py        # needs the service running; writes ./textkit_build.py
```

A build is saved under `~/.ai-gateway/jobs/` as it goes, so the finished parts survive Bill stopping an idle gateway
(after five minutes without a call). Parts not yet written at that moment are reported as missing and can be
reworked; running the gateway as an HTTP service (`serve -http`) avoids that.

## Replies are normalised

A reply that is nothing but one fenced code block (` ```python ... ``` `) comes back as the bare code, so every
provider answers a "reply with only the code" task the same way. A reply with prose around its code, or several
blocks, keeps its fences. A reply that stops inside a code block, or that the provider marks as cut at its output limit, counts as a failure and is retried on the next model or provider.

## What may leave this machine

Every prompt sent to a provider outside your network passes through the privacy rules, whichever tool or endpoint
it came in by. A provider marked `"local": true` (the coder on rigel) is exempt: it gets prompts whole.

- **Secrets are redacted.** API keys, tokens, private keys, passwords in assignments and `.env` lines, and
  credentials in URLs are replaced with `[REDACTED:kind]` before the text is sent. So are the gateway's own API keys.
- **Credentials files stay home.** A prompt carrying a file that by its name holds credentials (`.env`, `*.pem`,
  `id_rsa`, anything under `~/.ssh` or `~/.aws`) goes to local models only.
- **Blocked directories stay home.** So does a prompt carrying a file from a directory you have blocked.
- **Allow list, optional.** If you list allowed directories, only files under them may leave.

"Kept home" means the prompt is still answered, by local models only: a vote is held among them, and a private task
in a build waits for a local agent. With no local model free, the request fails and says why.

```bash
ai-gateway privacy show                     # the rules in force
ai-gateway privacy block ~/work/client-x    # files under it never leave
ai-gateway privacy allow ~/Documents/GitHub # if any are listed, only files under these may leave
ai-gateway privacy remove ~/work/client-x
ai-gateway privacy check path/to/file       # what would be redacted from it, or whether it would be kept home
```

The rules live in `~/.ai-gateway/privacy.json` and a running gateway follows edits at once. If that file cannot be
read, nothing is sent outside until it is fixed. Each redaction is noted in the log (kinds and counts, never values),
and replies say when a prompt was redacted or kept home.

The path rules recognise the files Bill the Cat attaches to a `delegate_code` hand-off, which carry their full
path. They cannot see where pasted code came from. Redaction is by pattern: it catches the usual shapes of a secret,
not every possible one.

## Running as a service, and the OpenAI-compatible endpoint

```bash
ai-gateway service install     # start in the background now and at every login (macOS launchd)
ai-gateway service status
ai-gateway service uninstall
```

The service listens on `127.0.0.1:8765` only and refuses requests that do not come from this machine. That port
is fixed in the installed job; if something else holds it the gateway fails to start instead of moving.

| Path | What |
|---|---|
| `/healthz` | `200 {"ok": true}` while it can serve; `503 {"ok": false, "state": "…"}` when no agent can answer |
| `/mcp` | The MCP tools, over HTTP |
| `POST /v1/chat/completions` | OpenAI-compatible, streamed or not. Model `coder` (also `consensus`, `vote`) runs the vote; `auto`, a provider id or `provider/model` gives one answer |
| `POST /v1/builds` | Start a build: `{"language", "header", "footer", "context", "tasks": [{"id", "prompt", "tests"}]}` → `202 {"build_id"}` |
| `GET /v1/builds/{id}?wait=N` | Where it stands, waiting up to N seconds (at most 120) for it to finish |
| `POST /v1/builds/{id}/rework` | `{"failures": [{"id", "error"}]}` → `202`; then poll the GET |
| `DELETE /v1/builds/{id}` | Cancel |
| `GET /v1/models` | The model names the chat endpoint takes |

The job records the `PATH` of the shell that installed it, because that is where the agents' CLIs are found. Run
`service install` again after installing another CLI or rebuilding the gateway.

### The contract with Bill the Cat

These shapes are shared with Bill (`phase12-gateway-contract.md`, v1): its `delegate_code` posts to the chat
endpoint and its `delegate_build` drives the build endpoints. Change them only together with Bill. The tests named
`TestContract…` pin each clause.

- **Errors.** Every error body is `{"error": {"code", "message"}}`. A quota error adds `retry_after_seconds` and a
  `Retry-After` header.
- **Local-only scope.** A request with `X-AI-Gateway-Scope: local` is shown to local models only, helper calls
  included, whatever its content. Every response to it carries the header back; with no local model free it is
  refused with `403 no_local_provider` before anything is sent. A build keeps the scope it was created with. Any other
  value is `400 bad_scope`. `/healthz` echoes the header too, so Bill can check that the scope is honoured before it
  sends anything private.
- **Status line.** A vote's reply is one line starting `[ai-gateway ` and ending `]` (at most 300 characters, never
  inside a code fence), one empty line, then the body. It opens in exactly one of two ways:
  `[ai-gateway vote: majority, N of M agree (…)`, with the winning code in one fence, or
  `[ai-gateway vote: no majority among M answers`, with every candidate in its own fence under a
  `### candidate N, from …` heading. When every agent that answered gave the same thing it is the first form, however
  few answered: a local-only request with one local model reads `majority, 1 of 1 agree`.
- **Streaming.** The response headers go out before the vote starts. Until the first data event a comment line
  `: keep-alive` is sent every 10 seconds. Then one content delta, a chunk with `finish_reason: "stop"`, and
  `data: [DONE]`. A failure after the headers is `data: {"error": {"code", "message"}}` then `[DONE]`, with code
  `vote_failed`, `no_provider`, `timeout` or `local_failed`. A vote waits at most 150 seconds for its agents. When
  the client disconnects, the agents still working are stopped.
- **Builds.** `status` is `running`, `done` (every part written), `partial` or `failed` (nothing usable). Each part
  has `id`, `status` (`pending`, `running`, `done`, `failed`, `missing`), `author`, `code`, `error`, `attempts` and
  `reworked`. `program` is empty while running; finished, it is the header, the parts in task order and the footer,
  with a part that was not written as one comment line (`# [ai-gateway: t2 missing]`). `tests` are Python
  expressions that call the function; a rework vote compares the candidates on them. `notes` report redactions,
  parts kept home, reworks and cancellation.

**Pointing Bill at it.** Set `BILLCAT_CODER_URL=http://127.0.0.1:8765` and `BILLCAT_CODER_TIMEOUT=180` in Bill's
environment. Note what that changes: the task and the text of the files Bill has read then go to outside services
instead of staying on your network, subject to the privacy rules above.

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
- **Learns from failures.** A model that asks for an API key is left out for a week. A busy model rests two minutes; one that times out rests thirty. A model whose free allowance is spent rests until the time the provider says it resets (Antigravity reports a weekly allowance as "Resets in 113h…"), or until midnight when it does not say; two such models in one ask rest the whole provider for as long. A rate limit counts as busy, not as a spent allowance. A provider that asks for payment is rested for a week. `ai-gateway models` shows what is resting and why; `ai-gateway reset [provider]` clears a rest.
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

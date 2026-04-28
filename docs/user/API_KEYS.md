# API keys

Most of `rlat` runs entirely on your machine — `rlat build`, `rlat search`,
`rlat profile`, `rlat compare`, `rlat summary`, `rlat refresh`, `rlat sync`,
`rlat convert`, `rlat skill-context`, `rlat memory`, and the entire RQL
surface need no API key, no internet, and no LLM.

**Two commands need an Anthropic API key.** They are opt-in, advanced
features for users who want a programmatic surface; everyone else gets
the equivalent value through Claude-Code-driven skills with no extra cost.

| Command | What it does | Why it needs an LLM |
|---|---|---|
| `rlat optimise` | Trains a corpus-specific 512-dim projection on top of the default encoder | Generates synth-query training data via Sonnet 4.6; one-time per corpus |
| `rlat deep-search` | Runs a multi-hop research loop (plan → retrieve → refine → synthesize) and returns a synthesised answer | Calls Sonnet 4.6 for the planner / refiner / synthesizer hops |

If you don't have an API key, you have three free alternatives to
`rlat deep-search`:

- The **`deep-research` skill** — drives the same loop natively in your
  Claude Code session. Same prompts, same hop budget, same
  name-verification check, same output shape. Pays nothing on top of your
  existing Claude subscription. **This is what most users should use.**
  See `.claude/skills/deep-research/SKILL.md`.
- **Single-shot `rlat search --format context`** — one-shot retrieval,
  zero LLM cost. The bench shows 76.5% accuracy at 3.9% hallucination on
  the Microsoft Fabric corpus vs deep-research's 92.2% / 0% — single-shot
  is enough for the majority of factual lookups.
- **Manual research using `rlat search` + your existing Claude session** —
  Claude reads passages, decides what to search next, synthesises. Same
  loop the skill formalises, more freeform.

`rlat optimise` has no in-session equivalent — if you want the optimised
band, you need an API key (or you skip it; it's opt-in).

## Getting an Anthropic API key

If you decide you want an API key for one of the two opt-in commands:

1. Sign up at https://console.anthropic.com/ if you don't already have an account.
2. Go to **Settings → API Keys** and create a new key. The console shows it once — copy it immediately.
3. Add a payment method. Pay-as-you-go pricing; Sonnet 4.6 is currently $3 / 1M input tokens, $15 / 1M output tokens.

Estimated cost for the two commands:

- `rlat deep-search`: **~$0.009-0.025 per question** depending on hop count.
  A 4-hop loop on a typical question is ~$0.012.
- `rlat optimise`: **~$2-8 one-time per corpus** depending on corpus size
  (40K passages ≈ $4). Run `rlat optimise km.rlat --estimate` for a
  per-corpus preview before committing.

If those numbers are higher than you want, the in-session skill
alternatives above are free and the bench data shows they reach the
same quality ceiling.

## Setting the key in your environment

`rlat` looks up the key in this order:

1. `RLAT_LLM_API_KEY_ENV` — the *name of another env var* that holds the
   key. Useful when you want the key under a different name in your
   secrets manager.
2. `CLAUDE_API` — the canonical name in this project (matches the rest
   of the Anthropic-using tooling).
3. `ANTHROPIC_API_KEY` — Anthropic's standard env var name; supported
   for compatibility with their other tooling.

Set whichever fits your workflow.

### Linux / macOS (bash, zsh)

```bash
export CLAUDE_API="sk-ant-..."
# Persist by adding to ~/.bashrc, ~/.zshrc, or your shell rc file.
```

### Windows PowerShell (current session only)

```powershell
$env:CLAUDE_API = "sk-ant-..."
```

### Windows PowerShell (persistent across sessions)

```powershell
[System.Environment]::SetEnvironmentVariable("CLAUDE_API", "sk-ant-...", "User")
# New PowerShell sessions will see it; current session needs:
$env:CLAUDE_API = [System.Environment]::GetEnvironmentVariable("CLAUDE_API", "User")
```

### .env file (development)

If you use `direnv` or a `.env` file loaded by your shell:

```sh
CLAUDE_API=sk-ant-...
```

Don't commit `.env` files containing keys — `.gitignore` should already
have them excluded.

### From a secrets manager (production / CI)

In a CI pipeline or hosted environment, fetch the key from your secrets
manager and export it as `CLAUDE_API` (or set `RLAT_LLM_API_KEY_ENV` to
indirect through whatever name the manager uses):

```bash
# Example: AWS Secrets Manager
export CLAUDE_API="$(aws secretsmanager get-secret-value --secret-id anthropic/api-key --query SecretString --output text)"
rlat deep-search project.rlat "$question"
```

## Verification

Once set, verify the key is visible to `rlat`:

```bash
rlat deep-search myproject.rlat "test question" --max-hops 1
```

If you see `error: no Anthropic API key found ...`, the key isn't in any
of the three env vars `rlat` checks. Re-run `echo $CLAUDE_API` (Linux /
macOS) or `echo $env:CLAUDE_API` (PowerShell) to confirm — the value
should start with `sk-ant-`.

## Cost guardrails

Both commands accept a budget cap:

- `rlat optimise km.rlat --estimate` — preview cost before running.
- `rlat deep-search km.rlat "<q>" --max-hops N` — lower hop count = lower cost.

`rlat deep-search` returns the cost in its output so you can track spend
per question:

```text
[deep-search] hops=4 in=2143 out=312 cost=$0.0118 evidence=8 passages
```

Per-corpus optimise costs are also reported up-front:

```text
[optimise] estimated cost: $3.42 (4,200 synth queries × ~$0.0008/query)
```

## Why isn't deep-search free in the CLI too?

Because the CLI verb is a separate process — it can't reach into a
running Claude Code conversation to use the user's subscription. When
you run `rlat deep-search` from a shell, it spawns Python, loads the
knowledge model, and needs a way to call an LLM for the planner /
refiner / synthesizer hops. Today that's the Anthropic API.

Future work: an `rlat deep-search --llm-server <url>` flag that points
at a local LLM (Ollama, llama.cpp) or an MCP server hosted by your
existing Claude Code session, so the CLI can use whichever LLM you
already have. Not in v2.0; tracked for a future version. The
`deep-research` skill is the answer in the meantime — same loop, no
extra cost, no API key.

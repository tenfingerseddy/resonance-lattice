# Longitudinal harness benchmark

The 20-session benchmark from the agent-harness manifesto's falsifiable claim:

> The harness gets measurably better at session N+1 than at session N — same model, same task distribution.

Pass condition: both scorecard axes — **useful** (proportion of intents satisfied, weighted by intent level) and **effortless** (user touches per satisfied intent) — must move correctly when sessions 16-20 are compared against sessions 1-5.

## Scope

This benchmark runs against the **rlat repo itself**. The claim is *per-substrate*: passing here doesn't generalise to other domains. Cross-substrate validation is a separate, larger commitment that doesn't gate the v1 ship.

## Task set

[tasks.jsonl](tasks.jsonl) — 20 tasks ordered for sequential execution. Each task carries:

- `session` — the run order (1..20)
- `kind` — intent kind for recall biasing (debug / design / implement / review / explain / refactor)
- `title` — short label
- `prompt` — the literal user prompt the operator pastes into Claude Code
- `expected_intents` — levels the operator expects to declare (task, step, or both)
- `criteria` — success conditions; mechanical (`measure:spec`) where possible, `user_confirms=...` for judgement calls

The task arc goes: orientation (1-2) → debug + minor implement (3-6) → design + structural (7-9) → mid-stack work (10-13) → ladder-spanning (14-17) → audit + look-forward (18-20). Intent kinds are mixed so the cross-domain confidence accumulation mechanism (mechanism 5) has signal to fire on.

## Procedure

Per session:

1. Open Claude Code in the rlat repo. Confirm the harness state is fresh enough — `rlat memory eval` should show prior sessions if they ran today; otherwise the SessionStart trajectory primer surfaces what's active.

2. Mark the session start (one shell line):
   ```
   rlat memory session-mark   # writes .rlat-state/ledger/sessions.jsonl
   ```
   (CLI ships with the session-id marker work in [tasks.jsonl §19](tasks.jsonl).)

3. Paste the session's `prompt` verbatim. Drive the agent through the work. Use `/want`, `/accept`, `/reject`, `/decompose`, `/what-next` as the conversation calls for them.

4. Resolve the session's intents — `/accept <id>` or `/reject <id>` for each one declared. The `criteria` field tells you what's needed: mechanical criteria run during the session (test pass / exit code); `user_confirms` ones are judgement calls at the end.

5. Close Claude Code. Run:
   ```
   rlat memory consolidate    # distil arrows + confidence raise + forget
   ```

6. Log the session number against the date for later slicing.

## Reading the scorecard

After session N, the cumulative scorecard is:

```
rlat memory eval --sessions 20            # latest window summary
rlat memory eval --sessions 20 --compare  # early-window vs late-window
```

The `--compare` mode aggregates the first 25% of windows ("early") against the last 25% ("late") and prints **PASS** / **FAIL** per axis plus the benchmark verdict.

## What's measured (and what isn't)

Captured automatically:

- **Useful axis**: weighted satisfied/total intents per window. Weights: step=1, task=3, goal=10, direction=30. The benchmark expects most sessions to land 1-2 task-level resolutions.
- **Effortless axis**: user touches / satisfied intent. Touches = UserPromptSubmit fires + intent accept/reject calls. Lower is better.

Tracked but not pass/fail:

- Recall hit-rate — proportion of recalls that surfaced rows
- Memory depth — count of rows by level over time; the slope from event → pattern → learning → principle is the development signal
- Verdict-confidence distribution — rising medium+ proportion = outcomes getting more attributable

## Cold start

Sessions 1-5 are the cold-start window — memory is empty or sparse, so recall returns little, distil has nothing to promote, and the loop hasn't started compounding yet. The architecture's expectation is that the harness *degrades gracefully to vanilla Claude Code* during cold start; the late-window comparison (16-20) is where improvement should show.

If you've run prior unrelated sessions in this workspace, consider clearing `.rlat-state/` before starting (or use a fresh workspace clone). That keeps the benchmark a clean longitudinal run rather than a prefix of accumulated noise.

## Cost note

Each session uses Claude Code's normal token budget plus the distil arrows' LLM calls at consolidate time (Sonnet 4.6, ~10-30 calls per session-end). 20 sessions × consolidate at end ≈ $5-15 total in Anthropic spend, plus the conversational tokens.

This is a soft constraint, not a pass/fail axis. Watching the cost curve does flag a secondary failure mode — distil + decomposition LLM calls scaling with memory size could make later sessions structurally more expensive even as they're more useful.

## What's NOT in scope

- Cross-substrate eval (running on a non-rlat repo). Separate commitment.
- LLM-judge of task quality. The `user_confirms` criteria are user-judged; the harness doesn't re-grade.
- Per-task token spend tracking. The CLI doesn't expose this today.

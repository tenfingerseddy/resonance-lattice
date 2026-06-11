# Memory subsystem — internal reference

Per-user earned-experience memory. Flat — one polarity-tagged,
confidence-graded `Claim` per row, no tier hierarchy.

> Source-of-truth code: `src/resonance_lattice/memory/` +
> `src/resonance_lattice/cli/memory.py` + the `Claim` record in
> `src/resonance_lattice/state/claim.py`.
> User-facing surface: [`docs/site/claim-system.html`](../site/claim-system.html).

## On-disk shape

```
~/.rlat/memory/<user-id>/
├── claims.jsonl     # one Claim per line, flattened (core + ExperienceFacts)
├── band.npz         # (N, 768) float32 embeddings, L2-normalised, position-keyed
├── .claims.lock     # portalocker advisory lock for atomic writes
└── feedback.log     # append-only good/bad votes on injections
```

`Claim` records hold no embedding; the band is the parallel array.
Atomic writes use a per-writer-unique tmp + `os.replace` for both
files under the portalocker lock — a crash mid-write leaves the
prior state intact. The lock is acquired by every mutating call
*and* every read, so a concurrent capture-side writer (SessionEnd
hook) and the recall daemon's snapshot-reload (different OS
processes) serialise via the same advisory lock — neither sees a
torn `claims.jsonl` / `band.npz` pair.

On POSIX, the per-user root is chmod'd to 0o700 on init and every
write tightens `claims.jsonl` / `band.npz` to 0o600. A multi-tenant
host's other users can't read another user's captured transcripts
via the filesystem. Windows: ACLs from the home tree already
restrict cross-user access. See
[FAILURE_MODES.md §"Data isolation contract"](FAILURE_MODES.md)
for the full per-user / per-workspace partition story.

A claim's row carries:

- **core**: `claim_id`, `source` (always `experience` here), `kind`
  (always `event`), `content`, `created_at`, `corroboration`,
  `falsification`, `trust_as_of`, `state`, `parent_ids`.
- **facts** (`ExperienceFacts`): `polarity` (tuple — primary tag +
  workspace scope tags), `recurrence_count`, `criticality`,
  `created_under_intent_kind`, `transcript_hash`, `origin`,
  `last_corroborated_at`, `is_bad`.

The store + JSONL plumbing lives in
`memory/claim_store.py::ExperienceClaimStore` + the shared core-field
serialisers in `state/claim_io.py`. Corpus claims live inside a `.rlat`
archive's insight layer; their serialisers + factory + legacy migration
are in `store/corpus_claim_io.py`.

## Capture pipeline

`memory/capture.py` — runs at SessionEnd. The path:

1. **Heuristic gate** — short / pure-tool-call / pure-prompt
   transcripts skip; `evaluate_gate` returns the reason a transcript
   was deemed not worth capturing.
2. **Layer-1 redaction** — `redaction.Redactor.scrub` strips
   **credentials** (cloud keys, GitHub PATs, JWTs, PEM private keys,
   long hex) plus tool-call payloads against a denylist of credential-
   shaped paths (`.env*`, `.aws/credentials`, `*.pem`, `*.key`,
   `id_rsa`). The audit log records the rule that fired and a
   `row_id` (`claim_id` on the single-claim path, `transcript_hash`
   on the atomic-extraction and no-write paths), never the secret.
   Per-project extensions land via
   `extra_patterns` / `denylist_paths`. Non-credential PII categories
   (emails, names, phone numbers) are **not** scrubbed by Layer 1 —
   mechanical regex over-flags them; a semantic-redaction layer would
   be needed for those (none ships today).
3. **Tail truncation** — the scrubbed transcript is capped at the
   24 KB tail (the most recent context, what's worth carrying).
4. **Same-text dedup** — a near-identical `(content, workspace_tag)`
   match in the existing store bumps that claim's `recurrence_count`
   and `last_corroborated_at` instead of writing a new row.
5. **Write** — new claims land as `event`-kind, `state="active"`, at
   the rung the gate inferred (default `medium`).

## Recall pipeline

`memory/recall.py::rank` — the §0.6 four-gate pipeline. The daemon
keeps the UserPromptSubmit hook synchronous — recall is the only
thing on its critical path between user keystroke and prompt submit.
The CLI one-shot path uses the same code.

1. **Cosine + is_bad filter** — drop below `cosine_floor` (default
   0.7) and any `is_bad=True` claim.
2. **Workspace gate** — keep claims whose polarity contains the
   caller's `workspace:<cwd_hash>` tag or `cross-workspace`.
3. **Confidence gate** — require `top1 ≥ floor` AND
   `(top1 - top2) ≥ gap` (default 0.05). Either failing → empty
   result.
4. **Recurrence gate** — drop claims with
   `recurrence_count < min_recurrence` (default 3).
5. **Manifesto rerank** — `effective_score = cosine × strength ×
   valence_match × confidence_floor`, dispatched on `claim.source`.
   Fires on every cold-start store and on any `intent_kind ≠ none`.

`auto_tune_cold_start=True` (set by the UserPromptSubmit hook)
relaxes the three numerical gates when the store has fewer than
200 rows — sparse-memory workloads surface something rather than
nothing.

## Confidence raising

`memory/confidence.py` — five architecture-specified calibration
mechanisms. Each writes Beta tallies only; the 4-rung confidence
label (`low / medium / high / verified`) is the derived band over
the Beta mean.

- **M1 — Outcome corroboration**. Each satisfied / failed outcome
  weights the Beta tallies. In practice 2 wins → medium, 3 wins →
  high, 5 wins → verified.
- **M2 — Corpus verification**. `rlat memory verify <km.rlat>` —
  manual, on-demand. Checks high-criticality `low` / `verified` claims
  against a corpus knowledge model: confirmed → verified, contradicted
  → low, silent → unchanged. No scheduler today; scheduling is gated
  on a Phase E ablation showing the scan lifts a measurable metric.
- **M3 — Implicit corroboration**. A claim that surfaced in recall
  and whose session then satisfied its intent without explicit
  attribution earns a fractional bump (3 such events → 1 unit).
- **M4 — User corroboration**. `rlat memory train --corroborate
  <claim_id>` — one-step raise. `--bad-vote` is the inverse.
- **M5 — Cross-domain accumulation**. Breadth weighs alongside
  depth: each new `intent_kind` a claim wins under adds weight.

`raise_confidence_pass` runs as the first step of every
`consolidation_pass`.

## Forget

`memory/forget.py::apply_forget` — five drop conditions, with three
protections:

1. **Decay below floor** — Beta-mean-weighted age decay.
2. **Redundant after promotion** — a cluster that consolidation has
   already covered.
3. **Falsified by outcomes** — sustained net-falsification.
4. **Stale due to corpus drift** — cited passages no longer exist
   (drives the corpus-aware caller's `drifted_claim_ids`).
5. **Trivial from start** — never reached any meaningful trust.

Protections (each blocks any drop):
- Recently active — corroborated within the last 14 days.
- Severe avoid / user-declared / active-provenance — once-burned
  claims, manually declared rules, and freshly-promoted claims.

## Session-end pass

`memory/session_end_pass.py::consolidation_pass` orchestrates the
per-session-end pipeline:

1. Read the outcome ledger (`ClaimOutcomeLog`).
2. Confidence raising (M1 + M5 from the cumulative ledger).
3. Forget (the five drop conditions on the freshly-derived state).

`state_root=None` skips both stages — useful for tests that want
the pipeline shape without touching the ledger. `dry_run=True`
runs every stage but suppresses every write.

## Daemon

`memory/daemon.py` — long-lived process the UserPromptSubmit hook
talks to over a Unix-domain (POSIX) / named-pipe (Windows) socket.
Caches `(claims, band)` so per-request work is just cosine + the
four gates + rerank — no encoder, no archive read, no LLM. Boots once
per session; fast enough that the hook completes inside one human
input cycle.

Fail-open: any error returns an empty result (None) with the
diagnostic line written to `recall_diagnostic.jsonl`. The hook never
blocks the user prompt on a memory failure.

## CLI surface

```bash
rlat memory add "text" --polarity prefer        # manual claim
rlat memory list --polarity avoid               # tabular view
rlat memory recall "query"                      # one-shot recall
rlat memory recall --daemon                     # spawn / use the daemon
rlat memory train --corroborate <id>            # M4 user corroboration
rlat memory train --bad-vote <id> --why "..."   # mark is_bad
rlat memory feedback {good|bad}                 # vote on last injection
rlat memory verify <km.rlat>                    # M2 corpus verification
rlat memory consolidate                         # confidence → forget
rlat memory gc --min-recurrence 2 --dry-run     # manual escape hatch
rlat memory doctor                              # diagnostic
rlat memory dedup                               # retroactive same-text collapse
rlat memory capture                             # SessionEnd hook entry
rlat memory hook                                # UserPromptSubmit hook entry
```

Global flags (top-level argparse): `--memory-root <base>` to
override the per-user base directory; `--user <id>` to pick a
specific user subdirectory under it.

## Cross-references

- [`docs/site/claim-system.html`](../site/claim-system.html) —
  user-facing description of the claim record.
- [`MEMORY_LENS_ROADMAP.md`](MEMORY_LENS_ROADMAP.md) — the
  production-readiness effort tracking this subsystem's evolution.
- [`MEMORY_LENS_RESEARCH.md`](MEMORY_LENS_RESEARCH.md) §3 — the A2
  ablation verdict (the distillation ladder, retired in Phase B/4).
- [`FAILURE_MODES.md`](FAILURE_MODES.md) §"Hooks", §"Daemon", and
  §"State stores" — failure-mode contract for the capture pipeline,
  the recall daemon, and the per-workspace ledger.

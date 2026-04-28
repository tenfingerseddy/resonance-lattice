# Memory primer

Layered memory at `benchmarks\user_bench\primer_effectiveness\fixtures\memory` — working=0, episodic=0, semantic=10.

## Semantic (consolidated knowledge)

- (+0.829, recurrence=1) ·: The cross-encoder reranker was measured against gte-modernbert-base on BEIR-5 and regressed 4 of 5 corpora. We dropped the rerank lane entirely from the v2.0 retrieval recipe.
- (+0.821, recurrence=1) ·: We renamed cartridge to knowledge model in v2.0. The artefact is unchanged and the file extension stays .rlat; only the term changed. Pre-commit grep on rebuild/v2 rejects the term cartridge in new files.
- (+0.811, recurrence=1) ·: Bench numbers measured on the rlat deep-search CLI surface (Anthropic API) apply equivalently to the deep-research skill (in-Claude-Code session) — same loop, same prompts, same hop budget. Small variance from Sonnet-version differences only.
- (+0.805, recurrence=1) ·: Sonnet 4.6 is the canonical answer model and judge for every LLM-judged benchmark. Haiku has not been validated; using a different model invalidates the bench numbers without re-running.
- (+0.800, recurrence=1) ·: The BEIR-5 floor at 0.5144 mean nDCG@10 is locked to encoder revision e7f32e3c00f91d699e8c43b53106206bcc72bb22. Re-running with a different revision invalidates the floor; the harness compares against the locked numbers on every commit.
- (+0.794, recurrence=1) ·: BEIR fiqa optimise regressed -0.042 nDCG@10. Combined with nfcorpus -0.043 and Fabric +0.032, the corpus-profile rule is distribution alignment between synth queries and test queries, NOT natural-language vs keyword form. Two of three publicly-tested corpora regressed.
- (+0.769, recurrence=1) ·: Pre-commit cadence is simplify -> codex review -> harness runner -> board update. Every commit that ships a board-tracked deliverable updates the GitHub Project item before the commit completes.
- (+0.764, recurrence=1) ·: rlat memory has three tiers: working (1-day half-life), episodic (14-day half-life), and semantic (infinite, never decays). Recall fuses scores across all tiers with default weights working=0.5 / episodic=0.3 / semantic=0.2.
- (+0.763, recurrence=1) ·: rlat is shipping under BSL-1.1 with a 3-year change date converting to Apache-2.0. The Business Source License keeps source-available with commercial-use restrictions during the change window.

## Episodic (per-session context)

_(empty)_

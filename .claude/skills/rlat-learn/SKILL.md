---
name: rlat-learn
description: >-
  Teach an rlat knowledge model (.rlat) the world it serves — from inside the
  session, on the user's subscription, no API key. Extract durable WORLD facts,
  standing constraints, and tried-and-falsified findings from what the USER said,
  apply the four capture gates (incl. the privacy gate: facts about the person
  never land), confirm with the user, then land each via the key-free
  `rlat capture-attribute`. Trigger when the user says "remember this",
  "rlat should know this", "teach the knowledge model", "capture that as a
  constraint", "record that X didn't work", "save what we learned about the
  project/tenant/garden", or at a session's natural end when durable world facts
  surfaced. NOT for: personal preferences or facts about the user themself (the
  privacy gate drops them); corpus content that already lives in the documents
  (build/refresh handles those); session-transient state.
allowed-tools: Bash(rlat:*), Read, Glob, Grep
---

# rlat-learn — teach the knowledge model its world, in-session

A `.rlat` carries more than documents: stable **world facts**, **standing
constraints**, and **tried-and-falsified findings** — the three content classes
that measurably change answers (receipts on the project's benchmarks page).
The CLI's background hook can mine these automatically with an API key; this
skill is the **subscription path**: *you* are the extractor, the user confirms,
and the landing command is key-free. Strictly better in one way — every fact is
human-approved before it enters the file.

## The four gates (apply to every candidate — when in doubt, DROP)

These are the validated gates of the production miner (E2c: precision 0.86,
recall 1.00, zero person-fact leaks). Mirror them exactly:

1. **DURABLE + STANDING** — true before this session, persists after it. Not
   transient state, not something discovered or changed during the session
   (a version a command just printed; "right now X is failing").
2. **STATED BY THE USER** — first-person/possessive ("our tenant…", "my
   garden…", "the practice never…"). Not your own statements, not a corpus
   fact, not a hypothetical.
3. **PREMISE-BEARING** — knowing it would change the correct answer to a
   realistic future question. Drop opinions and incidental nouns.
4. **ABOUT THE WORLD, not the person** *(the privacy gate)* — true for anyone
   who uses this knowledge model (the tenant, the project, the garden, the
   practice). Drop the user's role, machine, habits, and preferences entirely.
   The band is shareable; person-facts never land.

## Classify each surviving fact

| Kind | What it is | Example |
|---|---|---|
| `attribute` | A stable world fact | "The tenant runs an F64 capacity." |
| `constraint` | A standing hard rule — served on EVERY query | "No preview features in production." |
| `falsified` | Tried and failed — include the evidence in the text | "Tried nightly full rebuilds; falsified by the May cost review." |

## The loop

1. **Collect** — re-read the conversation for USER statements that pass all
   four gates. Phrase each ≤30 words, third-person, self-contained, about the
   world ("The tenant…", "The garden…"), never about the person.
2. **Confirm** — show the user the candidate list with proposed kinds and
   subject keys. Let them edit, drop, or re-kind each one. Nothing lands
   without their say-so.
3. **Land** — find the knowledge model (`Glob "*.rlat"`; prefer the
   project-named one, ask if several). For each approved fact:

   ```
   rlat capture-attribute <km.rlat> "<fact>" --kind <attribute|constraint|falsified> [--attribute-key "<subject>"]
   ```

   Give a `--attribute-key` (a short normalized subject, e.g. `"capacity"`,
   `"water restrictions"`) whenever the fact is a *value of a subject that can
   change* — at serve time the newest value per subject wins, so a later
   correction supersedes cleanly. Leave it off for one-off facts.
4. **Receipt** — tell the user what landed and where to review or remove it:
   `rlat lens` / `rlat profile` (full trail: `rlat audit`, `rlat trace`).
   Constraints will now lead every `rlat search --format context` output as
   "Standing constraints for this environment:".

## Worked example

> User, during a session: "Remember our deploys only happen on Tuesdays, and
> we already tried blue-green — it broke the warm cache."

- Gate-check: both durable, user-stated, premise-bearing, about the world. ✓
- Classify: constraint + falsified.
- Confirm with the user, then:

```
rlat capture-attribute project.rlat "Deploys to production happen on Tuesdays only." --kind constraint --attribute-key "deploy window"
rlat capture-attribute project.rlat "Tried blue-green deploys; falsified — broke the warm cache (2026-05)." --kind falsified
```

- Receipt: "Landed 1 constraint + 1 falsified finding into project.rlat —
  review with `rlat lens`. The deploy rule now precedes every context serve."

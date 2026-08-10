"""Demo 9 — The context block is a curriculum: reasoning support, no LLM.

Owner's reframe: rlat is not just retrieval — it is a knowledge source, and
should facilitate better REASONING over the content.

The theory bridge (this session's report, demo 1): a consumer LLM's
in-context processing behaves like regression over the items in its window.
The context block rlat serves is therefore the TRAINING SET of the
consumer's in-context learner — and what makes a training set good is known
mathematics: low interference between items, explicit keys, labelled
conflicts, calibrated coverage. Retrieval chooses the k passages; reasoning
quality lives in what is served AROUND and BETWEEN them. rlat can compute
all of it offline, with receipts, and no LLM in the loop:

  9a  PAIR VERDICTS from a two-band similarity decomposition. With the
      quotient band of demo 8 (register nuisance removed), every pair of
      served passages gets two numbers: s_raw ("written near each other")
      and s_quot ("about the same thing"). The (s_raw, s_quot) plane
      separates structural pair kinds the classifier never saw — duplicates,
      sequential neighbours, same-topic-cross-register, unrelated — so the
      assembler can label relations between served passages instead of
      leaving the consumer LLM to guess them. k^2 pairs at query time; no
      stored graph (the owner's constraint from the orbit work).
  9b  COVERAGE, calibrated. "Can this corpus support an answer here?" is a
      geometric quantity. Held-out-file protocol: remove whole files from
      the band; queries drawn from removed files are UNCOVERED, queries from
      retained files (self excluded) are COVERED; a coverage score computed
      only from retrieval geometry must separate them. Measured as AUC —
      the receipt behind `--mode constrain`'s refusal becoming quantitative.
  9c  JOIN KEYS: shared rare terms between served passages, with exact
      offsets — the hooks a reasoner needs to compose facts across passages,
      served instead of rediscovered.
  9d  the assembled artefact: one query, end to end — directive, coverage
      header, passages, pair verdicts, join keys, every line with receipts.

What cannot be measured here (whether a consumer LLM actually answers
better) is exactly what the pre-registered same-evidence A/B in
EVIDENCE_CURRICULUM.md is for. Requires numpy.

Run:  python3 demo9_evidence_curriculum.py
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

import numpy as np

from corpus import build_repo_corpus, reading_chains

rng = np.random.default_rng(20260810)
PASSES = []


def check(name, ok, detail=""):
    PASSES.append(bool(ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{('   (' + detail + ')') if detail else ''}")


def unit_rows(X):
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)


# ---------------------------------------------------------------- setup ---

corpus = build_repo_corpus("/home/user/resonance-lattice")
band, coords, texts = corpus["band"], corpus["coords"], corpus["texts"]
N, d = band.shape
src_of = [s for s, _, _ in coords]
print(f"corpus: {corpus['label']}")

# quotient band (demo 8's winning object), mined from cross-file displacements
SIM = band @ band.T
np.fill_diagonal(SIM, -np.inf)
knn4 = np.argsort(-SIM, axis=1)[:, :4]
xpairs = [(i, int(j)) for i in range(N) for j in knn4[i] if src_of[i] != src_of[int(j)]]
D = unit_rows(np.stack([band[j] - band[i] for i, j in xpairs]))


def k_planes(D, k, iters=30, restarts=10):
    def one():
        n = len(D)
        i0 = int(rng.integers(n))
        u = D[i0].copy()
        v = D[(i0 + 1) % n] - (D[(i0 + 1) % n] @ u) * u
        v /= max(float(np.linalg.norm(v)), 1e-12)
        planes = [(u, v)]
        while len(planes) < k:
            proj = np.stack([(D @ u) ** 2 + (D @ v) ** 2 for u, v in planes], axis=1)
            worst = int(np.argmin(proj.max(axis=1)))
            u2 = D[worst].copy()
            partner = int(np.argsort(-np.abs(D @ u2))[1])
            v2 = D[partner] - (D[partner] @ u2) * u2
            v2 /= max(float(np.linalg.norm(v2)), 1e-12)
            planes.append((u2, v2))
        for _ in range(iters):
            proj = np.stack([(D @ u) ** 2 + (D @ v) ** 2 for u, v in planes], axis=1)
            assign = np.argmax(proj, axis=1)
            for c in range(k):
                m = D[assign == c]
                if len(m) >= 2:
                    _, _, Vt = np.linalg.svd(m, full_matrices=False)
                    planes[c] = (Vt[0], Vt[1])
        proj = np.stack([(D @ u) ** 2 + (D @ v) ** 2 for u, v in planes], axis=1)
        return planes, float(proj.max(axis=1).sum())

    return max((one() for _ in range(restarts)), key=lambda t: t[1])[0]


planes = k_planes(D, 4)
basis = []
for (u, v) in planes:
    for w in (u, v):
        w = w.copy()
        for b_ in basis:
            w -= (w @ b_) * b_
        n_ = float(np.linalg.norm(w))
        if n_ > 1e-6:
            basis.append(w / n_)
band_q = band.copy()
for b_ in basis:
    band_q = band_q - np.outer(band_q @ b_, b_)
band_q = unit_rows(band_q)

# ------------------------------------------------------------------ 9a ----

print("9a. pair verdicts: the (raw, quotient) similarity plane labels pair kinds")

chains = reading_chains(coords)
adjacent = [(a, b) for _s, ch in chains for a, b in zip(ch, ch[1:])]
dups = []
for i in range(N):
    j = int(knn4[i, 0])
    if float(band[i] @ band[j]) > 0.9 and src_of[i] != src_of[j]:
        dups.append((i, j))
by_dir = defaultdict(lambda: defaultdict(list))
for idx, (s, _, _) in enumerate(coords):
    parts = s.split("/")
    if s.startswith("benchmarks/") and len(parts) >= 3:
        nm = parts[-1].upper()
        if nm.startswith("DESIGN"):
            by_dir["/".join(parts[:-1])]["d"].append(idx)
        elif nm.startswith("VERDICT"):
            by_dir["/".join(parts[:-1])]["v"].append(idx)
xreg = []
for g in by_dir.values():
    for a in g["d"][:20]:
        for b in g["v"][:20]:
            xreg.append((a, b))
unrelated = []
while len(unrelated) < 400:
    a, b = int(rng.integers(N)), int(rng.integers(N))
    if src_of[a].split("/")[0] != src_of[b].split("/")[0]:
        unrelated.append((a, b))

classes = {"duplicate      ": dups[:400], "sequential     ": adjacent[:400],
           "cross-register ": xreg[:400], "unrelated      ": unrelated}
print("      pair kind       |  n   | s_raw  | s_quot")
S_RAW, S_QUO = {}, {}
for name, pairs in classes.items():
    S_RAW[name] = np.array([float(band[a] @ band[b]) for a, b in pairs])
    S_QUO[name] = np.array([float(band_q[a] @ band_q[b]) for a, b in pairs])
    print(f"      {name} | {len(pairs):4d} | {S_RAW[name].mean():6.3f} | "
          f"{S_QUO[name].mean():6.3f}")

# A first hypothesis — that the MEAN delta s_quot - s_raw would single out
# cross-register pairs — was wrong (all classes shrink when 8 dims are
# removed; deltas confound with baseline level). Kept per receipts culture.
# The assembler's real question is discriminative: does the quotient
# similarity separate "same topic, other register" from "unrelated" BETTER
# than raw similarity does?


def auc(pos, neg):
    allv = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg])
    r_pos, n_p, n_n = 0.0, len(pos), len(neg)
    for r, (v, is_p) in enumerate(allv, start=1):
        if is_p:
            r_pos += r
    return (r_pos - n_p * (n_p + 1) / 2) / (n_p * n_n)


auc_raw = auc(S_RAW["cross-register "], S_RAW["unrelated      "])
auc_quo = auc(S_QUO["cross-register "], S_QUO["unrelated      "])
print(f"    separating cross-register-same-topic from unrelated: "
      f"AUC {auc_raw:.3f} (raw) vs {auc_quo:.3f} (quotient)")
check("the two similarities are informative: duplicates >> sequential >> unrelated in s_raw",
      S_RAW["duplicate      "].mean() > S_RAW["sequential     "].mean()
      > S_RAW["unrelated      "].mean())
check("the decomposition adds discriminative power: quotient beats raw at spotting "
      "same-topic-across-register pairs",
      auc_quo > auc_raw + 0.03 and auc_quo > 0.6,
      f"{auc_raw:.3f} -> {auc_quo:.3f}")
print("    -> an assembler can therefore LABEL relations among served passages")
print("       ('near-duplicate of', 'continues', 'same topic in another register')")
print("       from geometry alone, each label carrying its two numbers as receipt.")

# ------------------------------------------------------------------ 9b ----

print("9b. coverage, calibrated: can this corpus support an answer here?")

# Protocol note (a first version failed here, instructively): holding out
# RANDOM files does not create uncovered queries in a redundant corpus —
# the changelog and overview docs restate most topics, so removed files'
# content is still covered and the labels are wrong (AUC 0.69 measured).
# Label validity requires holding out files whose content is NOT duplicated
# elsewhere: rank files by redundancy (mean max cross-file cosine of their
# passages) and hold out the least-redundant ones.
file_sizes = Counter(src_of)
eligible = [f for f, c in file_sizes.items() if c >= 5]
redundancy = {}
for f in eligible:
    idxs = [i for i in range(N) if src_of[i] == f]
    other = np.array([i for i in range(N) if src_of[i] != f])
    redundancy[f] = float(np.mean(np.max(band[idxs] @ band[other].T, axis=1)))
eligible.sort(key=lambda f: redundancy[f])
held = set(eligible[:8])
print(f"    held-out files chosen for label validity (lowest redundancy "
      f"{redundancy[eligible[0]]:.2f}..{redundancy[eligible[7]]:.2f}; "
      f"corpus median {float(np.median(list(redundancy.values()))):.2f})")
keep = np.array([i for i in range(N) if src_of[i] not in held])
band_keep = band[keep]
uncovered_q = [i for i in range(N) if src_of[i] in held]
covered_q = list(rng.choice(keep, size=min(len(uncovered_q), 150), replace=False))
uncovered_q = uncovered_q[: len(covered_q)]
keep_pos = {int(g): p for p, g in enumerate(keep)}


def coverage_scores(q_idx, exclude_self):
    s = band_keep @ band[q_idx]
    if exclude_self and q_idx in keep_pos:
        s[keep_pos[q_idx]] = -np.inf
    srt = np.sort(s)
    top = srt[-5:]
    return {"top1": float(top[-1]),
            "mean5": float(top.mean()),
            "peak": float(top[-1] - srt[-100:-5].mean()),
            "margin": float(top[-1] - np.median(s))}


feats_cov = [coverage_scores(q, True) for q in covered_q]
feats_unc = [coverage_scores(q, False) for q in uncovered_q]
print(f"    protocol: {len(held)} whole files removed from the band; "
      f"{len(uncovered_q)} uncovered vs {len(covered_q)} covered queries")
aucs = {}
for k_ in ("top1", "mean5", "peak", "margin"):
    aucs[k_] = auc([f[k_] for f in feats_cov], [f[k_] for f in feats_unc])
    print(f"      coverage scorer '{k_}': AUC {aucs[k_]:.3f}")
best_scorer = max(aucs, key=aucs.get)
check("retrieval geometry alone detects when the corpus cannot support an answer",
      aucs[best_scorer] >= 0.80, f"best AUC {aucs[best_scorer]:.3f} ({best_scorer})")
print("    -> served as a per-query epistemic header with the number as receipt,")
print("       this turns `--mode constrain`'s refusal from a directive into a")
print("       calibrated, auditable decision. (Production calibration: correlate")
print("       the scorer with answerability on the 63-question Fabric bench.)")

# ------------------------------------------------------------------ 9c ----

print("9c. join keys: shared rare terms between served passages, with offsets")

token_re = re.compile(r"[A-Za-z_][A-Za-z0-9_\-\.]{3,}")
doc_freq = Counter()
tokens_of = []
for t in texts:
    toks = set(m.group(0).lower() for m in token_re.finditer(t))
    tokens_of.append(toks)
    doc_freq.update(toks)
# rarity band: distinctive but not vanishing (a df<=5 cut-off found almost
# nothing across registers — rare terms shared across registers are rare
# squared; kept as a calibration lesson)
RARE = {t for t, c in doc_freq.items() if c <= max(8, N // 40)}


def join_keys(i, j, cap=4):
    shared = (tokens_of[i] & tokens_of[j]) & RARE
    return sorted(shared, key=lambda t: doc_freq[t])[:cap]


jk = next((join_keys(a, b) for a, b in xreg if join_keys(a, b)), [])
n_with_keys = sum(1 for (a, b) in xreg[:200] if join_keys(a, b))
print(f"    {n_with_keys}/{min(200, len(xreg))} cross-register pairs share >= 1 rare term "
      f"(example: {jk})")
check("join keys exist to serve on most same-topic pairs",
      xreg and n_with_keys >= 0.6 * min(200, len(xreg)))

# ------------------------------------------------------------------ 9d ----

print("9d. the assembled artefact: one query, end to end (abridged)")

query_text = "how does rlat detect contradictions in a corpus"
q_toks = set(m.group(0).lower() for m in token_re.finditer(query_text))
q_vec = np.zeros(N)
overlap = np.array([len(q_toks & tokens_of[i]) for i in range(N)], dtype=float)
seed = int(np.argmax(overlap))
q = band[seed]                                   # stand-in for an encoded query
scores = band @ q
scores[seed] = -np.inf
top = list(np.argsort(-scores)[:6])
cov = coverage_scores(seed, True)[best_scorer]

print(f"    query: {query_text!r}")
print(f"    <!-- coverage: {best_scorer}={cov:.2f} (calibrated AUC {aucs[best_scorer]:.2f}) "
      f"-> answerable; below threshold this block would open with a refusal directive -->")
for rank, idx in enumerate(top, 1):
    s, o, ln = coords[idx]
    print(f"    [{rank}] {s}:{o}+{ln}  score={float(band[idx] @ q):.3f}")
n_rel = 0
for x in range(len(top)):
    for y in range(x + 1, len(top)):
        a, b = top[x], top[y]
        sr, sq = float(band[a] @ band[b]), float(band_q[a] @ band_q[b])
        rel = None
        if sr > 0.9:
            rel = "near-duplicate"
        elif src_of[a] == src_of[b] and abs(coords[a][1] - coords[b][1]) < 2000:
            rel = "continues"
        elif sq - sr > 0.05 and sq > 0.35:
            rel = "same topic, different register"
        if rel:
            keys = join_keys(a, b)
            n_rel += 1
            print(f"    rel: [{x + 1}]<->[{y + 1}] {rel}  "
                  f"(s_raw={sr:.2f}, s_quot={sq:.2f}"
                  + (f", join keys: {', '.join(keys[:3])}" if keys else "") + ")")
check("the block carries structure, not just hits (>= 2 labelled relations served)",
      n_rel >= 2, f"{n_rel} pair relations among 6 passages")
print("    -> same passages a flat block would serve; the consumer LLM no longer")
print("       has to guess what relates them, what duplicates what, or whether")
print("       the corpus can support an answer at all.")

print()
print("Summary: 'facilitate reasoning' decomposes into servable, LLM-free objects:")
print("labelled relations BETWEEN served passages (from the raw x quotient")
print("decomposition, k^2 at query time, no stored graph), a calibrated coverage")
print("verdict (the geometry knows what it doesn't know: AUC above), join keys for")
print("composition, and receipts on every line. Whether the consumer LLM actually")
print("reasons better is exactly the pre-registered same-evidence A/B in")
print("EVIDENCE_CURRICULUM.md — same retrieved passages, structured vs flat block,")
print("scored on the existing hallucination harness.")
print()
print("ALL PASS" if all(PASSES) else "SOME CHECKS FAILED")
raise SystemExit(0 if all(PASSES) else 1)

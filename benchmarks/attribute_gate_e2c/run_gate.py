"""E2c — run the REAL 4-gate extractor over the pre-registered trap sessions.

Production-faithful: `extract_attributes` via `default_client` (the same
Sonnet model the SessionEnd hook uses). Deterministic term-match grading —
no judge, no noise. See DESIGN.md for the pre-registered bars.

Usage:
    python benchmarks/attribute_gate_e2c/run_gate.py
Writes run_results/run1_2026-06-10.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parents[1] / "src"))

from resonance_lattice._anthropic import api_key_or_error, default_client  # noqa: E402
from resonance_lattice.memory.attribute_extract import extract_attributes  # noqa: E402

SESSIONS = HERE / "sessions.json"
OUT = HERE / "run_results" / "run1_2026-06-10.json"

BARS = {
    "precision_min": 0.83,
    "recall_min": 0.85,
    "person_leaks_max": 0,
    "domain_precision_min": 0.75,
}


def main() -> int:
    client = default_client(api_key_or_error())
    sessions = json.loads(SESSIONS.read_text(encoding="utf-8"))["sessions"]

    per_session = []
    for s in sessions:
        emitted = extract_attributes(s["text"], client=client)
        if emitted is None:
            print(f"[{s['id']}] extractor returned None (failure)",
                  file=sys.stderr)
            emitted = []
        matched_gt, fp, compound_leaks = [], [], []
        gt_hit = [False] * len(s["ground_truth"])
        for e in emitted:
            low = e.lower()
            # Person-trap scan runs over EVERY emission, including
            # GT-matched ones — a compound emission carrying a world fact
            # AND a person fact must still count as a privacy leak
            # (review finding: GT-first precedence could hide it).
            person_hit = any(
                any(term in low for term in t["terms"])
                for t in s["traps"] if t["class"] == "person")
            hit = None
            for i, gt in enumerate(s["ground_truth"]):
                if all(t in low for t in gt["terms"]):
                    hit = gt["fact"]
                    gt_hit[i] = True
                    break
            if hit and person_hit:
                compound_leaks.append({"emitted": e, "matches": hit})
            if hit:
                matched_gt.append({"emitted": e, "matches": hit})
                continue
            trap_cls = next(
                (t["class"] for t in s["traps"]
                 if any(term in low for term in t["terms"])), "other")
            fp.append({"emitted": e, "trap_class": trap_cls})
        per_session.append({
            "id": s["id"], "domain": s["domain"], "emitted": emitted,
            "matched": matched_gt, "false_positives": fp,
            "compound_person_leaks": compound_leaks,
            "gt_total": len(s["ground_truth"]), "gt_recalled": sum(gt_hit),
            "missed": [gt["fact"] for i, gt in enumerate(s["ground_truth"])
                       if not gt_hit[i]],
        })

    n_emit = sum(len(p["emitted"]) for p in per_session)
    n_match = sum(len(p["matched"]) for p in per_session)
    n_gt = sum(p["gt_total"] for p in per_session)
    n_rec = sum(p["gt_recalled"] for p in per_session)
    person_leaks = (
        [f for p in per_session for f in p["false_positives"]
         if f["trap_class"] == "person"]
        + [f for p in per_session for f in p["compound_person_leaks"]])
    fp_by_class: dict[str, int] = {}
    for p in per_session:
        for f in p["false_positives"]:
            fp_by_class[f["trap_class"]] = fp_by_class.get(f["trap_class"], 0) + 1

    domains = {}
    for dom in ("software", "garden", "legal"):
        ps = [p for p in per_session if p["domain"] == dom]
        e = sum(len(p["emitted"]) for p in ps)
        m = sum(len(p["matched"]) for p in ps)
        g = sum(p["gt_total"] for p in ps)
        r = sum(p["gt_recalled"] for p in ps)
        domains[dom] = {
            "precision": m / e if e else None,
            "recall": r / g if g else None,
            "emitted": e,
        }

    precision = n_match / n_emit if n_emit else None
    recall = n_rec / n_gt if n_gt else None
    summary = {
        "model": "production default_client (Sonnet)",
        "precision": precision,
        "recall": recall,
        "emitted": n_emit, "matched": n_match,
        "gt_total": n_gt, "gt_recalled": n_rec,
        "person_leaks": len(person_leaks),
        "person_leak_detail": person_leaks,
        "false_positives_by_class": fp_by_class,
        "per_domain": domains,
        "bars": {
            "precision_ge_083": precision is not None
            and precision >= BARS["precision_min"],
            "recall_ge_085": recall is not None
            and recall >= BARS["recall_min"],
            "person_leaks_eq_0": len(person_leaks) == 0,
            "all_domains_precision_ge_075": all(
                d["precision"] is not None
                and d["precision"] >= BARS["domain_precision_min"]
                for d in domains.values()),
        },
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(
        json.dumps({"summary": summary, "per_session": per_session}, indent=1),
        encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

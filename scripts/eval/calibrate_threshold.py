#!/usr/bin/env python3
"""
Calibrate the legacy-pass similarity threshold from a golden-set eval run.

Why
---
The legacy pass criterion requires the top-3 average cosine similarity to
clear a threshold.  The right value depends on the embedding model:
bge-small-en-v1.5 runs "cool" — correct retrievals routinely score 0.42-0.63
— so the historical fixed 0.50 cutoff failed queries whose retrieval was
actually correct (4 of the 7 legacy-pass misses in the 2026-06-25 baseline
retrieved the right source and lost only on this threshold).

Method
------
Derive the threshold from a saved eval-results artifact:

  positives = in-corpus queries whose retrieval was correct
              (hit_rate_at_k == 1.0) — these SHOULD clear the threshold
  negatives = out-of-corpus probes — these should NOT clear it

Sweep candidate thresholds (midpoints between adjacent observed similarity
values) and report the one maximizing Youden's J = TPR - FPR.

The chosen value is stored as a constant in ``scripts/eval/metrics.py``
(``PASS_SIM_THRESHOLDS``); this script is the documented, reproducible
derivation — rerun it after an embedding-model or index change.

Limitation
----------
If the populations overlap, no threshold separates perfectly; the report
shows the misclassified cases so the operating point is chosen with eyes
open.  On the 2026-07-02 run, neg-003 (the nonexistent-spec trap TS 99.999,
avg sim 0.560) scores above every correct-retrieval threshold — a similarity
cutoff cannot catch a probe that is lexically native to the corpus.

Usage:
    python scripts/eval/calibrate_threshold.py data/eval_results.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Allow running as a script from the repo root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def extract_populations(results: Dict) -> Tuple[List[float], List[float]]:
    """Split a results artifact into (positives, negatives) similarity lists.

    positives: avg_similarity of in-corpus cases with hit_rate_at_k == 1.0
    negatives: avg_similarity of out-of-corpus probes
    """
    positives: List[float] = []
    negatives: List[float] = []
    for case in results.get("cases", []):
        sim = case.get("avg_similarity")
        if sim is None:
            continue
        if case.get("is_out_of_corpus"):
            negatives.append(sim)
        elif case.get("hit_rate_at_k") == 1.0:
            positives.append(sim)
    return positives, negatives


def youden_optimal(positives: List[float], negatives: List[float]) -> Dict:
    """Return the Youden-J-optimal threshold for separating the populations.

    Candidates are midpoints between adjacent observed values (plus outer
    bounds), so the optimum lands in the middle of the widest valid gap
    rather than exactly on an observed data point.

    Returns a dict: {"threshold", "tpr", "fpr", "j", "candidates"} where
    candidates is the full sweep table for the report.
    """
    if not positives or not negatives:
        raise ValueError("Need at least one positive and one negative case")

    values = sorted(set(positives) | set(negatives))
    candidates = [values[0] - 0.01]
    candidates += [(a + b) / 2 for a, b in zip(values, values[1:])]
    candidates.append(values[-1] + 0.01)

    table = []
    for t in candidates:
        tpr = sum(1 for p in positives if p >= t) / len(positives)
        fpr = sum(1 for n in negatives if n >= t) / len(negatives)
        table.append({"threshold": round(t, 4), "tpr": round(tpr, 4),
                      "fpr": round(fpr, 4), "j": round(tpr - fpr, 4)})

    best = max(table, key=lambda row: row["j"])
    return {**best, "candidates": table}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Derive the legacy-pass similarity threshold from an eval artifact"
    )
    parser.add_argument("results", help="Path to an eval results JSON (e.g. data/eval_results.json)")
    args = parser.parse_args()

    with open(args.results) as f:
        results = json.load(f)

    positives, negatives = extract_populations(results)
    print(f"Positives (in-corpus, correct retrieval): N={len(positives)}, "
          f"min={min(positives):.3f}, max={max(positives):.3f}")
    print(f"Negatives (out-of-corpus probes):         N={len(negatives)}, "
          f"min={min(negatives):.3f}, max={max(negatives):.3f}")

    result = youden_optimal(positives, negatives)

    print(f"\n{'threshold':>10} {'TPR':>7} {'FPR':>7} {'J':>7}")
    for row in result["candidates"]:
        marker = "  <-- optimal" if row["threshold"] == result["threshold"] else ""
        print(f"{row['threshold']:>10.4f} {row['tpr']:>7.3f} {row['fpr']:>7.3f} "
              f"{row['j']:>7.3f}{marker}")

    print(f"\nRecommended threshold: {result['threshold']:.3f} "
          f"(TPR {result['tpr']:.3f}, FPR {result['fpr']:.3f}, J {result['j']:.3f})")

    missed_pos = [p for p in positives if p < result["threshold"]]
    leaked_neg = [n for n in negatives if n >= result["threshold"]]
    if missed_pos:
        print(f"Correct retrievals still below threshold: {sorted(missed_pos)}")
    if leaked_neg:
        print(f"OOC probes above threshold (inseparable): {sorted(leaked_neg)}")
    print("\nRecord the chosen value in scripts/eval/metrics.py PASS_SIM_THRESHOLDS.")


if __name__ == "__main__":
    main()

"""
Regression comparison for the 3GPP RAG eval harness.

Usage
-----
    from scripts.eval.regression import compare_to_baseline, save_baseline

    # After a run, save it as the new baseline:
    save_baseline(current_summary, baseline_path="data/eval/baseline.json")

    # On the next run, compare and exit non-zero if regressions detected:
    regressions = compare_to_baseline(current_summary, baseline_path)
    if regressions:
        for r in regressions:
            print(r)
        sys.exit(1)

Regression definition
---------------------
A metric REGRESSES when:
    current_value < baseline_value - tolerance

Tolerances are metric-specific (see METRIC_TOLERANCES below).
The intent is to allow normal run-to-run noise without false alarms while
catching genuine quality drops, e.g. a new embedding model or chunking
change that hurts recall.

Only metrics listed in TRACKED_METRICS are compared; unknown keys are
silently ignored so that adding new metrics doesn't break old baselines.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Metrics to track (key path = "section.metric")
TRACKED_METRICS = [
    # Standard IR (primary signals)
    "retrieval.avg_hit_rate_at_k",
    "retrieval.avg_recall_at_k",
    "retrieval.avg_mrr",
    "retrieval.avg_ndcg_at_k",
    # Heuristic (secondary)
    "retrieval.avg_context_precision",
    "retrieval.avg_context_recall",
    # Latency (upper-bound check: regression = got SLOWER)
    "latency.retrieve.p50",
    "latency.retrieve.p95",
]

# Per-metric absolute tolerance before a regression is flagged.
# For latency, regression = current > baseline + tolerance (inverted logic).
METRIC_TOLERANCES: Dict[str, float] = {
    "retrieval.avg_hit_rate_at_k":    0.05,
    "retrieval.avg_recall_at_k":      0.05,
    "retrieval.avg_mrr":              0.05,
    "retrieval.avg_ndcg_at_k":        0.05,
    "retrieval.avg_context_precision": 0.05,
    "retrieval.avg_context_recall":    0.05,
    "latency.retrieve.p50":            0.10,   # seconds
    "latency.retrieve.p95":            0.20,   # seconds
}

LATENCY_METRICS = {"latency.retrieve.p50", "latency.retrieve.p95"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_nested(d: Dict, key_path: str) -> Optional[float]:
    """Retrieve a value from a nested dict using dot-separated key path."""
    parts = key_path.split(".")
    node = d
    for part in parts:
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return float(node) if node is not None else None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def save_baseline(summary: Dict, baseline_path: str = "data/eval/baseline.json") -> None:
    """Persist current eval summary as the regression baseline.

    Only the keys in TRACKED_METRICS are saved; the full summary may contain
    per-case details that would make the diff noisy.

    Args:
        summary: The structured summary dict produced by the eval harness.
        baseline_path: File path to write.
    """
    path = Path(baseline_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    snapshot: Dict = {}
    for key in TRACKED_METRICS:
        val = _get_nested(summary, key)
        if val is not None:
            # Store preserving nested structure
            parts = key.split(".")
            node = snapshot
            for part in parts[:-1]:
                node = node.setdefault(part, {})
            node[parts[-1]] = val

    with open(path, "w") as f:
        json.dump({"baseline": snapshot, "tracked_metrics": TRACKED_METRICS}, f, indent=2)
    print(f"Baseline saved to {path}")


def load_baseline(baseline_path: str = "data/eval/baseline.json") -> Optional[Dict]:
    """Load a saved baseline snapshot. Returns None if the file doesn't exist."""
    path = Path(baseline_path)
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    return data.get("baseline", {})


def compare_to_baseline(
    current_summary: Dict,
    baseline_path: str = "data/eval/baseline.json",
    tolerances: Optional[Dict[str, float]] = None,
) -> List[str]:
    """Compare current eval results to the stored baseline.

    Returns a list of human-readable regression messages.  An empty list
    means no regressions detected.  Non-empty list should trigger exit(1)
    in CI.

    Args:
        current_summary: Structured summary dict from the current run.
        baseline_path:   Path to the baseline JSON written by save_baseline().
        tolerances:      Override default tolerances per metric key.

    Returns:
        List of regression description strings (empty = clean).
    """
    baseline = load_baseline(baseline_path)
    if baseline is None:
        print(f"[regression] No baseline found at {baseline_path}. "
              f"Run with --save-baseline to create one.")
        return []

    tols = {**METRIC_TOLERANCES, **(tolerances or {})}
    regressions = []

    for key in TRACKED_METRICS:
        current_val = _get_nested(current_summary, key)
        baseline_val = _get_nested(baseline, key)

        if current_val is None or baseline_val is None:
            continue

        tol = tols.get(key, 0.05)

        if key in LATENCY_METRICS:
            # Regression = got slower
            if current_val > baseline_val + tol:
                regressions.append(
                    f"REGRESSION [{key}]: {baseline_val:.4f}s → {current_val:.4f}s "
                    f"(+{current_val - baseline_val:.4f}s exceeds tolerance {tol}s)"
                )
        else:
            # Regression = quality dropped
            if current_val < baseline_val - tol:
                regressions.append(
                    f"REGRESSION [{key}]: {baseline_val:.4f} → {current_val:.4f} "
                    f"(drop {baseline_val - current_val:.4f} exceeds tolerance {tol})"
                )

    return regressions


def print_comparison_table(
    current_summary: Dict,
    baseline_path: str = "data/eval/baseline.json",
) -> None:
    """Print a side-by-side comparison table of current vs baseline metrics."""
    baseline = load_baseline(baseline_path)
    if baseline is None:
        print("No baseline to compare against.")
        return

    print(f"\n{'Metric':<40} {'Baseline':>10} {'Current':>10} {'Delta':>10}")
    print("-" * 74)
    for key in TRACKED_METRICS:
        cur = _get_nested(current_summary, key)
        bas = _get_nested(baseline, key)
        if cur is None and bas is None:
            continue
        cur_str = f"{cur:.4f}" if cur is not None else "n/a"
        bas_str = f"{bas:.4f}" if bas is not None else "n/a"
        if cur is not None and bas is not None:
            delta = cur - bas
            delta_str = f"{delta:+.4f}"
            if key in LATENCY_METRICS:
                flag = " !" if delta > METRIC_TOLERANCES.get(key, 0.05) else ""
            else:
                flag = " !" if delta < -METRIC_TOLERANCES.get(key, 0.05) else ""
        else:
            delta_str = "n/a"
            flag = ""
        print(f"{key:<40} {bas_str:>10} {cur_str:>10} {delta_str:>10}{flag}")
    print()

"""Offline analysis of reranker score distributions from evaluation output.

This script reads a saved ``scripts/run_evaluation.py`` JSON artifact. It does
not call OpenAI, Pinecone, or the reranker model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
from typing import Any, Dict, Iterable, List, Sequence


def load_cases(path: str | Path) -> List[Dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return list(data.get("cases", []))


def top_scores(cases: Sequence[Dict[str, Any]], expected_behavior: str) -> List[float]:
    scores: List[float] = []
    for case in cases:
        if case.get("expected_behavior") != expected_behavior:
            continue
        case_scores = [
            float(score) for score in case.get("reranker_scores", []) or [] if score is not None
        ]
        if case_scores:
            scores.append(case_scores[0])
    return scores


def describe(values: Iterable[float]) -> Dict[str, Any]:
    data = sorted(float(value) for value in values)
    if not data:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "q1": None,
            "q3": None,
        }

    return {
        "count": len(data),
        "min": round(data[0], 4),
        "max": round(data[-1], 4),
        "mean": round(sum(data) / len(data), 4),
        "median": round(statistics.median(data), 4),
        "q1": round(_percentile(data, 0.25), 4),
        "q3": round(_percentile(data, 0.75), 4),
    }


def overlap(answerable: Sequence[float], refusals: Sequence[float]) -> Dict[str, Any]:
    if not answerable or not refusals:
        return {
            "answerable_at_or_below_refusal_max": 0,
            "refusals_at_or_above_answerable_min": 0,
            "answerable_min": min(answerable) if answerable else None,
            "refusal_max": max(refusals) if refusals else None,
        }

    answerable_min = min(answerable)
    refusal_max = max(refusals)
    return {
        "answerable_at_or_below_refusal_max": sum(score <= refusal_max for score in answerable),
        "refusals_at_or_above_answerable_min": sum(score >= answerable_min for score in refusals),
        "answerable_min": round(answerable_min, 4),
        "refusal_max": round(refusal_max, 4),
    }


def analyze(path: str | Path) -> Dict[str, Any]:
    cases = load_cases(path)
    answerable = top_scores(cases, "answer")
    refusals = top_scores(cases, "refuse")
    return {
        "artifact": str(path),
        "answerable_top_reranker_score": describe(answerable),
        "expected_refusal_top_reranker_score": describe(refusals),
        "overlap": overlap(answerable, refusals),
    }


def render_report(analysis: Dict[str, Any]) -> str:
    answerable = analysis["answerable_top_reranker_score"]
    refusals = analysis["expected_refusal_top_reranker_score"]
    overlap_data = analysis["overlap"]
    return "\n".join(
        [
            "# Reranker Score Analysis",
            "",
            f"Artifact: {analysis['artifact']}",
            "",
            "## Answerable Top Reranker Score",
            _format_stats(answerable),
            "",
            "## Expected-Refusal Top Reranker Score",
            _format_stats(refusals),
            "",
            "## Overlap",
            f"- Answerable at or below refusal max: "
            f"{overlap_data['answerable_at_or_below_refusal_max']}",
            f"- Refusals at or above answerable min: "
            f"{overlap_data['refusals_at_or_above_answerable_min']}",
            f"- Answerable min: {_fmt(overlap_data['answerable_min'])}",
            f"- Refusal max: {_fmt(overlap_data['refusal_max'])}",
            "",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze reranker score distributions from an evaluation JSON artifact."
    )
    parser.add_argument("artifact", help="Path to run_*.json from scripts/run_evaluation.py")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON instead of Markdown.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    analysis = analyze(args.artifact)
    if args.json:
        print(json.dumps(analysis, indent=2))
    else:
        print(render_report(analysis))
    return 0


def _percentile(values: Sequence[float], fraction: float) -> float:
    if len(values) == 1:
        return values[0]
    index = (len(values) - 1) * fraction
    lower = int(index)
    upper = min(lower + 1, len(values) - 1)
    weight = index - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def _format_stats(stats: Dict[str, Any]) -> str:
    return "\n".join(
        [
            f"- Count: {stats['count']}",
            f"- Min: {_fmt(stats['min'])}",
            f"- Max: {_fmt(stats['max'])}",
            f"- Mean: {_fmt(stats['mean'])}",
            f"- Median: {_fmt(stats['median'])}",
            f"- Q1: {_fmt(stats['q1'])}",
            f"- Q3: {_fmt(stats['q3'])}",
        ]
    )


def _fmt(value: Any) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.4f}"


if __name__ == "__main__":
    raise SystemExit(main())

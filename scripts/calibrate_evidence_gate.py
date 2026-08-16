#!/usr/bin/env python3
"""Offline EvidenceGate threshold calibration from recorded eval scores.

This script never calls OpenAI, Pinecone, or the live RAG stack. It replays the
current EvidenceGate decision logic over retrieval similarity scores already
stored in an evaluation JSON artifact.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
import statistics
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


BASELINE_JSON = Path("eval_results/run_20260815_165508.json")
SPLIT_JSON = Path("data/eval/eval_split.json")
DEFAULT_OUTPUT_JSON = Path("eval_results/evidence_gate_calibration.json")
DEFAULT_OUTPUT_MD = Path("eval_results/evidence_gate_calibration_offline.md")

CURRENT_TOP = 0.70
CURRENT_DOC = 0.65
CURRENT_MEAN = 0.60
FIXED_MIN_DOCS = 1
FIXED_MEAN_TOP_N = 3

ANSWER = "answer"
REFUSE = "refuse"


@dataclass(frozen=True)
class GateConfig:
    """EvidenceGate thresholds replayed offline."""

    min_top_score: float
    min_doc_score: float
    min_mean_score: float
    min_docs: int = FIXED_MIN_DOCS
    mean_top_n: int = FIXED_MEAN_TOP_N


@dataclass(frozen=True)
class StrategyConfig:
    """Simple deterministic gate strategy for offline score-source sweeps."""

    strategy: str
    score_source: str
    min_top_score: float
    min_doc_score: float = 0.0
    min_docs: int = 0
    min_mean_score: float = 0.0
    mean_top_n: int = FIXED_MEAN_TOP_N


@dataclass(frozen=True)
class GateDecision:
    """Offline EvidenceGate decision."""

    sufficient: bool
    reason: str
    top_score: float
    mean_score: float
    qualifying_docs: int
    total_docs: int


def load_eval_cases(path: Path | str = BASELINE_JSON) -> List[Dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return list(data["cases"])


def load_split(path: Path | str = SPLIT_JSON) -> Dict[str, List[str]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {
        "calibration": list(data["calibration"]),
        "validation": list(data["validation"]),
    }


def cases_for_ids(cases: Sequence[Dict[str, Any]], ids: Sequence[str]) -> List[Dict[str, Any]]:
    by_id = {case["id"]: case for case in cases}
    missing = [case_id for case_id in ids if case_id not in by_id]
    if missing:
        raise ValueError(f"Split references missing case IDs: {missing}")
    return [by_id[case_id] for case_id in ids]


def validate_split(cases: Sequence[Dict[str, Any]], split: Dict[str, List[str]]) -> None:
    all_ids = {case["id"] for case in cases}
    calibration = set(split["calibration"])
    validation = set(split["validation"])
    if calibration & validation:
        raise ValueError("Calibration and validation split IDs overlap")
    if calibration | validation != all_ids:
        missing = sorted(all_ids - (calibration | validation))
        extra = sorted((calibration | validation) - all_ids)
        raise ValueError(f"Split does not cover dataset exactly; missing={missing} extra={extra}")


def replay_gate(scores: Sequence[float], config: GateConfig) -> GateDecision:
    ordered = sorted([float(score) for score in scores], reverse=True)
    total_docs = len(ordered)
    if not ordered:
        return GateDecision(False, "no_documents", 0.0, 0.0, 0, 0)

    top_score = ordered[0]
    top_scores = ordered[: config.mean_top_n]
    mean_score = sum(top_scores) / len(top_scores) if top_scores else 0.0
    qualifying_docs = sum(score >= config.min_doc_score for score in ordered)

    if top_score < config.min_top_score:
        reason = "top_score_below_threshold"
        sufficient = False
    elif qualifying_docs < config.min_docs:
        reason = "not_enough_qualifying_documents"
        sufficient = False
    elif mean_score < config.min_mean_score:
        reason = "mean_score_below_threshold"
        sufficient = False
    else:
        reason = "sufficient_evidence"
        sufficient = True

    return GateDecision(
        sufficient=sufficient,
        reason=reason,
        top_score=top_score,
        mean_score=mean_score,
        qualifying_docs=qualifying_docs,
        total_docs=total_docs,
    )


def replay_strategy(scores: Sequence[float], config: StrategyConfig) -> GateDecision:
    ordered = sorted([float(score) for score in scores], reverse=True)
    total_docs = len(ordered)
    if not ordered:
        return GateDecision(False, "no_documents", 0.0, 0.0, 0, 0)

    top_score = ordered[0]
    mean_score = mean_top_n(ordered, config.mean_top_n)
    qualifying_docs = sum(score >= config.min_doc_score for score in ordered)

    if top_score < config.min_top_score:
        return GateDecision(
            False,
            "top_score_below_threshold",
            top_score,
            mean_score,
            qualifying_docs,
            total_docs,
        )
    if config.min_docs > 0 and qualifying_docs < config.min_docs:
        return GateDecision(
            False,
            "not_enough_qualifying_documents",
            top_score,
            mean_score,
            qualifying_docs,
            total_docs,
        )
    if config.min_mean_score > 0.0 and mean_score < config.min_mean_score:
        return GateDecision(
            False,
            "mean_score_below_threshold",
            top_score,
            mean_score,
            qualifying_docs,
            total_docs,
        )
    return GateDecision(
        True,
        "sufficient_evidence",
        top_score,
        mean_score,
        qualifying_docs,
        total_docs,
    )


def evaluate_config(cases: Sequence[Dict[str, Any]], config: GateConfig) -> Dict[str, Any]:
    decisions = []
    for case in cases:
        decision = replay_gate(case.get("retrieval_similarity_scores", []), config)
        decisions.append((case, decision))

    answerable = [(c, d) for c, d in decisions if c["expected_behavior"] == ANSWER]
    refusals = [(c, d) for c, d in decisions if c["expected_behavior"] == REFUSE]
    answer_pass = sum(1 for _, d in answerable if d.sufficient)
    refusal_pass = sum(1 for _, d in refusals if d.sufficient)

    return {
        "config": asdict(config),
        "total": len(decisions),
        "answerable": len(answerable),
        "expected_refusals": len(refusals),
        "answerable_evidence_pass_rate": rate(answer_pass, len(answerable)),
        "expected_refusal_gate_pass_rate": rate(refusal_pass, len(refusals)),
        "false_evidence_refusal_rate": rate(len(answerable) - answer_pass, len(answerable)),
        "reason_distribution": distribution(d.reason for _, d in decisions),
        "answerable_pass_ids": [c["id"] for c, d in answerable if d.sufficient],
        "answerable_refusal_ids": [c["id"] for c, d in answerable if not d.sufficient],
        "expected_refusal_gate_pass_ids": [c["id"] for c, d in refusals if d.sufficient],
    }


def evaluate_strategy(
    cases: Sequence[Dict[str, Any]],
    config: StrategyConfig,
) -> Dict[str, Any]:
    decisions = []
    for case in cases:
        decision = replay_strategy(scores_for_case(case, config.score_source), config)
        decisions.append((case, decision))

    answerable = [(c, d) for c, d in decisions if c["expected_behavior"] == ANSWER]
    refusals = [(c, d) for c, d in decisions if c["expected_behavior"] == REFUSE]
    answer_pass = sum(1 for _, d in answerable if d.sufficient)
    refusal_pass = sum(1 for _, d in refusals if d.sufficient)

    return {
        "config": asdict(config),
        "total": len(decisions),
        "answerable": len(answerable),
        "expected_refusals": len(refusals),
        "answerable_evidence_pass_rate": rate(answer_pass, len(answerable)),
        "expected_refusal_gate_pass_rate": rate(refusal_pass, len(refusals)),
        "false_evidence_refusal_rate": rate(len(answerable) - answer_pass, len(answerable)),
        "reason_distribution": distribution(d.reason for _, d in decisions),
        "answerable_pass_ids": [c["id"] for c, d in answerable if d.sufficient],
        "answerable_refusal_ids": [c["id"] for c, d in answerable if not d.sufficient],
        "expected_refusal_gate_pass_ids": [c["id"] for c, d in refusals if d.sufficient],
    }


def score_distribution_report(cases: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    groups: Dict[str, List[Dict[str, Any]]] = {
        "expected_answer": [c for c in cases if c["expected_behavior"] == ANSWER],
        "expected_refusal": [c for c in cases if c["expected_behavior"] == REFUSE],
    }
    for category in sorted({c["category"] for c in cases if c["expected_behavior"] == ANSWER}):
        groups[f"answer_category:{category}"] = [
            c for c in cases if c["expected_behavior"] == ANSWER and c["category"] == category
        ]

    report = {}
    for name, group_cases in groups.items():
        report[name] = describe_case_scores(group_cases)
    answer_tops = [top_score(c) for c in groups["expected_answer"]]
    refusal_tops = [top_score(c) for c in groups["expected_refusal"]]
    report["overlap"] = overlap_summary(answer_tops, refusal_tops)
    return report


def score_source_distribution_report(
    cases: Sequence[Dict[str, Any]], score_source: str
) -> Dict[str, Any]:
    groups: Dict[str, List[Dict[str, Any]]] = {
        "expected_answer": [c for c in cases if c["expected_behavior"] == ANSWER],
        "expected_refusal": [c for c in cases if c["expected_behavior"] == REFUSE],
    }
    report = {}
    for name, group_cases in groups.items():
        top_scores = [top_score_for_source(case, score_source) for case in group_cases]
        mean_scores = [mean_top_n(scores_for_case(case, score_source), 3) for case in group_cases]
        margins = [float(case.get("reranker_margin") or 0.0) for case in group_cases]
        report[name] = {
            "count": len(group_cases),
            "top_score": describe_values(top_scores),
            "top3_mean": describe_values(mean_scores),
            "reranker_margin": describe_values(margins),
        }
    report["overlap"] = {
        "top_score": overlap_summary(
            [top_score_for_source(case, score_source) for case in groups["expected_answer"]],
            [top_score_for_source(case, score_source) for case in groups["expected_refusal"]],
        ),
        "top3_mean": overlap_summary(
            [
                mean_top_n(scores_for_case(case, score_source), 3)
                for case in groups["expected_answer"]
            ],
            [
                mean_top_n(scores_for_case(case, score_source), 3)
                for case in groups["expected_refusal"]
            ],
        ),
        "reranker_margin": overlap_summary(
            [float(case.get("reranker_margin") or 0.0) for case in groups["expected_answer"]],
            [float(case.get("reranker_margin") or 0.0) for case in groups["expected_refusal"]],
        ),
    }
    return report


def describe_case_scores(cases: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    top_scores = [top_score(case) for case in cases]
    top3_means = [mean_top_n(case.get("retrieval_similarity_scores", []), 3) for case in cases]
    rank_scores = {}
    for rank in range(1, 6):
        rank_values = [
            float(case["retrieval_similarity_scores"][rank - 1])
            for case in cases
            if len(case.get("retrieval_similarity_scores", [])) >= rank
        ]
        rank_scores[f"rank_{rank}"] = describe_values(rank_values)

    return {
        "count": len(cases),
        "top_score": describe_values(top_scores),
        "top3_mean": describe_values(top3_means),
        "chunks_above_current_doc_threshold": describe_values(
            [
                sum(
                    float(score) >= CURRENT_DOC
                    for score in case.get("retrieval_similarity_scores", [])
                )
                for case in cases
            ]
        ),
        "rank_scores": rank_scores,
    }


def generate_candidate_values(cases: Sequence[Dict[str, Any]]) -> Dict[str, List[float]]:
    top_scores = [top_score(case) for case in cases]
    top3_means = [mean_top_n(case.get("retrieval_similarity_scores", []), 3) for case in cases]
    observed = top_scores + top3_means
    quantile_points = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60]
    base = {round(percentile(observed, q), 2) for q in quantile_points if observed}
    # Add a compact bounded range around the baseline and observed lower scores.
    bounded = {round(value / 100, 2) for value in range(45, 76, 5)}
    common = sorted(
        value
        for value in (base | bounded | {CURRENT_TOP, CURRENT_DOC, CURRENT_MEAN})
        if 0.0 <= value <= 1.0
    )
    return {
        "min_top_score": common,
        "min_doc_score": common,
        "min_mean_score": common,
    }


def sweep_candidates(cases: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    values = generate_candidate_values(cases)
    results = []
    for top in values["min_top_score"]:
        for doc in values["min_doc_score"]:
            for mean in values["min_mean_score"]:
                config = GateConfig(top, doc, mean)
                metrics = evaluate_config(cases, config)
                results.append(metrics)
    return results


def sweep_strategy_candidates(
    cases: Sequence[Dict[str, Any]],
    *,
    score_source: str,
) -> List[Dict[str, Any]]:
    if score_source == "vector":
        return sweep_candidates(cases)

    values = generate_strategy_values(cases, score_source)
    results: List[Dict[str, Any]] = []

    for top in values["top"]:
        config = StrategyConfig(
            strategy="A_top",
            score_source=score_source,
            min_top_score=top,
        )
        results.append(evaluate_strategy(cases, config))

    for top in values["top"]:
        for doc in values["doc"]:
            for min_docs in values["min_docs"]:
                config = StrategyConfig(
                    strategy="B_top_and_min_docs",
                    score_source=score_source,
                    min_top_score=top,
                    min_doc_score=doc,
                    min_docs=min_docs,
                )
                results.append(evaluate_strategy(cases, config))

    for top in values["top"]:
        for mean in values["mean"]:
            config = StrategyConfig(
                strategy="C_top_and_mean_top3",
                score_source=score_source,
                min_top_score=top,
                min_mean_score=mean,
            )
            results.append(evaluate_strategy(cases, config))

    return results


def generate_strategy_values(
    cases: Sequence[Dict[str, Any]],
    score_source: str,
) -> Dict[str, List[float] | List[int]]:
    scores: List[float] = []
    means: List[float] = []
    for case in cases:
        case_scores = scores_for_case(case, score_source)
        scores.extend(case_scores)
        means.append(mean_top_n(case_scores, 3))

    observed = scores + means
    quantiles = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
    values = {round(percentile(observed, q), 2) for q in quantiles if observed}
    values |= {round(value / 100, 2) for value in range(0, 101, 5)}
    bounded = sorted(value for value in values if 0.0 <= value <= 1.0)
    return {
        "top": bounded,
        "doc": bounded,
        "mean": bounded,
        "min_docs": [1, 2, 3],
    }


def rank_candidates(
    results: Sequence[Dict[str, Any]],
    *,
    safety_bound: float = 0.10,
) -> List[Dict[str, Any]]:
    """Rank candidates using calibration metrics only.

    The safety bound is a hard filter: candidates that admit too many expected
    refusals are discarded. Within that safe envelope, reduce false refusals.
    If nothing satisfies the bound, fall back to strict safety-first ranking.
    """
    eligible = [
        result
        for result in results
        if (result["expected_refusal_gate_pass_rate"] or 0.0) <= safety_bound
    ]

    def eligible_key(result: Dict[str, Any]) -> Tuple[float, float, int, float, float, float]:
        cfg = result["config"]
        return (
            result["expected_refusal_gate_pass_rate"] or 0.0,
            result["false_evidence_refusal_rate"] or 0.0,
            strategy_complexity(cfg),
            -cfg["min_top_score"],
            -cfg.get("min_doc_score", 0.0),
            -cfg.get("min_mean_score", 0.0),
        )

    def fallback_key(result: Dict[str, Any]) -> Tuple[float, float, int, float, float, float]:
        cfg = result["config"]
        return (
            result["expected_refusal_gate_pass_rate"] or 0.0,
            result["false_evidence_refusal_rate"] or 0.0,
            strategy_complexity(cfg),
            -cfg["min_top_score"],
            -cfg.get("min_doc_score", 0.0),
            -cfg.get("min_mean_score", 0.0),
        )

    if eligible:
        return sorted(eligible, key=eligible_key)
    return sorted(results, key=fallback_key)


def strategy_complexity(config: Dict[str, Any]) -> int:
    strategy = config.get("strategy")
    if strategy == "A_top":
        return 1
    if strategy == "B_top_and_min_docs":
        return 2
    if strategy == "C_top_and_mean_top3":
        return 2
    return 3


def shortlist_candidates(ranked: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return up to three distinct conservative/balanced/relaxed candidates."""
    if any("strategy" in result.get("config", {}) for result in ranked):
        return shortlist_strategy_candidates(ranked)

    selected: List[Dict[str, Any]] = []
    seen = set()
    for result in ranked:
        cfg_tuple = tuple(
            result["config"][key] for key in ("min_top_score", "min_doc_score", "min_mean_score")
        )
        if cfg_tuple in seen:
            continue
        selected.append(result)
        seen.add(cfg_tuple)
        if len(selected) == 3:
            break
    labels = ["Candidate A - Conservative", "Candidate B - Balanced", "Candidate C - Relaxed"]
    return [{**candidate, "label": labels[index]} for index, candidate in enumerate(selected)]


def shortlist_strategy_candidates(candidates: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build conservative/balanced/relaxed strategy frontiers from calibration only."""

    def choose(label: str, eligible: Sequence[Dict[str, Any]], key) -> Optional[Dict[str, Any]]:
        if not eligible:
            return None
        selected = sorted(eligible, key=key)[0]
        return {**selected, "label": label}

    positive = [c for c in candidates if (c["answerable_evidence_pass_rate"] or 0.0) > 0.0]
    conservative = choose(
        "Conservative",
        positive,
        lambda c: (
            c["expected_refusal_gate_pass_rate"] or 0.0,
            c["false_evidence_refusal_rate"] or 0.0,
            strategy_complexity(c["config"]),
            -c["config"]["min_top_score"],
        ),
    )
    balanced = choose(
        "Balanced",
        [
            c
            for c in candidates
            if (c["answerable_evidence_pass_rate"] or 0.0) >= 0.50
            and (c["expected_refusal_gate_pass_rate"] or 0.0) <= 0.20
        ],
        lambda c: (
            c["expected_refusal_gate_pass_rate"] or 0.0,
            c["false_evidence_refusal_rate"] or 0.0,
            strategy_complexity(c["config"]),
            -c["config"]["min_top_score"],
        ),
    )
    relaxed = choose(
        "Relaxed",
        candidates,
        lambda c: (
            c["false_evidence_refusal_rate"] or 0.0,
            c["expected_refusal_gate_pass_rate"] or 0.0,
            strategy_complexity(c["config"]),
            -c["config"]["min_top_score"],
        ),
    )

    selected: List[Dict[str, Any]] = []
    seen = set()
    for candidate in (conservative, balanced, relaxed):
        if not candidate:
            continue
        cfg = candidate["config"]
        key = (
            cfg.get("strategy"),
            cfg.get("min_top_score"),
            cfg.get("min_doc_score"),
            cfg.get("min_docs"),
            cfg.get("min_mean_score"),
        )
        if key in seen:
            continue
        selected.append(candidate)
        seen.add(key)
    return selected


def select_preferred_candidate(shortlist: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Choose one calibration-only candidate for validation."""
    if not shortlist:
        return None
    balanced = [
        candidate
        for candidate in shortlist
        if candidate.get("label") == "Balanced"
        and (candidate["answerable_evidence_pass_rate"] or 0.0) >= 0.50
        and (candidate["expected_refusal_gate_pass_rate"] or 0.0) <= 0.20
        and (candidate["false_evidence_refusal_rate"] or 0.0) <= 0.25
    ]
    if balanced:
        return balanced[0]
    useful = [
        candidate
        for candidate in shortlist
        if (candidate["answerable_evidence_pass_rate"] or 0.0) > 0.0
    ]
    return useful[0] if useful else shortlist[0]


def calibration_payload(
    baseline_path: Path | str = BASELINE_JSON,
    split_path: Path | str = SPLIT_JSON,
    safety_bound: float = 0.10,
    score_source: str = "vector",
) -> Dict[str, Any]:
    cases = load_eval_cases(baseline_path)
    split = load_split(split_path)
    validate_split(cases, split)
    calibration_cases = cases_for_ids(cases, split["calibration"])
    validation_cases = cases_for_ids(cases, split["validation"])
    baseline_config = GateConfig(CURRENT_TOP, CURRENT_DOC, CURRENT_MEAN)
    sweep = sweep_strategy_candidates(calibration_cases, score_source=score_source)
    ranked = rank_candidates(sweep, safety_bound=safety_bound)
    shortlist = shortlist_candidates(ranked)
    distributions = (
        score_source_distribution_report(cases, score_source)
        if score_source != "vector"
        else score_distribution_report(cases)
    )
    return {
        "created_at": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "baseline_path": str(baseline_path),
        "split_path": str(split_path),
        "score_source": score_source,
        "split_counts": split_counts(cases, split),
        "score_distributions": distributions,
        "baseline_gate_on_calibration": evaluate_config(calibration_cases, baseline_config),
        "baseline_gate_on_validation": evaluate_config(validation_cases, baseline_config),
        "candidate_value_counts": (
            {
                k: len(v)
                for k, v in generate_strategy_values(calibration_cases, score_source).items()
            }
            if score_source != "vector"
            else {k: len(v) for k, v in generate_candidate_values(calibration_cases).items()}
        ),
        "candidates_evaluated": len(sweep),
        "safety_bound": safety_bound,
        "shortlist": shortlist,
        "selected": select_preferred_candidate(shortlist),
    }


def split_counts(cases: Sequence[Dict[str, Any]], split: Dict[str, List[str]]) -> Dict[str, Any]:
    output = {}
    for name in ("calibration", "validation"):
        subset = cases_for_ids(cases, split[name])
        output[name] = {
            "total": len(subset),
            "answer": sum(1 for case in subset if case["expected_behavior"] == ANSWER),
            "refuse": sum(1 for case in subset if case["expected_behavior"] == REFUSE),
            "categories": distribution(case["category"] for case in subset),
        }
    return output


def render_report(payload: Dict[str, Any]) -> str:
    selected = payload["selected"]
    baseline = payload["baseline_gate_on_calibration"]
    lines = [
        "# Evidence Gate Offline Calibration",
        "",
        f"Baseline artifact: `{payload['baseline_path']}`",
        f"Split: `{payload['split_path']}`",
        f"Score source: `{payload.get('score_source', 'vector')}`",
        f"Safety bound: expected-refusal gate-pass rate <= {payload['safety_bound']:.1%}",
        "",
        "## Split Counts",
        "",
        "Split | Total | Answer | Refuse",
        "--- | ---: | ---: | ---:",
    ]
    for name, counts in payload["split_counts"].items():
        lines.append(f"{name} | {counts['total']} | {counts['answer']} | {counts['refuse']}")

    overlap = payload["score_distributions"]["overlap"]
    top_overlap = overlap.get("top_score", overlap)
    lines.extend(
        [
            "",
            "## Retrieval Score Overlap",
            "",
            f"- Answerable top-score min: {payload['score_distributions']['expected_answer']['top_score']['min']:.4f}",
            f"- Refusal top-score max: {payload['score_distributions']['expected_refusal']['top_score']['max']:.4f}",
            f"- Answerable cases at or below refusal max: {top_overlap['answer_scores_at_or_below_refusal_max']}",
            f"- Refusal cases at or above answer min: {top_overlap['refusal_scores_at_or_above_answer_min']}",
            "",
            "## Baseline Gate On Calibration",
            "",
            f"- Answerable evidence-pass rate: {fmt_rate(baseline['answerable_evidence_pass_rate'])}",
            f"- Expected-refusal gate-pass rate: {fmt_rate(baseline['expected_refusal_gate_pass_rate'])}",
            f"- False evidence-refusal rate: {fmt_rate(baseline['false_evidence_refusal_rate'])}",
            "",
            "## Shortlist",
            "",
            "Label | Top | Doc | Mean | Answer pass | Refusal gate pass | False refusal",
            "--- | ---: | ---: | ---: | ---: | ---: | ---:",
        ]
    )
    for candidate in payload["shortlist"]:
        cfg = candidate["config"]
        lines.append(
            f"{candidate['label']} | {cfg['min_top_score']:.2f} | "
            f"{cfg.get('min_doc_score', 0.0):.2f} | {cfg.get('min_mean_score', 0.0):.2f} | "
            f"{fmt_rate(candidate['answerable_evidence_pass_rate'])} | "
            f"{fmt_rate(candidate['expected_refusal_gate_pass_rate'])} | "
            f"{fmt_rate(candidate['false_evidence_refusal_rate'])}"
        )
    if selected:
        cfg = selected["config"]
        lines.extend(
            [
                "",
                "## Selected From Calibration Only",
                "",
                f"Selected: `{selected['label']}` with top={cfg['min_top_score']:.2f}, "
                f"doc={cfg.get('min_doc_score', 0.0):.2f}, "
                f"mean={cfg.get('min_mean_score', 0.0):.2f}.",
                "Selection used calibration data only; validation metrics are not used here.",
            ]
        )
    lines.append("")
    return "\n".join(lines)


def describe_values(values: Sequence[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "q1": None,
            "median": None,
            "q3": None,
            "max": None,
            "mean": None,
        }
    ordered = sorted(float(v) for v in values)
    return {
        "count": len(ordered),
        "min": round(ordered[0], 4),
        "q1": round(percentile(ordered, 0.25), 4),
        "median": round(statistics.median(ordered), 4),
        "q3": round(percentile(ordered, 0.75), 4),
        "max": round(ordered[-1], 4),
        "mean": round(sum(ordered) / len(ordered), 4),
    }


def top_score(case: Dict[str, Any]) -> float:
    scores = case.get("retrieval_similarity_scores", [])
    return float(scores[0]) if scores else 0.0


def top_score_for_source(case: Dict[str, Any], score_source: str) -> float:
    scores = scores_for_case(case, score_source)
    return float(scores[0]) if scores else 0.0


def scores_for_case(case: Dict[str, Any], score_source: str) -> List[float]:
    if score_source == "vector":
        source_scores = case.get("vector_scores") or case.get("retrieval_similarity_scores", [])
    elif score_source == "reranker":
        source_scores = case.get("reranker_scores", [])
    else:
        raise ValueError("score_source must be 'vector' or 'reranker'")
    return [float(score) for score in source_scores if score is not None]


def mean_top_n(scores: Sequence[float], n: int) -> float:
    selected = [float(score) for score in scores[:n]]
    return sum(selected) / len(selected) if selected else 0.0


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1 - frac) + ordered[hi] * frac


def overlap_summary(answer_tops: Sequence[float], refusal_tops: Sequence[float]) -> Dict[str, Any]:
    if not answer_tops or not refusal_tops:
        return {}
    refusal_max = max(refusal_tops)
    answer_min = min(answer_tops)
    return {
        "answer_min": round(answer_min, 4),
        "refusal_max": round(refusal_max, 4),
        "score_ranges_overlap": answer_min <= refusal_max,
        "answer_scores_at_or_below_refusal_max": sum(score <= refusal_max for score in answer_tops),
        "refusal_scores_at_or_above_answer_min": sum(score >= answer_min for score in refusal_tops),
    }


def distribution(values: Iterable[str]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def rate(numerator: int, denominator: int) -> Optional[float]:
    if denominator == 0:
        return None
    return round(numerator / denominator, 4)


def fmt_rate(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    return f"{value:.1%}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline EvidenceGate calibration")
    parser.add_argument("--baseline", type=Path, default=BASELINE_JSON)
    parser.add_argument("--split", type=Path, default=SPLIT_JSON)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--safety-bound", type=float, default=0.10)
    parser.add_argument("--score-source", choices=["vector", "reranker"], default="vector")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = calibration_payload(
        args.baseline,
        args.split,
        safety_bound=args.safety_bound,
        score_source=args.score_source,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    args.output_md.write_text(render_report(payload), encoding="utf-8")
    selected = payload["selected"]
    print(f"Loaded baseline: {args.baseline}")
    print(f"Candidates evaluated: {payload['candidates_evaluated']}")
    if selected:
        cfg = selected["config"]
        print(
            "Selected "
            f"top={cfg['min_top_score']:.2f} "
            f"doc={cfg['min_doc_score']:.2f} "
            f"mean={cfg['min_mean_score']:.2f}"
        )
        print(
            "Calibration metrics: "
            f"answer_pass={fmt_rate(selected['answerable_evidence_pass_rate'])} "
            f"refusal_gate_pass={fmt_rate(selected['expected_refusal_gate_pass_rate'])} "
            f"false_refusal={fmt_rate(selected['false_evidence_refusal_rate'])}"
        )
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

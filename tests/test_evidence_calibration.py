"""Offline tests for EvidenceGate calibration tooling."""

import argparse
import json
from pathlib import Path

from scripts import calibrate_evidence_gate as calib
from scripts.run_evaluation import _build_evidence_gate


def _case(case_id, expected, category, scores):
    return {
        "id": case_id,
        "expected_behavior": expected,
        "category": category,
        "retrieval_similarity_scores": scores,
        "reranker_scores": scores,
        "reranker_margin": scores[0] - scores[1] if len(scores) > 1 else 0.0,
    }


def _cases():
    return [
        _case("a1", "answer", "direct_fact", [0.82, 0.70, 0.68]),
        _case("a2", "answer", "comparison", [0.62, 0.58, 0.54]),
        _case("r1", "refuse", "out_of_domain", [0.30, 0.25, 0.20]),
        _case("r2", "refuse", "adversarial", [0.46, 0.40, 0.35]),
    ]


def test_fixed_split_file_is_deterministic_and_complete():
    cases = calib.load_eval_cases("eval_results/run_20260815_165508.json")
    split = calib.load_split("data/eval/eval_split.json")

    calib.validate_split(cases, split)
    counts = calib.split_counts(cases, split)

    assert counts["calibration"]["total"] == 40
    assert counts["validation"]["total"] == 20
    assert counts["calibration"]["answer"] == 25
    assert counts["validation"]["refuse"] == 6


def test_score_distribution_calculations():
    report = calib.score_distribution_report(_cases())

    assert report["expected_answer"]["top_score"]["min"] == 0.62
    assert report["expected_refusal"]["top_score"]["max"] == 0.46
    assert report["expected_answer"]["top3_mean"]["count"] == 2
    assert report["answer_category:comparison"]["count"] == 1


def test_gate_replay_matches_expected_reasons():
    strict = calib.GateConfig(0.70, 0.65, 0.60)
    relaxed = calib.GateConfig(0.50, 0.45, 0.45)

    assert calib.replay_gate([0.62, 0.58, 0.54], strict).reason == "top_score_below_threshold"
    assert calib.replay_gate([0.62, 0.58, 0.54], relaxed).sufficient is True
    assert calib.replay_gate([], relaxed).reason == "no_documents"


def test_candidate_sweep_and_safety_first_ranking():
    results = [
        {
            "config": {"min_top_score": 0.50, "min_doc_score": 0.45, "min_mean_score": 0.45},
            "expected_refusal_gate_pass_rate": 0.2,
            "false_evidence_refusal_rate": 0.0,
        },
        {
            "config": {"min_top_score": 0.60, "min_doc_score": 0.55, "min_mean_score": 0.50},
            "expected_refusal_gate_pass_rate": 0.0,
            "false_evidence_refusal_rate": 0.5,
        },
    ]

    ranked = calib.rank_candidates(results, safety_bound=0.5)

    assert ranked[0]["expected_refusal_gate_pass_rate"] == 0.0
    assert ranked[0]["false_evidence_refusal_rate"] == 0.5


def test_reranker_strategy_sweep_includes_required_strategies():
    results = calib.sweep_strategy_candidates(_cases(), score_source="reranker")
    strategies = {result["config"]["strategy"] for result in results}

    assert {"A_top", "B_top_and_min_docs", "C_top_and_mean_top3"}.issubset(strategies)


def test_reranker_candidate_selection_uses_calibration_only(tmp_path):
    cases = [
        _case("a1", "answer", "direct_fact", [0.95, 0.90, 0.85]),
        _case("r1", "refuse", "out_of_domain", [0.20, 0.10, 0.05]),
        _case("a2", "answer", "direct_fact", [0.10, 0.09, 0.08]),
        _case("r2", "refuse", "out_of_domain", [0.99, 0.98, 0.97]),
    ]
    artifact = tmp_path / "reranker.json"
    split = tmp_path / "split.json"
    artifact.write_text(json.dumps({"cases": cases}), encoding="utf-8")
    split.write_text(
        json.dumps({"calibration": ["a1", "r1"], "validation": ["a2", "r2"]}),
        encoding="utf-8",
    )

    payload = calib.calibration_payload(
        artifact,
        split,
        safety_bound=0.0,
        score_source="reranker",
    )

    assert payload["score_source"] == "reranker"
    assert payload["split_counts"]["validation"]["total"] == 2
    assert set(payload["selected"]["answerable_pass_ids"]).issubset({"a1"})


def test_no_validation_leakage_in_selected_candidate(tmp_path):
    cases = _cases()
    baseline = tmp_path / "baseline.json"
    split = tmp_path / "split.json"
    baseline.write_text(json.dumps({"cases": cases}), encoding="utf-8")
    split.write_text(
        json.dumps({"calibration": ["a1", "r1"], "validation": ["a2", "r2"]}),
        encoding="utf-8",
    )

    payload = calib.calibration_payload(baseline, split, safety_bound=0.5)

    assert payload["split_counts"]["validation"]["total"] == 2
    assert set(payload["selected"]["answerable_pass_ids"]).issubset({"a1"})


def test_candidate_serialization_and_report(tmp_path):
    payload = calib.calibration_payload(
        "eval_results/run_20260815_165508.json",
        "data/eval/eval_split.json",
    )
    output = tmp_path / "candidate.json"
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    loaded = json.loads(output.read_text(encoding="utf-8"))
    report = calib.render_report(loaded)

    assert loaded["selected"]["config"]["min_top_score"] is not None
    assert "# Evidence Gate Offline Calibration" in report
    assert "Selection used calibration data only" in report


def test_configuration_override_behavior():
    args = argparse.Namespace(
        evidence_min_top_score=0.55,
        evidence_min_doc_score=0.50,
        evidence_min_mean_score=0.45,
        evidence_min_docs=2,
        evidence_score_source="reranker",
    )

    gate = _build_evidence_gate(args)

    assert gate.min_top_score == 0.55
    assert gate.min_doc_score == 0.50
    assert gate.min_mean_score == 0.45
    assert gate.min_docs == 2
    assert gate.score_source == "reranker"


def test_no_override_returns_none():
    args = argparse.Namespace(
        evidence_min_top_score=None,
        evidence_min_doc_score=None,
        evidence_min_mean_score=None,
    )

    assert _build_evidence_gate(args) is None

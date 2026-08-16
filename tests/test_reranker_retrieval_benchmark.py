"""Offline tests for retrieval-only reranker benchmark helpers."""

from scripts.run_reranker_retrieval_benchmark import (
    build_case_record,
    separation_summary,
    split_name_for,
    validate_split_coverage,
)


def _example(case_id="df-001", expected="answer"):
    return {
        "id": case_id,
        "category": "direct_fact",
        "question": "What is a gNB?",
        "expected_behavior": expected,
        "gold_spec_numbers": ["38.300"],
        "gold_sources": ["38300"],
    }


def test_build_case_record_preserves_vector_and_reranked_scores():
    candidates = [
        {
            "source": "a.docx",
            "spec_number": "38.300",
            "rank_before_reranking": 2,
            "vector_similarity": 0.5,
            "rank_after_reranking": 1,
            "reranker_raw_score": 2.0,
            "reranker_score": 0.9,
        },
        {
            "source": "b.docx",
            "spec_number": "38.401",
            "rank_before_reranking": 1,
            "vector_similarity": 0.8,
            "rank_after_reranking": 2,
            "reranker_raw_score": 0.1,
            "reranker_score": 0.6,
        },
    ]

    record = build_case_record(
        _example(),
        candidates,
        split_name="calibration",
        retrieve_latency=1.23,
        rerank_latency=0.45,
    )

    assert record["top_vector_score"] == 0.8
    assert record["top_reranker_score"] == 0.9
    assert record["second_best_reranker_score"] == 0.6
    assert record["reranker_margin"] == 0.30000000000000004
    assert record["candidates"][0]["reranked_rank"] == 1
    assert record["candidates"][1]["vector_rank"] == 1


def test_separation_summary_reports_overlap_counts():
    cases = [
        {"expected_behavior": "answer", "top_reranker_score": 0.8},
        {"expected_behavior": "answer", "top_reranker_score": 0.4},
        {"expected_behavior": "refuse", "top_reranker_score": 0.5},
    ]

    summary = separation_summary(cases, "top_reranker_score")

    assert summary["answer"]["p90"] is not None
    assert summary["overlap"]["answerable_at_or_below_refusal_max"] == 1
    assert summary["overlap"]["refusals_at_or_above_answerable_min"] == 1


def test_split_reuse_and_coverage_validation():
    split = {"calibration": ["a"], "validation": ["b"]}

    assert split_name_for("a", split) == "calibration"
    assert split_name_for("b", split) == "validation"
    validate_split_coverage([_example("a"), _example("b")], split)

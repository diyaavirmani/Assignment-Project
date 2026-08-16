"""Tests for src/core/evidence_gate.py."""

import pytest

from src.core.evidence_gate import (
    EvidenceGate,
    REASON_DISABLED,
    REASON_MISSING_SCORES,
    REASON_MEAN_SCORE_LOW,
    REASON_NO_DOCUMENTS,
    REASON_NOT_ENOUGH_DOCS,
    REASON_SUFFICIENT,
    REASON_TOP_SCORE_LOW,
)


def _doc(score: float) -> dict:
    return {
        "text": "chunk",
        "source": "38300.docx",
        "chunk_index": 0,
        "similarity": score,
    }


def _reranked_doc(vector_score: float, reranker_score: float) -> dict:
    doc = _doc(vector_score)
    doc["vector_similarity"] = vector_score
    doc["reranker_score"] = reranker_score
    return doc


class TestEvidenceGate:
    def test_no_documents_is_insufficient(self):
        gate = EvidenceGate()

        decision = gate.evaluate([])

        assert decision.sufficient is False
        assert decision.reason == REASON_NO_DOCUMENTS
        assert decision.total_docs == 0

    def test_extremely_weak_result_is_insufficient(self):
        gate = EvidenceGate(score_source="vector", min_top_score=0.7, min_doc_score=0.65)

        decision = gate.evaluate([_doc(0.1)])

        assert decision.sufficient is False
        assert decision.reason == REASON_TOP_SCORE_LOW

    def test_best_result_below_top_score_threshold(self):
        gate = EvidenceGate(score_source="vector", min_top_score=0.8, min_doc_score=0.5)

        decision = gate.evaluate([_doc(0.79), _doc(0.75)])

        assert decision.sufficient is False
        assert decision.reason == REASON_TOP_SCORE_LOW

    def test_strong_top_but_not_enough_qualifying_documents(self):
        gate = EvidenceGate(
            score_source="vector",
            min_top_score=0.8,
            min_doc_score=0.75,
            min_docs=2,
            min_mean_score=0.0,
        )

        decision = gate.evaluate([_doc(0.9), _doc(0.7)])

        assert decision.sufficient is False
        assert decision.reason == REASON_NOT_ENOUGH_DOCS
        assert decision.qualifying_docs == 1

    def test_mean_top_n_score_below_threshold(self):
        gate = EvidenceGate(
            score_source="vector",
            min_top_score=0.8,
            min_doc_score=0.1,
            min_docs=1,
            mean_top_n=3,
            min_mean_score=0.7,
        )

        decision = gate.evaluate([_doc(0.9), _doc(0.4), _doc(0.4)])

        assert decision.sufficient is False
        assert decision.reason == REASON_MEAN_SCORE_LOW
        assert decision.mean_score == pytest.approx(0.5667, abs=0.0001)

    def test_strong_evidence_is_sufficient(self):
        gate = EvidenceGate(
            score_source="vector",
            min_top_score=0.7,
            min_doc_score=0.65,
            min_docs=2,
            mean_top_n=3,
            min_mean_score=0.6,
        )

        decision = gate.evaluate([_doc(0.9), _doc(0.8), _doc(0.7)])

        assert decision.sufficient is True
        assert decision.reason == REASON_SUFFICIENT
        assert decision.top_score == 0.9
        assert decision.qualifying_docs == 3
        assert decision.score_source == "vector"

    def test_vector_score_source_uses_vector_similarity(self):
        gate = EvidenceGate(
            score_source="vector",
            min_top_score=0.7,
            min_doc_score=0.7,
            min_docs=1,
            min_mean_score=0.0,
        )

        decision = gate.evaluate([_reranked_doc(0.8, 0.1)])

        assert decision.sufficient is True
        assert decision.top_score == 0.8
        assert decision.score_source == "vector"

    def test_reranker_score_source_uses_reranker_score(self):
        gate = EvidenceGate(
            score_source="reranker",
            min_top_score=0.7,
            min_doc_score=0.7,
            min_docs=1,
            min_mean_score=0.0,
        )

        decision = gate.evaluate([_reranked_doc(0.1, 0.8)])

        assert decision.sufficient is True
        assert decision.top_score == 0.8
        assert decision.score_source == "reranker"

    def test_missing_reranker_scores_fail_safely(self):
        gate = EvidenceGate(score_source="reranker", min_top_score=0.1)

        decision = gate.evaluate([_doc(0.9)])

        assert decision.sufficient is False
        assert decision.reason == REASON_MISSING_SCORES
        assert decision.score_source == "reranker"

    def test_exactly_on_threshold_passes(self):
        gate = EvidenceGate(
            score_source="vector",
            min_top_score=0.7,
            min_doc_score=0.7,
            min_docs=1,
            mean_top_n=3,
            min_mean_score=0.7,
        )

        decision = gate.evaluate([_doc(0.7)])

        assert decision.sufficient is True
        assert decision.reason == REASON_SUFFICIENT

    def test_fewer_documents_than_mean_top_n_is_graceful(self):
        gate = EvidenceGate(
            score_source="vector",
            min_top_score=0.8,
            min_doc_score=0.7,
            min_docs=1,
            mean_top_n=5,
            min_mean_score=0.8,
        )

        decision = gate.evaluate([_doc(0.8)])

        assert decision.sufficient is True
        assert decision.mean_score == pytest.approx(0.8)

    def test_gate_disabled_allows_previous_behavior(self):
        gate = EvidenceGate(enabled=False, min_top_score=1.0, min_mean_score=1.0)

        decision = gate.evaluate([_doc(0.0)])

        assert decision.sufficient is True
        assert decision.reason == REASON_DISABLED

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"min_top_score": -0.1},
            {"min_doc_score": 1.1},
            {"min_mean_score": 1.1},
            {"min_docs": -1},
            {"mean_top_n": 0},
            {"score_source": "bad"},
        ],
    )
    def test_invalid_configuration_raises(self, kwargs):
        with pytest.raises(ValueError):
            EvidenceGate(**kwargs)

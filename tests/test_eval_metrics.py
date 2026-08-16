"""
Unit tests for scripts.eval.metrics — standard IR metric functions.

All tests use mocked/inline data; no vector store, no LLM, no real index required.
These tests verify the mathematical correctness of:
  - hit_rate_at_k
  - recall_at_k
  - mrr
  - ndcg_at_k (binary AND graded)
  - context_precision  (heuristic)
  - context_recall_keywords  (heuristic)
  - compute_retrieval_metrics  (bundle)

And for scripts.eval.regression:
  - _get_nested
  - compare_to_baseline (no-baseline path)
  - save_baseline / load_baseline (temp-file path)

And the new additions (Round 2):
  - ndcg_at_k graded mode (graded_relevance dict, 0/1/2 gains)
  - _is_relevant word-boundary / spec-number-token matching (tightened)
  - is_refusal / refusal_rate (negative/out-of-corpus eval axis)

Run:
    .venv/bin/python -m pytest tests/test_eval_metrics.py -q
"""
import json
import math
import sys
import tempfile
from pathlib import Path
from typing import List, Dict

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.eval.metrics import (
    hit_rate_at_k,
    recall_at_k,
    mrr,
    ndcg_at_k,
    context_precision,
    context_recall_keywords,
    compute_retrieval_metrics,
    is_refusal,
    refusal_rate,
    _is_relevant,
    pass_sim_threshold,
    PASS_SIM_THRESHOLDS,
    DEFAULT_PASS_SIM_THRESHOLD,
    REFUSAL_SIM_THRESHOLD,
)
from scripts.eval.calibrate_threshold import extract_populations, youden_optimal
from scripts.eval.regression import (
    _get_nested,
    compare_to_baseline,
    save_baseline,
    load_baseline,
    TRACKED_METRICS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_docs(sources: List[str], texts: List[str] = None, sims: List[float] = None) -> List[Dict]:
    """Build a minimal doc list for testing."""
    n = len(sources)
    texts = texts or ["content"] * n
    sims = sims or [0.9 - 0.05 * i for i in range(n)]
    return [
        {"source": src, "text": txt, "similarity": sim}
        for src, txt, sim in zip(sources, texts, sims)
    ]


RELEVANT = ["38300", "38401"]


# ---------------------------------------------------------------------------
# _is_relevant
# ---------------------------------------------------------------------------

class TestIsRelevant:
    def test_match(self):
        assert _is_relevant({"source": "38300-g10.docx"}, ["38300"]) is True

    def test_no_match(self):
        assert _is_relevant({"source": "other.docx"}, ["38300"]) is False

    def test_multiple_patterns_first_matches(self):
        assert _is_relevant({"source": "38401-h10.docx"}, ["38300", "38401"]) is True

    def test_empty_relevant_sources(self):
        assert _is_relevant({"source": "38300.docx"}, []) is False

    def test_missing_source_key(self):
        assert _is_relevant({}, ["38300"]) is False


# ---------------------------------------------------------------------------
# hit_rate_at_k
# ---------------------------------------------------------------------------

class TestHitRateAtK:
    def test_first_doc_relevant(self):
        docs = _make_docs(["38300-g10.docx", "other.docx"])
        assert hit_rate_at_k(docs, ["38300"], k=5) == 1.0

    def test_no_relevant_docs(self):
        docs = _make_docs(["other1.docx", "other2.docx"])
        assert hit_rate_at_k(docs, ["38300"], k=5) == 0.0

    def test_relevant_beyond_k_not_counted(self):
        docs = _make_docs(["other1.docx", "other2.docx", "38300-g10.docx"])
        # k=2 → only first 2 checked; relevant is at position 3
        assert hit_rate_at_k(docs, ["38300"], k=2) == 0.0

    def test_relevant_within_k_counts(self):
        docs = _make_docs(["other1.docx", "38300-g10.docx", "other2.docx"])
        assert hit_rate_at_k(docs, ["38300"], k=2) == 1.0

    def test_empty_docs(self):
        assert hit_rate_at_k([], ["38300"], k=5) == 0.0

    def test_empty_relevant_sources(self):
        docs = _make_docs(["38300-g10.docx"])
        assert hit_rate_at_k(docs, [], k=5) == 0.0

    def test_returns_1_or_0(self):
        docs = _make_docs(["38300-g10.docx", "38401-h10.docx"])
        result = hit_rate_at_k(docs, ["38300"], k=5)
        assert result in (0.0, 1.0)


# ---------------------------------------------------------------------------
# recall_at_k
# ---------------------------------------------------------------------------

class TestRecallAtK:
    def test_all_sources_found(self):
        docs = _make_docs(["38300-g10.docx", "38401-h10.docx", "other.docx"])
        assert recall_at_k(docs, ["38300", "38401"], k=5) == 1.0

    def test_no_sources_found(self):
        docs = _make_docs(["other1.docx", "other2.docx"])
        assert recall_at_k(docs, ["38300", "38401"], k=5) == 0.0

    def test_partial_recall(self):
        docs = _make_docs(["38300-g10.docx", "other.docx"])
        assert recall_at_k(docs, ["38300", "38401"], k=5) == 0.5

    def test_k_cutoff_respected(self):
        # "38401" appears only at position 4 (index 3); k=3 should miss it
        docs = _make_docs(["38300-g10.docx", "other1.docx", "other2.docx", "38401-h10.docx"])
        assert recall_at_k(docs, ["38300", "38401"], k=3) == 0.5

    def test_single_source(self):
        docs = _make_docs(["38300-g10.docx"])
        assert recall_at_k(docs, ["38300"], k=5) == 1.0

    def test_empty_docs(self):
        assert recall_at_k([], ["38300"], k=5) == 0.0

    def test_empty_relevant_sources(self):
        docs = _make_docs(["38300-g10.docx"])
        assert recall_at_k(docs, [], k=5) == 0.0

    def test_duplicate_source_counts_once(self):
        # Both docs from 38300 — should only count as 1 unique find
        docs = _make_docs(["38300-g10.docx", "38300-g20.docx"])
        # relevant_sources = ["38300", "38401"], found = {38300}, recall = 0.5
        assert recall_at_k(docs, ["38300", "38401"], k=5) == 0.5

    def test_result_in_range(self):
        docs = _make_docs(["38300-g10.docx", "other.docx"])
        r = recall_at_k(docs, ["38300", "38401"], k=5)
        assert 0.0 <= r <= 1.0

    def test_no_substring_false_positive(self):
        # Regression guard: recall_at_k must route through the tightened
        # _is_relevant matcher, not a raw substring test. "38300" must NOT
        # be counted as found in "383001-h10.docx" (trailing digit) — the
        # naive `rs in source` check would wrongly score this 1.0.
        docs = _make_docs(["383001-h10.docx"])
        assert recall_at_k(docs, ["38300"], k=5) == 0.0

    def test_no_short_token_false_positive(self):
        # "300" must not match the longer spec number "38300".
        docs = _make_docs(["38300-g10.docx"])
        assert recall_at_k(docs, ["300"], k=5) == 0.0


# ---------------------------------------------------------------------------
# mrr
# ---------------------------------------------------------------------------

class TestMRR:
    def test_first_position(self):
        docs = _make_docs(["38300-g10.docx", "other.docx"])
        assert mrr(docs, ["38300"]) == 1.0

    def test_second_position(self):
        docs = _make_docs(["other.docx", "38300-g10.docx"])
        assert mrr(docs, ["38300"]) == 0.5

    def test_third_position(self):
        docs = _make_docs(["other1.docx", "other2.docx", "38300-g10.docx"])
        expected = round(1.0 / 3, 10)
        assert abs(mrr(docs, ["38300"]) - expected) < 1e-9

    def test_no_relevant(self):
        docs = _make_docs(["other1.docx", "other2.docx"])
        assert mrr(docs, ["38300"]) == 0.0

    def test_empty_docs(self):
        assert mrr([], ["38300"]) == 0.0

    def test_empty_relevant_sources(self):
        docs = _make_docs(["38300-g10.docx"])
        assert mrr(docs, []) == 0.0

    def test_first_relevant_wins(self):
        # 38401 at position 2, 38300 at position 3 — first relevant is 38401 → MRR = 0.5
        docs = _make_docs(["other.docx", "38401-h10.docx", "38300-g10.docx"])
        assert mrr(docs, ["38300", "38401"]) == 0.5

    def test_result_in_range(self):
        docs = _make_docs(["38300-g10.docx"])
        assert 0.0 <= mrr(docs, ["38300"]) <= 1.0


# ---------------------------------------------------------------------------
# ndcg_at_k
# ---------------------------------------------------------------------------

class TestNDCGAtK:
    def test_perfect_ranking(self):
        """All relevant docs first → nDCG = 1.0."""
        docs = _make_docs(["38300-g10.docx", "38401-h10.docx", "other.docx"])
        score = ndcg_at_k(docs, ["38300", "38401"], k=5)
        assert score == 1.0

    def test_no_relevant_docs(self):
        docs = _make_docs(["other1.docx", "other2.docx"])
        assert ndcg_at_k(docs, ["38300"], k=5) == 0.0

    def test_empty_docs(self):
        assert ndcg_at_k([], ["38300"], k=5) == 0.0

    def test_empty_relevant_sources(self):
        docs = _make_docs(["38300-g10.docx"])
        assert ndcg_at_k(docs, [], k=5) == 0.0

    def test_result_in_0_1_range(self):
        docs = _make_docs(["other.docx", "38300-g10.docx", "other2.docx"])
        score = ndcg_at_k(docs, ["38300"], k=5)
        assert 0.0 <= score <= 1.0

    def test_lower_score_for_relevant_at_position_2(self):
        """Relevant at position 2 should score lower than at position 1."""
        docs_pos1 = _make_docs(["38300-g10.docx", "other.docx"])
        docs_pos2 = _make_docs(["other.docx", "38300-g10.docx"])
        assert ndcg_at_k(docs_pos1, ["38300"], k=5) > ndcg_at_k(docs_pos2, ["38300"], k=5)

    def test_single_relevant_at_position_1(self):
        """One relevant doc at position 1, k=5: DCG = 1/log2(2) = 1.0, IDCG = 1.0."""
        docs = _make_docs(["38300-g10.docx", "other1.docx", "other2.docx"])
        score = ndcg_at_k(docs, ["38300"], k=5)
        assert abs(score - 1.0) < 1e-9

    def test_known_value_position_2(self):
        """One relevant doc at rank 2: DCG = 1/log2(3), IDCG = 1/log2(2) = 1."""
        docs = _make_docs(["other.docx", "38300-g10.docx"])
        score = ndcg_at_k(docs, ["38300"], k=5)
        expected = (1.0 / math.log2(3)) / (1.0 / math.log2(2))
        assert abs(score - expected) < 1e-4


# ---------------------------------------------------------------------------
# context_precision (heuristic)
# ---------------------------------------------------------------------------

class TestContextPrecision:
    def test_all_relevant(self):
        docs = _make_docs(["38300-g10.docx", "38300-g20.docx"])
        assert context_precision(docs, ["38300"]) == 1.0

    def test_none_relevant(self):
        docs = _make_docs(["other.docx"])
        assert context_precision(docs, ["38300"]) == 0.0

    def test_partial_relevant(self):
        docs = _make_docs(["38300-g10.docx", "other.docx"])
        assert context_precision(docs, ["38300"]) == 0.5

    def test_empty_docs(self):
        assert context_precision([], ["38300"]) == 0.0

    def test_empty_relevant_sources(self):
        docs = _make_docs(["38300-g10.docx"])
        assert context_precision(docs, []) == 0.0


# ---------------------------------------------------------------------------
# context_recall_keywords (heuristic)
# ---------------------------------------------------------------------------

class TestContextRecallKeywords:
    def test_all_keywords_found(self):
        docs = _make_docs(["38300-g10.docx"], texts=["gnb base station node ran"])
        score = context_recall_keywords(docs, ["gnb", "base station", "node", "ran"])
        assert score == 1.0

    def test_no_keywords_found(self):
        docs = _make_docs(["38300-g10.docx"], texts=["Completely unrelated content"])
        score = context_recall_keywords(docs, ["gnb", "base station"])
        assert score == 0.0

    def test_partial_keywords(self):
        docs = _make_docs(["38300-g10.docx"], texts=["The gnb architecture"])
        score = context_recall_keywords(docs, ["gnb", "base station", "node", "ran"])
        assert score == 0.25

    def test_empty_docs(self):
        assert context_recall_keywords([], ["gnb"]) == 0.0

    def test_empty_keywords(self):
        docs = _make_docs(["38300-g10.docx"], texts=["some content"])
        assert context_recall_keywords(docs, []) == 0.0

    def test_uses_top_3_only(self):
        # Keyword only in 4th doc — must not be found
        docs = [
            {"source": "a", "text": "doc one", "similarity": 0.9},
            {"source": "b", "text": "doc two", "similarity": 0.8},
            {"source": "c", "text": "doc three", "similarity": 0.7},
            {"source": "d", "text": "gnb is here", "similarity": 0.6},
        ]
        assert context_recall_keywords(docs, ["gnb"]) == 0.0

    def test_case_insensitive(self):
        docs = _make_docs(["38300.docx"], texts=["The GNB base station"])
        score = context_recall_keywords(docs, ["gnb", "BASE STATION"])
        assert score == 1.0


# ---------------------------------------------------------------------------
# compute_retrieval_metrics (bundle)
# ---------------------------------------------------------------------------

class TestComputeRetrievalMetrics:
    def _make_perfect(self):
        return _make_docs(
            ["38300-g10.docx", "38401-h10.docx", "other.docx"],
            texts=["gnb base station node ran", "interface cu du f1", "other content"],
        )

    def test_returns_all_required_keys(self):
        docs = self._make_perfect()
        result = compute_retrieval_metrics(
            docs, ["38300", "38401"], ["gnb", "node", "ran"], k=5
        )
        for key in [
            "hit_rate_at_k", "recall_at_k", "mrr", "ndcg_at_k",
            "context_precision", "context_recall_keywords", "k", "num_docs",
        ]:
            assert key in result, f"Missing key: {key}"

    def test_perfect_retrieval_scores(self):
        docs = self._make_perfect()
        result = compute_retrieval_metrics(
            docs, ["38300", "38401"], ["gnb", "node", "ran", "base station"], k=5
        )
        assert result["hit_rate_at_k"] == 1.0
        assert result["recall_at_k"] == 1.0
        assert result["mrr"] == 1.0
        assert result["ndcg_at_k"] == 1.0

    def test_empty_docs_all_zeros(self):
        result = compute_retrieval_metrics([], ["38300"], ["gnb"], k=5)
        assert result["hit_rate_at_k"] == 0.0
        assert result["recall_at_k"] == 0.0
        assert result["mrr"] == 0.0
        assert result["ndcg_at_k"] == 0.0

    def test_k_stored(self):
        docs = self._make_perfect()
        result = compute_retrieval_metrics(docs, ["38300"], ["gnb"], k=10)
        assert result["k"] == 10

    def test_num_docs_counted(self):
        docs = self._make_perfect()
        result = compute_retrieval_metrics(docs, ["38300"], ["gnb"])
        assert result["num_docs"] == len(docs)

    def test_all_values_in_range(self):
        docs = self._make_perfect()
        result = compute_retrieval_metrics(docs, ["38300"], ["gnb"])
        for key in ["hit_rate_at_k", "recall_at_k", "mrr", "ndcg_at_k",
                    "context_precision", "context_recall_keywords"]:
            assert 0.0 <= result[key] <= 1.0, f"{key}={result[key]} out of [0,1]"


# ---------------------------------------------------------------------------
# Regression helpers
# ---------------------------------------------------------------------------

class TestGetNested:
    def test_simple_path(self):
        d = {"retrieval": {"avg_mrr": 0.75}}
        assert _get_nested(d, "retrieval.avg_mrr") == 0.75

    def test_deep_path(self):
        d = {"latency": {"retrieve": {"p50": 0.32}}}
        assert _get_nested(d, "latency.retrieve.p50") == 0.32

    def test_missing_key(self):
        d = {"retrieval": {}}
        assert _get_nested(d, "retrieval.avg_mrr") is None

    def test_missing_top_level(self):
        assert _get_nested({}, "retrieval.avg_mrr") is None


class TestCompareToBaseline:
    def test_no_baseline_file_returns_empty(self, tmp_path):
        summary = {"retrieval": {"avg_mrr": 0.8}}
        # Point to a path that doesn't exist
        result = compare_to_baseline(summary, baseline_path=str(tmp_path / "nonexistent.json"))
        assert result == []

    def test_no_regression_detected(self, tmp_path):
        baseline_path = str(tmp_path / "baseline.json")
        summary = {
            "retrieval": {
                "avg_hit_rate_at_k": 0.9,
                "avg_recall_at_k":   0.8,
                "avg_mrr":           0.75,
                "avg_ndcg_at_k":     0.7,
                "avg_context_precision": 1.0,
                "avg_context_recall":    0.8,
            },
            "latency": {"retrieve": {"p50": 0.3, "p95": 0.4}},
        }
        save_baseline(summary, baseline_path=baseline_path)
        # Same summary → no regression
        result = compare_to_baseline(summary, baseline_path=baseline_path)
        assert result == []

    def test_regression_detected_on_mrr_drop(self, tmp_path):
        baseline_path = str(tmp_path / "baseline.json")
        good_summary = {
            "retrieval": {
                "avg_hit_rate_at_k": 0.9,
                "avg_recall_at_k":   0.8,
                "avg_mrr":           0.8,
                "avg_ndcg_at_k":     0.7,
                "avg_context_precision": 1.0,
                "avg_context_recall":    0.8,
            },
            "latency": {"retrieve": {"p50": 0.3, "p95": 0.4}},
        }
        save_baseline(good_summary, baseline_path=baseline_path)

        # MRR drops by 0.2 (exceeds tolerance of 0.05)
        bad_summary = {
            "retrieval": {
                "avg_hit_rate_at_k": 0.9,
                "avg_recall_at_k":   0.8,
                "avg_mrr":           0.6,   # <-- regression
                "avg_ndcg_at_k":     0.7,
                "avg_context_precision": 1.0,
                "avg_context_recall":    0.8,
            },
            "latency": {"retrieve": {"p50": 0.3, "p95": 0.4}},
        }
        result = compare_to_baseline(bad_summary, baseline_path=baseline_path)
        assert len(result) >= 1
        assert any("avg_mrr" in r for r in result)

    def test_latency_regression_detected(self, tmp_path):
        baseline_path = str(tmp_path / "baseline.json")
        fast_summary = {
            "retrieval": {
                "avg_hit_rate_at_k": 0.9, "avg_recall_at_k": 0.8,
                "avg_mrr": 0.8, "avg_ndcg_at_k": 0.7,
                "avg_context_precision": 1.0, "avg_context_recall": 0.8,
            },
            "latency": {"retrieve": {"p50": 0.3, "p95": 0.4}},
        }
        save_baseline(fast_summary, baseline_path=baseline_path)

        slow_summary = {
            "retrieval": {
                "avg_hit_rate_at_k": 0.9, "avg_recall_at_k": 0.8,
                "avg_mrr": 0.8, "avg_ndcg_at_k": 0.7,
                "avg_context_precision": 1.0, "avg_context_recall": 0.8,
            },
            "latency": {"retrieve": {"p50": 0.6, "p95": 0.9}},  # much slower
        }
        result = compare_to_baseline(slow_summary, baseline_path=baseline_path)
        assert any("latency.retrieve.p50" in r or "latency.retrieve.p95" in r for r in result)


class TestSaveLoadBaseline:
    def test_roundtrip(self, tmp_path):
        baseline_path = str(tmp_path / "baseline.json")
        summary = {
            "retrieval": {
                "avg_hit_rate_at_k": 0.9,
                "avg_recall_at_k":   0.8,
                "avg_mrr":           0.75,
                "avg_ndcg_at_k":     0.7,
                "avg_context_precision": 1.0,
                "avg_context_recall":    0.8,
            },
            "latency": {"retrieve": {"p50": 0.3, "p95": 0.4}},
        }
        save_baseline(summary, baseline_path=baseline_path)
        loaded = load_baseline(baseline_path=baseline_path)
        assert loaded is not None
        assert loaded["retrieval"]["avg_mrr"] == 0.75

    def test_load_nonexistent_returns_none(self, tmp_path):
        result = load_baseline(str(tmp_path / "missing.json"))
        assert result is None


# ---------------------------------------------------------------------------
# ROUND 2 ADDITIONS
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# _is_relevant — tightened word-boundary / spec-number-token matching
# ---------------------------------------------------------------------------

class TestIsRelevantTightened:
    """Tests for the tightened _is_relevant that uses word-boundary regex.

    The old implementation used plain substring matching (`rs in source`).
    The new implementation requires the spec token to NOT be flanked by digits,
    preventing partial-number false positives.
    """

    # --- Regression: original passing cases must still work ---

    def test_exact_spec_prefix_match(self):
        """38300 in "38300-g10.docx" — standard real-corpus filename."""
        assert _is_relevant({"source": "38300-g10.docx"}, ["38300"]) is True

    def test_spec_with_version_suffix(self):
        """38401 in "38401-h20.docx" — spec + version."""
        assert _is_relevant({"source": "38401-h20.docx"}, ["38401"]) is True

    def test_spec_standalone(self):
        """38300 in "38300.docx"."""
        assert _is_relevant({"source": "38300.docx"}, ["38300"]) is True

    def test_multiple_sources_second_matches(self):
        """Second entry in relevant_sources matches."""
        assert _is_relevant({"source": "38401-h10.docx"}, ["38300", "38401"]) is True

    # --- New: false-positive prevention ---

    def test_partial_number_not_matched(self):
        """Token "300" must NOT match "38300-g10.docx" (digits flanking on left)."""
        assert _is_relevant({"source": "38300-g10.docx"}, ["300"]) is False

    def test_leading_digit_overlap_not_matched(self):
        """Token "83" must NOT match "38300-g10.docx" — it is a sub-token."""
        assert _is_relevant({"source": "38300-g10.docx"}, ["83"]) is False

    def test_longer_number_not_matched_by_prefix(self):
        """Token "38300" must NOT match "383001-g10.docx" (digit on the right)."""
        assert _is_relevant({"source": "383001-g10.docx"}, ["38300"]) is False

    def test_different_series_not_confused(self):
        """36300 must NOT match 38300-g10.docx (different first digits)."""
        assert _is_relevant({"source": "38300-g10.docx"}, ["36300"]) is False

    def test_empty_source_returns_false(self):
        assert _is_relevant({"source": ""}, ["38300"]) is False

    def test_no_source_key_returns_false(self):
        assert _is_relevant({}, ["38300"]) is False

    def test_empty_relevant_sources_returns_false(self):
        assert _is_relevant({"source": "38300-g10.docx"}, []) is False

    def test_case_insensitive(self):
        """Source with uppercase spec chars should still match."""
        assert _is_relevant({"source": "TS38300-g10.DOCX"}, ["38300"]) is True


# ---------------------------------------------------------------------------
# ndcg_at_k — graded relevance mode (0/1/2 gains)
# ---------------------------------------------------------------------------

class TestNDCGGraded:
    """Tests for ndcg_at_k with graded_relevance argument."""

    def test_perfect_graded_ranking(self):
        """Grade-2 doc at rank 1, grade-1 at rank 2 — nDCG should be 1.0."""
        docs = [
            {"source": "38300-g10.docx", "text": "a"},   # grade 2
            {"source": "38401-h10.docx", "text": "b"},   # grade 1
            {"source": "other.docx", "text": "c"},       # grade 0
        ]
        graded = {"38300": 2, "38401": 1}
        score = ndcg_at_k(docs, ["38300", "38401"], k=5, graded_relevance=graded)
        assert score == 1.0

    def test_suboptimal_order_below_1(self):
        """Grade-1 at rank 1, grade-2 at rank 2 — nDCG should be < 1.0."""
        docs = [
            {"source": "38401-h10.docx", "text": "b"},   # grade 1 first
            {"source": "38300-g10.docx", "text": "a"},   # grade 2 second
            {"source": "other.docx", "text": "c"},
        ]
        graded = {"38300": 2, "38401": 1}
        score = ndcg_at_k(docs, ["38300", "38401"], k=5, graded_relevance=graded)
        assert 0.0 < score < 1.0

    def test_graded_score_lower_than_perfect(self):
        """Reversed order must score strictly lower than optimal order."""
        docs_optimal = [
            {"source": "38300-g10.docx", "text": "a"},
            {"source": "38401-h10.docx", "text": "b"},
        ]
        docs_reversed = [
            {"source": "38401-h10.docx", "text": "b"},
            {"source": "38300-g10.docx", "text": "a"},
        ]
        graded = {"38300": 2, "38401": 1}
        s_opt = ndcg_at_k(docs_optimal, ["38300", "38401"], k=5, graded_relevance=graded)
        s_rev = ndcg_at_k(docs_reversed, ["38300", "38401"], k=5, graded_relevance=graded)
        assert s_opt > s_rev

    def test_no_relevant_docs_graded(self):
        """No doc matches any graded spec — score is 0.0."""
        docs = [{"source": "other.docx", "text": "x"}]
        score = ndcg_at_k(docs, ["38300"], k=5, graded_relevance={"38300": 2})
        assert score == 0.0

    def test_graded_result_in_0_1_range(self):
        """Graded nDCG must always be in [0, 1]."""
        docs = [
            {"source": "38300-g10.docx", "text": "a"},
            {"source": "38300-g10.docx", "text": "b"},   # second chunk same spec
            {"source": "38401-h10.docx", "text": "c"},
            {"source": "other.docx", "text": "d"},
        ]
        graded = {"38300": 2, "38401": 1}
        score = ndcg_at_k(docs, ["38300", "38401"], k=5, graded_relevance=graded)
        assert 0.0 <= score <= 1.0, f"nDCG={score} outside [0,1]"

    def test_binary_mode_unchanged_when_no_graded(self):
        """Passing graded_relevance=None should give same result as omitting it."""
        docs = [
            {"source": "38300-g10.docx", "text": "a"},
            {"source": "other.docx", "text": "b"},
        ]
        s1 = ndcg_at_k(docs, ["38300"], k=5)
        s2 = ndcg_at_k(docs, ["38300"], k=5, graded_relevance=None)
        assert s1 == s2

    def test_multiple_chunks_same_spec_grade_2_perfect_score(self):
        """Two chunks from the same grade-2 spec at rank 1 and 2 is the ideal ranking."""
        docs = [
            {"source": "38300-g10.docx", "text": "a"},  # grade 2, rank 1
            {"source": "38300-g10.docx", "text": "b"},  # grade 2, rank 2
            {"source": "other.docx", "text": "c"},
        ]
        graded = {"38300": 2}
        score = ndcg_at_k(docs, ["38300"], k=5, graded_relevance=graded)
        assert score == 1.0

    def test_empty_docs_graded(self):
        assert ndcg_at_k([], ["38300"], k=5, graded_relevance={"38300": 2}) == 0.0

    def test_empty_relevant_sources_graded(self):
        docs = [{"source": "38300-g10.docx", "text": "a"}]
        assert ndcg_at_k(docs, [], k=5, graded_relevance={"38300": 2}) == 0.0

    def test_known_graded_value(self):
        """One grade-2 doc at rank 1, one grade-1 doc at rank 3.

        DCG  = 2/log2(2) + 0 + 1/log2(4) = 2.0 + 0.5 = 2.5
        IDCG = 2/log2(2) + 1/log2(3) = 2.0 + 0.6309 = 2.6309
        nDCG = 2.5 / 2.6309 ≈ 0.9502
        """
        docs = [
            {"source": "38300-g10.docx", "text": "a"},  # grade 2, rank 1
            {"source": "other.docx",    "text": "x"},  # grade 0, rank 2
            {"source": "38401-h10.docx", "text": "b"},  # grade 1, rank 3
        ]
        graded = {"38300": 2, "38401": 1}
        score = ndcg_at_k(docs, ["38300", "38401"], k=5, graded_relevance=graded)
        expected = 2.5 / (2.0 + 1.0 / math.log2(3))
        assert abs(score - expected) < 1e-3, f"got {score}, expected {expected:.4f}"


# ---------------------------------------------------------------------------
# is_refusal / refusal_rate — negative/out-of-corpus eval axis
# ---------------------------------------------------------------------------

class TestIsRefusal:
    """Tests for the heuristic refusal detector."""

    def test_explicit_no_info(self):
        assert is_refusal("I don't have information about that in the 3GPP specs.") is True

    def test_not_covered(self):
        assert is_refusal("This topic is not covered by any indexed specification.") is True

    def test_out_of_scope(self):
        assert is_refusal("That question is out of scope for this 3GPP assistant.") is True

    def test_cannot_find(self):
        assert is_refusal("I cannot find any relevant 3GPP specification for this query.") is True

    def test_no_relevant(self):
        assert is_refusal("There is no relevant information available in the indexed specs.") is True

    def test_normal_answer_not_refusal(self):
        answer = (
            "The gNB (next generation NodeB) provides NR radio access to the UE. "
            "According to TS 38.300, gNBs connect to 5GC via the NG interface."
        )
        assert is_refusal(answer) is False

    def test_empty_answer_not_refusal(self):
        """An empty answer is not detected as a refusal (no signal matched)."""
        assert is_refusal("") is False

    def test_case_insensitive(self):
        assert is_refusal("I DO NOT HAVE INFORMATION ABOUT THAT.") is True

    def test_partial_phrase_not_triggered(self):
        """'available' alone should not trigger (it's not in the phrase list)."""
        assert is_refusal("The feature is available in NR Release 15.") is False


class TestIsRefusalRealWorld:
    """Regression fixtures: real refusal openings from the 2026-07-02
    full-index run (data/eval_results.json, neg-001..005 answer_preview).
    The original phrase list detected 0/5 of these."""

    def test_neg001_unrelated_to_context(self):
        answer = (
            "I must point out that the user's question about the caloric content "
            "of a Big Mac is unrelated to the provided context excerpts from "
            "3GPP technical specification documents."
        )
        assert is_refusal(answer) is True

    def test_neg002_cannot_provide_unrelated(self):
        answer = (
            "I cannot provide an answer to your question about configuring "
            "Kubernetes pod networking with Calico, as it is unrelated to the "
            "provided context."
        )
        assert is_refusal(answer) is True

    def test_neg003_cannot_provide_based_on_context(self):
        answer = (
            "I cannot provide an answer based on the provided context excerpts. "
            "The context excerpts appear to be related to 3GPP technical specifications."
        )
        assert is_refusal(answer) is True

    def test_neg004_cannot_provide_realtime(self):
        answer = (
            "I cannot provide real-time financial information or stock prices. "
            "However, I can tell you that Ericsson is a company that provides "
            "telecommunications equipment."
        )
        assert is_refusal(answer) is True

    def test_neg005_cannot_provide_cisco(self):
        answer = (
            "I cannot provide an answer to your question about configuring a "
            "Cisco IOS BGP peer for MPLS VPN. The provided context excerpts are "
            "from 3GPP specifications."
        )
        assert is_refusal(answer) is True

    def test_signal_beyond_scan_window_not_triggered(self):
        """Signals appearing deep in a long technical answer (past the scan
        window) must not flag it: only the answer's opening is scanned."""
        filler = "The gNB provides NR radio access to connected UEs. " * 12
        assert len(filler) > 400
        answer = filler + " Values outside this range are not covered."
        assert is_refusal(answer) is False

    def test_refusal_within_scan_window_triggered(self):
        answer = "I cannot provide an answer to this. " + ("Details follow. " * 50)
        assert is_refusal(answer) is True


class TestRefusalRate:
    """Tests for the refusal_rate aggregation function."""

    def test_all_refusals(self):
        answers = [
            "I don't have information about that.",
            "This is not covered in 3GPP specs.",
            "I cannot find any relevant specification.",
        ]
        assert refusal_rate(answers) == 1.0

    def test_no_refusals(self):
        answers = [
            "The gNB connects to 5GC via NG interface.",
            "MAC performs HARQ retransmission.",
        ]
        assert refusal_rate(answers) == 0.0

    def test_partial_refusals(self):
        answers = [
            "I don't have information about that.",   # refusal
            "The gNB connects to 5GC via NG.",        # not refusal
            "No relevant information available.",     # refusal
            "PDCP performs header compression.",      # not refusal
        ]
        rate = refusal_rate(answers)
        assert rate == 0.5

    def test_empty_list(self):
        assert refusal_rate([]) == 0.0

    def test_result_in_range(self):
        answers = ["I cannot answer this.", "MAC schedules resources."]
        r = refusal_rate(answers)
        assert 0.0 <= r <= 1.0


# ---------------------------------------------------------------------------
# Similarity threshold calibration
# ---------------------------------------------------------------------------

class TestPassSimThreshold:
    """Per-embedding-model calibrated legacy-pass thresholds."""

    def test_bge_small_calibrated(self):
        """bge-small threshold from the 2026-07-02 calibration run."""
        assert pass_sim_threshold("bge-small") == 0.42

    def test_unknown_model_falls_back_to_default(self):
        assert pass_sim_threshold("some-future-model") == DEFAULT_PASS_SIM_THRESHOLD

    def test_default_is_historical_value(self):
        assert DEFAULT_PASS_SIM_THRESHOLD == 0.50

    def test_refusal_threshold_is_separate_dial(self):
        """The OOC refusal boundary stays at 0.50 regardless of calibration."""
        assert REFUSAL_SIM_THRESHOLD == 0.50
        assert REFUSAL_SIM_THRESHOLD != PASS_SIM_THRESHOLDS["bge-small"]


class TestThresholdCalibration:
    """Youden-J calibration in scripts/eval/calibrate_threshold.py."""

    def test_separable_populations_perfect_j(self):
        result = youden_optimal([0.5, 0.6, 0.7], [0.1, 0.2])
        assert result["j"] == 1.0
        assert 0.2 < result["threshold"] <= 0.5

    def test_overlapping_populations_best_tradeoff(self):
        # negative 0.48 sits between positives 0.45 and 0.50: best J drops
        # the low positive rather than admit the negative.
        result = youden_optimal([0.45, 0.5], [0.48])
        assert result["threshold"] == pytest.approx(0.49)
        assert result["tpr"] == 0.5
        assert result["fpr"] == 0.0

    def test_requires_both_populations(self):
        with pytest.raises(ValueError):
            youden_optimal([], [0.3])
        with pytest.raises(ValueError):
            youden_optimal([0.5], [])

    def test_extract_populations_from_artifact_shape(self):
        results = {
            "cases": [
                {"avg_similarity": 0.6, "hit_rate_at_k": 1.0, "is_out_of_corpus": False},
                {"avg_similarity": 0.5, "hit_rate_at_k": 1.0, "is_out_of_corpus": False},
                # in-corpus retrieval MISS: excluded from positives
                {"avg_similarity": 0.55, "hit_rate_at_k": 0.0, "is_out_of_corpus": False},
                {"avg_similarity": 0.2, "hit_rate_at_k": 0.0, "is_out_of_corpus": True},
            ]
        }
        positives, negatives = extract_populations(results)
        assert positives == [0.6, 0.5]
        assert negatives == [0.2]

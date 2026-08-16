"""
Tests for the evaluation pipeline (scripts/eval_retrieval.py) and GET /eval endpoint.

Covers:
  - context_precision / context_recall helpers
  - answer_relevance / faithfulness helpers
  - evaluate_answer_quality composite scoring
  - compute_latency_benchmarks
  - GET /eval endpoint (no results, with results)
"""
import json
import sys
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Make sure project root is importable when running directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.eval_retrieval import (
    _context_precision,
    _context_recall,
    _answer_relevance,
    _faithfulness,
    evaluate_answer_quality,
    compute_latency_benchmarks,
    _percentile,
    evaluate_retrieval_case,
    TEST_CASES,
)


# ---------------------------------------------------------------------------
# Helpers: _context_precision
# ---------------------------------------------------------------------------

class TestContextPrecision:
    def test_all_relevant(self):
        docs = [
            {"source": "38300-g10.docx", "text": "...", "similarity": 0.9},
            {"source": "38300-g20.docx", "text": "...", "similarity": 0.8},
        ]
        score = _context_precision(docs, ["38300"])
        assert score == 1.0

    def test_none_relevant(self):
        docs = [
            {"source": "other.docx", "text": "...", "similarity": 0.9},
        ]
        score = _context_precision(docs, ["38300"])
        assert score == 0.0

    def test_partial_relevant(self):
        docs = [
            {"source": "38300-g10.docx", "text": "...", "similarity": 0.9},
            {"source": "other.docx", "text": "...", "similarity": 0.8},
        ]
        score = _context_precision(docs, ["38300"])
        assert score == 0.5

    def test_empty_docs(self):
        assert _context_precision([], ["38300"]) == 0.0

    def test_empty_relevant_sources(self):
        docs = [{"source": "38300-g10.docx", "text": "...", "similarity": 0.9}]
        # No relevant_sources means no hits
        assert _context_precision(docs, []) == 0.0

    def test_multiple_relevant_source_patterns(self):
        docs = [
            {"source": "38300-g10.docx", "text": "...", "similarity": 0.9},
            {"source": "38401-h10.docx", "text": "...", "similarity": 0.8},
            {"source": "other.docx", "text": "...", "similarity": 0.7},
        ]
        score = _context_precision(docs, ["38300", "38401"])
        assert round(score, 3) == round(2 / 3, 3)


# ---------------------------------------------------------------------------
# Helpers: _context_recall
# ---------------------------------------------------------------------------

class TestContextRecall:
    def test_all_keywords_found(self):
        docs = [{"text": "The gnb base station node ran architecture", "source": "x", "similarity": 0.9}]
        score = _context_recall(docs, ["gnb", "base station", "node", "ran"])
        assert score == 1.0

    def test_no_keywords_found(self):
        docs = [{"text": "Completely unrelated content here", "source": "x", "similarity": 0.9}]
        score = _context_recall(docs, ["gnb", "base station"])
        assert score == 0.0

    def test_partial_keywords(self):
        docs = [{"text": "The gnb architecture", "source": "x", "similarity": 0.9}]
        score = _context_recall(docs, ["gnb", "base station", "node", "ran"])
        assert score == 0.25

    def test_empty_docs(self):
        assert _context_recall([], ["gnb"]) == 0.0

    def test_empty_keywords(self):
        docs = [{"text": "some content", "source": "x", "similarity": 0.9}]
        assert _context_recall(docs, []) == 0.0

    def test_uses_top_3_docs_only(self):
        # Keyword only in 4th doc — should not be found
        docs = [
            {"text": "doc one content", "source": "a", "similarity": 0.9},
            {"text": "doc two content", "source": "b", "similarity": 0.8},
            {"text": "doc three content", "source": "c", "similarity": 0.7},
            {"text": "gnb is here", "source": "d", "similarity": 0.6},
        ]
        score = _context_recall(docs, ["gnb"])
        assert score == 0.0


# ---------------------------------------------------------------------------
# Helpers: _answer_relevance
# ---------------------------------------------------------------------------

class TestAnswerRelevance:
    def test_all_keywords_in_answer(self):
        case = {"answer_keywords": ["gnb", "base station", "radio"]}
        answer = "The gNB (base station) handles radio access."
        score = _answer_relevance(answer, case)
        assert score == 1.0

    def test_no_keywords_in_answer(self):
        case = {"answer_keywords": ["gnb", "base station", "radio"]}
        answer = "This is completely unrelated."
        score = _answer_relevance(answer, case)
        assert score == 0.0

    def test_case_insensitive(self):
        case = {"answer_keywords": ["GNB", "BASE STATION"]}
        answer = "The gnb is a base station."
        score = _answer_relevance(answer, case)
        assert score == 1.0

    def test_empty_keywords(self):
        case = {"answer_keywords": [], "expected_keywords": []}
        assert _answer_relevance("any answer", case) == 0.0

    def test_falls_back_to_expected_keywords(self):
        """Golden-set cases define only expected_keywords; without the
        fallback every golden-set case scored 0.0 (2026-07-02 run)."""
        case = {"expected_keywords": ["gnb", "radio"]}
        answer = "The gNB handles radio access."
        assert _answer_relevance(answer, case) == 1.0

    def test_answer_keywords_take_precedence(self):
        case = {
            "answer_keywords": ["handover"],
            "expected_keywords": ["gnb", "radio"],
        }
        answer = "The gNB handles radio access."  # matches expected, not answer_keywords
        assert _answer_relevance(answer, case) == 0.0

    def test_no_keyword_fields_scores_zero(self):
        assert _answer_relevance("any answer", {}) == 0.0


# ---------------------------------------------------------------------------
# Helpers: _faithfulness
# ---------------------------------------------------------------------------

class TestFaithfulness:
    def test_grounded_answer_scores_higher(self):
        docs = [{"text": "The gNB is a base station node in the 3GPP specification.", "source": "x", "similarity": 0.9}]
        grounded = "According to the 3GPP TS specification, the gNB base station node handles radio."
        ungrounded = "Pizza is delicious and everyone loves it very much indeed."
        assert _faithfulness(grounded, docs) > _faithfulness(ungrounded, docs)

    def test_empty_docs_returns_zero(self):
        assert _faithfulness("some answer", []) == 0.0

    def test_score_in_range(self):
        docs = [{"text": "The gNB is a base station.", "source": "x", "similarity": 0.9}]
        score = _faithfulness("The gNB base station is described in 3GPP.", docs)
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# evaluate_answer_quality
# ---------------------------------------------------------------------------

class TestEvaluateAnswerQuality:
    def _make_docs(self):
        return [{"text": "The gNB is a base station node ran.", "source": "38300-g10.docx", "similarity": 0.9}]

    def test_good_answer_passes(self):
        case = TEST_CASES[0]  # "What is gNB?"
        docs = self._make_docs()
        answer = (
            "According to 3GPP specification, the gNB (base station) is a radio access "
            "node that provides NR radio access to user equipment."
        )
        result = evaluate_answer_quality(answer, case, docs)
        assert result["answer_pass"] is True
        assert result["answer_score"] > 0.40

    def test_empty_answer_fails(self):
        case = TEST_CASES[0]
        result = evaluate_answer_quality("", case, self._make_docs())
        assert result["is_substantive"] is False
        assert result["answer_score"] < 0.40

    def test_refusal_detected(self):
        case = TEST_CASES[0]
        answer = "I cannot find relevant information to answer this question."
        result = evaluate_answer_quality(answer, case, self._make_docs())
        assert result["is_refusal"] is True

    def test_result_has_required_keys(self):
        case = TEST_CASES[0]
        result = evaluate_answer_quality("Some answer text here.", case, self._make_docs())
        for key in ["answer_relevance", "faithfulness", "answer_score",
                    "is_substantive", "is_refusal", "answer_pass", "answer_preview"]:
            assert key in result

    def test_score_bounded_0_to_1(self):
        case = TEST_CASES[0]
        result = evaluate_answer_quality("test answer", case, self._make_docs())
        assert 0.0 <= result["answer_score"] <= 1.0


# ---------------------------------------------------------------------------
# compute_latency_benchmarks
# ---------------------------------------------------------------------------

class TestLatencyBenchmarks:
    def test_retrieve_only(self):
        results = [
            {"retrieve_time": 0.1},
            {"retrieve_time": 0.2},
            {"retrieve_time": 0.3},
        ]
        b = compute_latency_benchmarks(results)
        assert "retrieve" in b
        # p50 of 3 items: idx = max(0, int(3*50/100)-1) = max(0,0) = 0 → 0.1
        assert b["retrieve"]["p50"] == 0.1
        # p95 of 3 items: idx = max(0, int(3*95/100)-1) = max(0,1) = 1 → 0.2
        assert b["retrieve"]["p95"] == 0.2
        assert "generate" not in b

    def test_with_generate_times(self):
        results = [
            {"retrieve_time": 0.1, "generate_time": 1.0},
            {"retrieve_time": 0.2, "generate_time": 2.0},
            {"retrieve_time": 0.3, "generate_time": 3.0},
        ]
        b = compute_latency_benchmarks(results)
        assert "generate" in b
        assert "total" in b
        assert b["total"]["p50"] > b["retrieve"]["p50"]

    def test_empty_results(self):
        b = compute_latency_benchmarks([])
        assert b["retrieve"]["p50"] == 0.0


class TestPercentile:
    def test_p50_odd_list(self):
        # idx = max(0, int(5*50/100)-1) = max(0,1) = 1 → 2.0
        assert _percentile([1.0, 2.0, 3.0, 4.0, 5.0], 50) == 2.0

    def test_p95_small_list(self):
        vals = [0.1, 0.2, 0.3, 0.4, 0.5]
        # p95 of 5 items: idx = max(0, int(5*95/100)-1) = max(0,3) = 3 → 0.4
        assert _percentile(vals, 95) == 0.4

    def test_empty_list(self):
        assert _percentile([], 50) == 0.0


# ---------------------------------------------------------------------------
# evaluate_retrieval_case (unit-level, mock retriever)
# ---------------------------------------------------------------------------

class TestEvaluateRetrievalCase:
    def _make_retriever(self, docs):
        r = MagicMock()
        r.retrieve.return_value = docs
        return r

    def test_pass_with_good_results(self):
        docs = [
            {"source": "38300-g10.docx", "text": "gnb base station node ran", "similarity": 0.8},
            {"source": "38300-g20.docx", "text": "gnb architecture", "similarity": 0.7},
        ]
        retriever = self._make_retriever(docs)
        result = evaluate_retrieval_case(retriever, TEST_CASES[0])
        assert result["retrieval_pass"] is True
        assert result["context_recall"] > 0.5

    def test_fail_with_empty_results(self):
        retriever = self._make_retriever([])
        result = evaluate_retrieval_case(retriever, TEST_CASES[0])
        assert result["retrieval_pass"] is False
        assert result["avg_similarity"] == 0.0

    def test_fail_with_low_similarity(self):
        docs = [
            {"source": "38300-g10.docx", "text": "gnb base station node ran", "similarity": 0.1},
        ]
        retriever = self._make_retriever(docs)
        result = evaluate_retrieval_case(retriever, TEST_CASES[0])
        assert result["retrieval_pass"] is False

    def test_result_has_required_keys(self):
        docs = [{"source": "38300-g10.docx", "text": "gnb", "similarity": 0.8}]
        retriever = self._make_retriever(docs)
        result = evaluate_retrieval_case(retriever, TEST_CASES[0])
        for key in ["query", "retrieval_pass", "context_precision", "context_recall",
                    "avg_similarity", "retrieve_time", "top_source"]:
            assert key in result

    def test_retrieve_time_is_positive(self):
        docs = [{"source": "38300-g10.docx", "text": "gnb", "similarity": 0.8}]
        retriever = self._make_retriever(docs)
        result = evaluate_retrieval_case(retriever, TEST_CASES[0])
        assert result["retrieve_time"] >= 0.0


# ---------------------------------------------------------------------------
# GET /eval API endpoint
# ---------------------------------------------------------------------------

class TestEvalEndpoint:
    @pytest.fixture(autouse=True)
    def mock_app_state(self):
        from src.api import main as api_main
        mock_vs = MagicMock()
        mock_vs.get_stats.return_value = {"collection_name": "3gpp_specs", "total_chunks": 250, "persist_directory": "data/vectordb"}
        mock_metrics = MagicMock()
        mock_metrics.summary.return_value = {"total_queries": 0}
        api_main.app_state.vector_store = mock_vs
        api_main.app_state.metrics = mock_metrics
        api_main.app_state.ready = True
        api_main.app_state.sessions = {}
        yield
        api_main.app_state.sessions = {}

    @pytest.fixture
    def client(self):
        with patch("src.api.main.RAGChain"), \
             patch("src.api.main.DocumentRetriever"), \
             patch("src.api.main.OllamaLLM"):
            from src.api.main import app
            from fastapi.testclient import TestClient
            with TestClient(app, raise_server_exceptions=True) as c:
                yield c

    def test_eval_returns_200(self, client, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        response = client.get("/eval")
        assert response.status_code == 200

    def test_eval_unavailable_when_no_file(self, client, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        data = client.get("/eval").json()
        assert data["available"] is False

    def test_eval_available_with_results_file(self, client, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        # Create a fake eval_results.json
        eval_dir = tmp_path / "data"
        eval_dir.mkdir()
        eval_data = {
            "evaluated_at": "2026-02-19T00:00:00Z",
            "config": {"top_k": 5, "full_eval": False, "total_chunks": 43121},
            "summary": {
                "retrieval": {"pass_rate": 0.8, "passed": 8, "total": 10},
                "answer": {},
                "latency": {"retrieve": {"p50": 0.25, "p95": 0.5, "mean": 0.3}},
            },
            "cases": [],
        }
        (eval_dir / "eval_results.json").write_text(json.dumps(eval_data))

        data = client.get("/eval").json()
        assert data["available"] is True
        assert data["evaluated_at"] == "2026-02-19T00:00:00Z"
        assert data["summary"]["retrieval"]["pass_rate"] == 0.8
        assert data["config"]["total_chunks"] == 43121

    def test_eval_summary_has_retrieval_key(self, client, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        eval_dir = tmp_path / "data"
        eval_dir.mkdir()
        eval_data = {
            "evaluated_at": "2026-02-19T00:00:00Z",
            "config": {},
            "summary": {"retrieval": {"pass_rate": 0.9}},
            "cases": [],
        }
        (eval_dir / "eval_results.json").write_text(json.dumps(eval_data))

        data = client.get("/eval").json()
        assert "retrieval" in data["summary"]

    def test_eval_in_root_response(self, client, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        data = client.get("/").json()
        assert "eval" in data

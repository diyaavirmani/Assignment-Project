"""
Tests for src/core/rag_chain.py

All external dependencies (retriever, LLM) are mocked so tests run without
a live vector store or Ollama server.
"""

import pytest
from unittest.mock import MagicMock, patch

from src.core.answer_verifier import (
    REASON_INVALID_CITATION,
    REASON_UNSUPPORTED,
    REASON_VERIFICATION_ERROR,
    REASON_VERIFIED,
    VerificationResult,
)
from src.core.evidence_gate import EvidenceGate, REASON_SUFFICIENT, REASON_TOP_SCORE_LOW


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_DOCS = [
    {
        "text": "The gNB-CU is a logical node that handles RRC and PDCP protocols.",
        "source": "38300-g30.docx",
        "chunk_index": 0,
        "similarity": 0.92,
        "vector_similarity": 0.92,
        "reranker_score": 0.92,
    },
    {
        "text": "The gNB-DU is connected to the gNB-CU via the F1 interface.",
        "source": "38401-g30.docx",
        "chunk_index": 1,
        "similarity": 0.87,
        "vector_similarity": 0.87,
        "reranker_score": 0.87,
    },
]


def _make_chain(docs=None, llm_answer="Test answer from LLM."):
    """Build a RAGChain with mocked retriever and LLM."""
    from src.core.rag_chain import RAGChain

    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = docs if docs is not None else SAMPLE_DOCS
    mock_retriever.format_context.return_value = "\n".join(d["text"] for d in (docs or SAMPLE_DOCS))

    mock_llm = MagicMock()
    mock_llm.model = "llama3.2"
    mock_llm.generate.return_value = llm_answer

    mock_verifier = MagicMock()
    mock_verifier.enabled = True
    mock_verifier.verify.return_value = VerificationResult(
        passed=True,
        reason=REASON_VERIFIED,
        total_claims=1,
        supported_claims=1,
        citation_valid=True,
        cited_sources=["S1"],
    )

    chain = RAGChain(
        retriever=mock_retriever,
        llm=mock_llm,
        answer_verifier=mock_verifier,
    )
    return chain, mock_retriever, mock_llm, mock_verifier


# ---------------------------------------------------------------------------
# RAGChain.query tests
# ---------------------------------------------------------------------------


class TestRAGChainQuery:
    def test_query_returns_required_keys(self):
        chain, _, _, _ = _make_chain()
        result = chain.query("What is gNB-CU?")
        for key in (
            "answer",
            "sources",
            "context",
            "query_time",
            "retrieve_time",
            "generate_time",
            "evidence",
            "verification",
            "answer_status",
        ):
            assert key in result

    def test_query_answer_from_llm(self):
        chain, _, _, _ = _make_chain(llm_answer="The gNB-CU handles PDCP [S1].")
        result = chain.query("Explain gNB-CU")
        assert result["answer"] == "The gNB-CU handles PDCP [S1]."

    def test_query_sources_structure(self):
        chain, _, _, _ = _make_chain()
        result = chain.query("gNB architecture")
        assert len(result["sources"]) == 2
        for i, s in enumerate(result["sources"], 1):
            assert s["source_id"] == f"S{i}"
            assert "source" in s
            assert "similarity" in s
            assert "text" in s

    def test_query_passes_source_filter(self):
        chain, mock_retriever, _, _ = _make_chain()
        chain.query("gNB", source_filter="38300")
        mock_retriever.retrieve.assert_called_once_with(
            "gNB",
            top_k=None,
            source_filter="38300",
            domain=None,
            generation=None,
        )

    def test_query_passes_top_k_override(self):
        chain, mock_retriever, _, _ = _make_chain()
        chain.query("gNB", top_k=3)
        mock_retriever.retrieve.assert_called_once_with(
            "gNB",
            top_k=3,
            source_filter=None,
            domain=None,
            generation=None,
        )

    def test_query_empty_result_when_no_docs(self):
        chain, _, _, _ = _make_chain(docs=[])
        result = chain.query("unknown topic")
        assert result["sources"] == []
        assert "could not find" in result["answer"].lower()
        assert result["evidence"]["reason"] == "no_documents"

    def test_query_times_are_non_negative(self):
        chain, _, _, _ = _make_chain()
        result = chain.query("timing test")
        assert result["query_time"] >= 0
        assert result["retrieve_time"] >= 0
        assert result["generate_time"] >= 0

    def test_weak_retrieval_refuses_without_llm_generate(self):
        weak_docs = [
            {
                "text": "Weak unrelated chunk",
                "source": "38300-g30.docx",
                "chunk_index": 0,
                "similarity": 0.2,
                "vector_similarity": 0.2,
                "reranker_score": 0.2,
            }
        ]
        chain, _, mock_llm, mock_verifier = _make_chain(docs=weak_docs)

        result = chain.query("Who won the FIFA World Cup?")

        mock_llm.generate.assert_not_called()
        mock_verifier.verify.assert_not_called()
        assert "sufficient supporting evidence" in result["answer"]
        assert result["generate_time"] == 0.0
        assert result["evidence"]["sufficient"] is False
        assert result["evidence"]["reason"] == REASON_TOP_SCORE_LOW

    def test_strong_retrieval_calls_llm_generate_once(self):
        chain, _, mock_llm, mock_verifier = _make_chain(llm_answer="Generated answer [S1]")

        result = chain.query("What is gNB-CU?")

        mock_llm.generate.assert_called_once()
        mock_verifier.verify.assert_called_once()
        assert result["answer"] == "Generated answer [S1]"
        assert result["evidence"]["sufficient"] is True
        assert result["evidence"]["reason"] == REASON_SUFFICIENT
        assert result["verification"]["passed"] is True
        assert result["answer_status"] == "answered"

    def test_generation_prompt_requires_narrow_cited_answers(self):
        chain, _, mock_llm, _ = _make_chain(
            llm_answer="The F1 interface connects gNB-CU and gNB-DU [S2]."
        )

        chain.query("What is the F1 interface used for?")

        prompt = mock_llm.generate.call_args.args[0]
        assert "Answer the user's exact question with the narrowest sufficient cited answer." in prompt
        assert "Do not add protocol split" in prompt
        assert "unless the supplied evidence explicitly supports them" in prompt

    def test_strong_retrieval_verification_failure_hides_generated_answer(self):
        chain, _, mock_llm, mock_verifier = _make_chain(
            llm_answer="The gNB-CU performs unsupported billing [S1]."
        )
        mock_verifier.verify.return_value = VerificationResult(
            passed=False,
            reason=REASON_UNSUPPORTED,
            total_claims=1,
            supported_claims=0,
            unsupported_claims=["The gNB-CU performs unsupported billing."],
            citation_valid=True,
            cited_sources=["S1"],
        )

        result = chain.query("What does gNB-CU do?")

        mock_llm.generate.assert_called_once()
        assert "unsupported billing" not in result["answer"]
        assert "could not verify" in result["answer"]
        assert result["answer_status"] == "refused_verification"
        assert result["verification"]["reason"] == REASON_UNSUPPORTED

    def test_verification_error_hides_generated_answer(self):
        chain, _, _, mock_verifier = _make_chain(llm_answer="Internal generated answer [S1].")
        mock_verifier.verify.return_value = VerificationResult(
            passed=False,
            reason=REASON_VERIFICATION_ERROR,
            citation_valid=True,
            cited_sources=["S1"],
        )

        result = chain.query("What does gNB-CU do?")

        assert "Internal generated answer" not in result["answer"]
        assert result["answer_status"] == "verifier_error"

    def test_invalid_citation_hides_generated_answer(self):
        chain, _, _, mock_verifier = _make_chain(llm_answer="The gNB-CU hosts RRC [S99].")
        mock_verifier.verify.return_value = VerificationResult(
            passed=False,
            reason=REASON_INVALID_CITATION,
            citation_valid=False,
            cited_sources=["S99"],
            invalid_citations=["S99"],
        )

        result = chain.query("What does gNB-CU do?")

        assert "[S99]" not in result["answer"]
        assert result["answer_status"] == "refused_verification"

    def test_gate_disabled_preserves_generation_path_for_weak_docs(self):
        weak_docs = [
            {
                "text": "Weak chunk",
                "source": "38300-g30.docx",
                "chunk_index": 0,
                "similarity": 0.0,
                "vector_similarity": 0.0,
                "reranker_score": 0.0,
            }
        ]
        from src.core.rag_chain import RAGChain

        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = weak_docs
        mock_retriever.format_context.return_value = "Weak chunk"
        mock_llm = MagicMock()
        mock_llm.model = "llama3.2"
        mock_llm.generate.return_value = "Generated despite weak evidence"
        mock_verifier = MagicMock()
        mock_verifier.enabled = True
        mock_verifier.verify.return_value = VerificationResult(
            passed=True,
            reason=REASON_VERIFIED,
            total_claims=1,
            supported_claims=1,
            citation_valid=True,
            cited_sources=["S1"],
        )
        chain = RAGChain(
            retriever=mock_retriever,
            llm=mock_llm,
            evidence_gate=EvidenceGate(enabled=False),
            answer_verifier=mock_verifier,
        )

        result = chain.query("weak query")

        mock_llm.generate.assert_called_once()
        assert result["answer"] == "Generated despite weak evidence"


# ---------------------------------------------------------------------------
# RAGChain conversation history tests
# ---------------------------------------------------------------------------


class TestRAGChainHistory:
    def test_history_accumulates_across_queries(self):
        chain, _, mock_llm, _ = _make_chain()
        chain.query("First question")
        chain.query("Second question")
        # After 2 queries the history should have 4 messages (2 user + 2 assistant)
        assert len(chain.get_history()) == 4

    def test_clear_history_resets(self):
        chain, _, _, _ = _make_chain()
        chain.query("First question")
        chain.clear_history()
        assert chain.get_history() == []

    def test_history_sent_to_llm(self):
        chain, _, mock_llm, _ = _make_chain()
        chain.query("First question")
        chain.query("Follow-up question")
        second_call_kwargs = mock_llm.generate.call_args_list[1][1]
        history = second_call_kwargs.get("history", [])
        assert len(history) >= 2  # at least the first turn


# ---------------------------------------------------------------------------
# RAGChain.stream_query tests
# ---------------------------------------------------------------------------


class TestRAGChainStream:
    def test_stream_yields_sources_first(self):
        chain, _, mock_llm, _ = _make_chain()
        mock_llm.stream.return_value = iter(["Answer ", "here."])

        chunks = list(chain.stream_query("gNB?"))
        first = chunks[0]
        assert first["type"] == "sources"
        assert isinstance(first["sources"], list)

    def test_stream_yields_tokens(self):
        chain, _, mock_llm, _ = _make_chain()
        mock_llm.stream.return_value = iter(["Hello", " world"])

        tokens = [c["token"] for c in chain.stream_query("hi") if c["type"] == "token"]
        assert tokens == ["Hello", " world"]

    def test_stream_ends_with_done(self):
        chain, _, mock_llm, _ = _make_chain()
        mock_llm.stream.return_value = iter(["ok"])

        chunks = list(chain.stream_query("test"))
        last = chunks[-1]
        assert last["type"] == "done"
        assert "query_time" in last

    def test_stream_empty_docs(self):
        chain, _, _, _ = _make_chain(docs=[])
        chunks = list(chain.stream_query("nothing"))
        types = [c["type"] for c in chunks]
        assert types == ["sources", "token", "done"]

    def test_stream_weak_retrieval_refuses_without_llm_stream(self):
        weak_docs = [
            {
                "text": "Weak unrelated chunk",
                "source": "38300-g30.docx",
                "chunk_index": 0,
                "similarity": 0.1,
                "vector_similarity": 0.1,
                "reranker_score": 0.1,
            }
        ]
        chain, _, mock_llm, mock_verifier = _make_chain(docs=weak_docs)

        chunks = list(chain.stream_query("Who won the FIFA World Cup?"))

        mock_llm.stream.assert_not_called()
        mock_verifier.verify.assert_not_called()
        assert [chunk["type"] for chunk in chunks] == ["sources", "token", "done"]
        assert "sufficient supporting evidence" in chunks[1]["token"]
        assert chunks[0]["evidence"]["sufficient"] is False

    def test_stream_failed_verification_never_emits_generated_text(self):
        chain, _, mock_llm, mock_verifier = _make_chain()
        mock_llm.stream.return_value = iter(["Unsupported ", "telecom claim [S1]"])
        mock_verifier.verify.return_value = VerificationResult(
            passed=False,
            reason=REASON_UNSUPPORTED,
            total_claims=1,
            unsupported_claims=["Unsupported telecom claim."],
            citation_valid=True,
            cited_sources=["S1"],
        )

        chunks = list(chain.stream_query("gNB?"))
        emitted = "".join(chunk["token"] for chunk in chunks if chunk["type"] == "token")

        assert "Unsupported telecom claim" not in emitted
        assert "could not verify" in emitted
        assert chunks[-1]["answer_status"] == "refused_verification"

    def test_stream_verified_answer_emitted_after_buffering(self):
        chain, _, mock_llm, mock_verifier = _make_chain()
        mock_llm.stream.return_value = iter(["Supported ", "answer [S1]"])

        chunks = list(chain.stream_query("gNB?"))
        tokens = [chunk["token"] for chunk in chunks if chunk["type"] == "token"]

        mock_verifier.verify.assert_called_once()
        assert tokens == ["Supported ", "answer [S1]"]
        assert chunks[-1]["answer_status"] == "answered"

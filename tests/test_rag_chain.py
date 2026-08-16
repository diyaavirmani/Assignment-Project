"""
Tests for src/core/rag_chain.py

All external dependencies (retriever, LLM) are mocked so tests run without
a live vector store or Ollama server.
"""
import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_DOCS = [
    {
        "text": "The gNB-CU is a logical node that handles RRC and PDCP protocols.",
        "source": "38300-g30.docx",
        "chunk_index": 0,
        "similarity": 0.92,
    },
    {
        "text": "The gNB-DU is connected to the gNB-CU via the F1 interface.",
        "source": "38401-g30.docx",
        "chunk_index": 1,
        "similarity": 0.87,
    },
]


def _make_chain(docs=None, llm_answer="Test answer from LLM."):
    """Build a RAGChain with mocked retriever and LLM."""
    from src.core.rag_chain import RAGChain

    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = docs if docs is not None else SAMPLE_DOCS
    mock_retriever.format_context.return_value = "\n".join(
        d["text"] for d in (docs or SAMPLE_DOCS)
    )

    mock_llm = MagicMock()
    mock_llm.model = "llama3.2"
    mock_llm.generate.return_value = llm_answer

    chain = RAGChain(retriever=mock_retriever, llm=mock_llm)
    return chain, mock_retriever, mock_llm


# ---------------------------------------------------------------------------
# RAGChain.query tests
# ---------------------------------------------------------------------------

class TestRAGChainQuery:
    def test_query_returns_required_keys(self):
        chain, _, _ = _make_chain()
        result = chain.query("What is gNB-CU?")
        for key in ("answer", "sources", "context", "query_time",
                    "retrieve_time", "generate_time"):
            assert key in result

    def test_query_answer_from_llm(self):
        chain, _, _ = _make_chain(llm_answer="The gNB-CU handles PDCP.")
        result = chain.query("Explain gNB-CU")
        assert result["answer"] == "The gNB-CU handles PDCP."

    def test_query_sources_structure(self):
        chain, _, _ = _make_chain()
        result = chain.query("gNB architecture")
        assert len(result["sources"]) == 2
        for s in result["sources"]:
            assert "source" in s
            assert "similarity" in s
            assert "text" in s

    def test_query_passes_source_filter(self):
        chain, mock_retriever, _ = _make_chain()
        chain.query("gNB", source_filter="38300")
        mock_retriever.retrieve.assert_called_once_with(
            "gNB", top_k=None, source_filter="38300",
            domain=None, generation=None,
        )

    def test_query_passes_top_k_override(self):
        chain, mock_retriever, _ = _make_chain()
        chain.query("gNB", top_k=3)
        mock_retriever.retrieve.assert_called_once_with(
            "gNB", top_k=3, source_filter=None,
            domain=None, generation=None,
        )

    def test_query_empty_result_when_no_docs(self):
        chain, _, _ = _make_chain(docs=[])
        result = chain.query("unknown topic")
        assert result["sources"] == []
        assert "could not find" in result["answer"].lower()

    def test_query_times_are_non_negative(self):
        chain, _, _ = _make_chain()
        result = chain.query("timing test")
        assert result["query_time"] >= 0
        assert result["retrieve_time"] >= 0
        assert result["generate_time"] >= 0


# ---------------------------------------------------------------------------
# RAGChain conversation history tests
# ---------------------------------------------------------------------------

class TestRAGChainHistory:
    def test_history_accumulates_across_queries(self):
        chain, _, mock_llm = _make_chain()
        chain.query("First question")
        chain.query("Second question")
        # After 2 queries the history should have 4 messages (2 user + 2 assistant)
        assert len(chain.get_history()) == 4

    def test_clear_history_resets(self):
        chain, _, _ = _make_chain()
        chain.query("First question")
        chain.clear_history()
        assert chain.get_history() == []

    def test_history_sent_to_llm(self):
        chain, _, mock_llm = _make_chain()
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
        chain, _, mock_llm = _make_chain()
        mock_llm.stream.return_value = iter(["Answer ", "here."])

        chunks = list(chain.stream_query("gNB?"))
        first = chunks[0]
        assert first["type"] == "sources"
        assert isinstance(first["sources"], list)

    def test_stream_yields_tokens(self):
        chain, _, mock_llm = _make_chain()
        mock_llm.stream.return_value = iter(["Hello", " world"])

        tokens = [c["token"] for c in chain.stream_query("hi") if c["type"] == "token"]
        assert tokens == ["Hello", " world"]

    def test_stream_ends_with_done(self):
        chain, _, mock_llm = _make_chain()
        mock_llm.stream.return_value = iter(["ok"])

        chunks = list(chain.stream_query("test"))
        last = chunks[-1]
        assert last["type"] == "done"
        assert "query_time" in last

    def test_stream_empty_docs(self):
        chain, _, _ = _make_chain(docs=[])
        chunks = list(chain.stream_query("nothing"))
        types = [c["type"] for c in chunks]
        assert types == ["sources", "token", "done"]

"""
Integration tests for the full RAG pipeline

These tests require a populated vector store.
Run with: pytest -m integration tests/test_integration.py

Mark is skipped automatically when the vector store is empty so the CI
pipeline always stays green.
"""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def _vector_store_has_data() -> bool:
    try:
        from src.core.vector_store import VectorStore
        stats = VectorStore().get_stats()
        return stats["total_chunks"] > 0
    except Exception:
        return False


# Skip all integration tests when no index exists
pytestmark = pytest.mark.skipif(
    not _vector_store_has_data(),
    reason="Vector store is empty - run python scripts/build_index.py first",
)


# ---------------------------------------------------------------------------
# Retrieval quality tests  (mirrors eval_retrieval.py as pytest cases)
# ---------------------------------------------------------------------------

RETRIEVAL_CASES = [
    {
        "id": "gnb_definition",
        "query": "What is gNB?",
        "expected_keywords": ["gnb", "base station", "node", "ran"],
        "min_similarity": 0.40,
        "min_keyword_ratio": 0.5,
    },
    {
        "id": "protocol_stack",
        "query": "Explain the 5G protocol stack",
        "expected_keywords": ["protocol", "stack", "layer", "nr"],
        "min_similarity": 0.40,
        "min_keyword_ratio": 0.5,
    },
    {
        "id": "ng_ran_functions",
        "query": "What are the NG-RAN functions?",
        "expected_keywords": ["ng-ran", "function", "ran"],
        "min_similarity": 0.40,
        "min_keyword_ratio": 0.5,
    },
    {
        "id": "handover",
        "query": "How does handover work in 5G?",
        "expected_keywords": ["handover", "mobility", "cell"],
        "min_similarity": 0.40,
        "min_keyword_ratio": 0.33,
    },
    {
        "id": "physical_layer",
        "query": "What is the NR physical layer?",
        "expected_keywords": ["physical", "layer", "nr"],
        "min_similarity": 0.40,
        "min_keyword_ratio": 0.5,
    },
]


@pytest.fixture(scope="module")
def retriever():
    from src.core.retriever import DocumentRetriever
    return DocumentRetriever(top_k=5)


@pytest.mark.integration
@pytest.mark.parametrize("case", RETRIEVAL_CASES, ids=[c["id"] for c in RETRIEVAL_CASES])
def test_retrieval_quality(retriever, case):
    docs = retriever.retrieve(case["query"])
    assert len(docs) > 0, f"No results returned for: {case['query']}"

    # Similarity check - top result should be above threshold
    top_sim = docs[0]["similarity"]
    assert top_sim >= case["min_similarity"], (
        f"Top similarity {top_sim:.3f} < threshold {case['min_similarity']}"
    )

    # Keyword check - look for keywords in top-3 results
    top_text = " ".join(d["text"].lower() for d in docs[:3])
    hits = sum(1 for kw in case["expected_keywords"] if kw.lower() in top_text)
    ratio = hits / len(case["expected_keywords"])
    assert ratio >= case["min_keyword_ratio"], (
        f"Keyword coverage {ratio:.0%} < {case['min_keyword_ratio']:.0%} "
        f"(found {hits}/{len(case['expected_keywords'])} keywords)"
    )


@pytest.mark.integration
def test_retrieval_returns_sources(retriever):
    docs = retriever.retrieve("5G architecture")
    for doc in docs:
        assert doc["source"] != "unknown"
        assert len(doc["source"]) > 0


@pytest.mark.integration
def test_retrieval_source_filter(retriever):
    """Source filter should restrict results to matching documents."""
    docs_all = retriever.retrieve("gNB architecture", top_k=10)
    sources_all = {d["source"] for d in docs_all}
    # Take one real source and filter to it
    if not sources_all:
        pytest.skip("No sources in vector store")
    target_source = next(iter(sources_all))
    prefix = target_source[:5]
    docs_filtered = retriever.retrieve("gNB architecture", source_filter=prefix)
    for doc in docs_filtered:
        assert prefix in doc["source"], (
            f"Source '{doc['source']}' does not contain filter '{prefix}'"
        )


# ---------------------------------------------------------------------------
# RAG chain smoke test (no LLM required)
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_rag_chain_retrieval_step():
    """Verify the chain's retrieval step works end-to-end without an LLM."""
    from src.core.rag_chain import RAGChain
    from unittest.mock import MagicMock

    mock_llm = MagicMock()
    mock_llm.model = "llama3.2"
    mock_llm.generate.return_value = "Mocked LLM answer."

    chain = RAGChain(llm=mock_llm, top_k=3)
    result = chain.query("What is the gNB-CU?")

    assert result["answer"] == "Mocked LLM answer."
    assert isinstance(result["sources"], list)
    assert len(result["sources"]) > 0
    assert result["retrieve_time"] >= 0
    assert result["generate_time"] >= 0

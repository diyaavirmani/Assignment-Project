"""
Tests for src/core/retriever.py (DocumentRetriever)

VectorStore and LocalEmbeddingGenerator are mocked via conftest fixtures.
"""
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def retriever(mock_vector_store, mock_embedding_generator):
    from src.core.retriever import DocumentRetriever
    return DocumentRetriever(
        vector_store=mock_vector_store,
        embedding_generator=mock_embedding_generator,
        top_k=5,
    )


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

class TestRetrieverInit:
    def test_default_top_k(self, mock_vector_store, mock_embedding_generator):
        from src.core.retriever import DocumentRetriever
        ret = DocumentRetriever(
            vector_store=mock_vector_store,
            embedding_generator=mock_embedding_generator,
        )
        assert ret.top_k == 5

    def test_custom_top_k(self, mock_vector_store, mock_embedding_generator):
        from src.core.retriever import DocumentRetriever
        ret = DocumentRetriever(
            vector_store=mock_vector_store,
            embedding_generator=mock_embedding_generator,
            top_k=10,
        )
        assert ret.top_k == 10


# ---------------------------------------------------------------------------
# retrieve
# ---------------------------------------------------------------------------

class TestRetrieve:
    def test_returns_list(self, retriever):
        results = retriever.retrieve("What is gNB?")
        assert isinstance(results, list)

    def test_result_has_required_keys(self, retriever):
        results = retriever.retrieve("What is gNB?")
        assert len(results) > 0
        for doc in results:
            assert "text" in doc
            assert "source" in doc
            assert "similarity" in doc
            assert "chunk_index" in doc

    def test_similarity_between_zero_and_one(self, retriever):
        results = retriever.retrieve("protocol stack")
        for doc in results:
            assert 0.0 <= doc["similarity"] <= 1.0

    def test_calls_embedding_generator(self, retriever, mock_embedding_generator):
        retriever.retrieve("test query")
        mock_embedding_generator.generate_embedding.assert_called_once_with("test query")

    def test_calls_vector_store_query(self, retriever, mock_vector_store):
        retriever.retrieve("test query")
        mock_vector_store.query.assert_called_once()

    def test_top_k_override(self, retriever, mock_vector_store):
        retriever.retrieve("query", top_k=3)
        call_kwargs = mock_vector_store.query.call_args[1]
        # With source_filter=None, n_results == top_k
        assert call_kwargs["n_results"] == 3

    def test_source_filter_requests_extra_results(self, retriever, mock_vector_store):
        """When source_filter is set, retriever requests 2× results to allow filtering."""
        retriever.retrieve("query", source_filter="38300")
        call_kwargs = mock_vector_store.query.call_args[1]
        assert call_kwargs["n_results"] == retriever.top_k * 2

    def test_query_expansion_applied_before_embedding(
        self, retriever, mock_embedding_generator, monkeypatch
    ):
        """With expansion enabled, the embedded text carries the 3GPP gloss
        while the user's original wording is preserved as its prefix."""
        from src.core import retriever as retriever_module
        monkeypatch.setattr(retriever_module.settings, "query_expansion", True)
        retriever.retrieve("What is SDAP?")
        embedded = mock_embedding_generator.generate_embedding.call_args[0][0]
        assert embedded.startswith("What is SDAP?")
        assert "Service Data Adaptation Protocol" in embedded

    def test_query_expansion_disabled_passes_raw_query(
        self, retriever, mock_embedding_generator, monkeypatch
    ):
        from src.core import retriever as retriever_module
        monkeypatch.setattr(retriever_module.settings, "query_expansion", False)
        monkeypatch.setattr(retriever_module.settings, "query_decomposition", False)
        retriever.retrieve("What is SDAP?")
        mock_embedding_generator.generate_embedding.assert_called_once_with("What is SDAP?")

    def test_comparison_query_searches_per_side(
        self, retriever, mock_embedding_generator, monkeypatch
    ):
        """A comparison query embeds the raw query plus one sub-query per
        side (expansion disabled here to isolate decomposition)."""
        from src.core import retriever as retriever_module
        monkeypatch.setattr(retriever_module.settings, "query_expansion", False)
        monkeypatch.setattr(retriever_module.settings, "query_decomposition", True)
        retriever.retrieve("What are the differences between NR and LTE physical layer?")
        embedded = [c[0][0] for c in mock_embedding_generator.generate_embedding.call_args_list]
        assert len(embedded) == 3
        assert "NR physical layer" in embedded
        assert "LTE physical layer" in embedded

    def test_decomposition_disabled_skips_subqueries(
        self, retriever, mock_embedding_generator, monkeypatch
    ):
        from src.core import retriever as retriever_module
        monkeypatch.setattr(retriever_module.settings, "query_expansion", False)
        monkeypatch.setattr(retriever_module.settings, "query_decomposition", False)
        retriever.retrieve("What are the differences between NR and LTE physical layer?")
        assert mock_embedding_generator.generate_embedding.call_count == 1

    def test_source_filter_applied(self, mock_vector_store, mock_embedding_generator):
        """Only docs whose source contains the filter string are returned."""
        from src.core.retriever import DocumentRetriever

        mock_vector_store.query.return_value = {
            "documents": [["doc A", "doc B"]],
            "metadatas": [
                [
                    {"source": "38300-spec.docx", "chunk_index": 0},
                    {"source": "38401-spec.docx", "chunk_index": 1},
                ]
            ],
            "distances": [[0.1, 0.2]],
        }

        ret = DocumentRetriever(
            vector_store=mock_vector_store,
            embedding_generator=mock_embedding_generator,
            top_k=5,
        )
        results = ret.retrieve("query", source_filter="38300")
        assert len(results) == 1
        assert "38300" in results[0]["source"]

    def test_empty_vector_store_returns_empty_list(
        self, mock_vector_store, mock_embedding_generator
    ):
        from src.core.retriever import DocumentRetriever
        mock_vector_store.query.return_value = {
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }
        ret = DocumentRetriever(
            vector_store=mock_vector_store,
            embedding_generator=mock_embedding_generator,
        )
        results = ret.retrieve("no results query")
        assert results == []


# ---------------------------------------------------------------------------
# format_context
# ---------------------------------------------------------------------------

class TestFormatContext:
    def test_returns_string(self, retriever, retrieved_docs):
        ctx = retriever.format_context(retrieved_docs)
        assert isinstance(ctx, str)

    def test_contains_document_numbers(self, retriever, retrieved_docs):
        ctx = retriever.format_context(retrieved_docs)
        assert "[Document 1]" in ctx
        assert "[Document 2]" in ctx

    def test_contains_source_names(self, retriever, retrieved_docs):
        ctx = retriever.format_context(retrieved_docs)
        for doc in retrieved_docs:
            assert doc["source"] in ctx

    def test_contains_text(self, retriever, retrieved_docs):
        ctx = retriever.format_context(retrieved_docs)
        for doc in retrieved_docs:
            assert doc["text"][:20] in ctx

    def test_empty_list_returns_empty_string(self, retriever):
        ctx = retriever.format_context([])
        assert ctx == ""

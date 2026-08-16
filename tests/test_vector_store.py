"""
Tests for src/core/vector_store.py (VectorStore)

Uses a real ChromaDB in-memory/temp-dir client so no persistent state leaks
between tests.
"""
import pytest
import tempfile
from pathlib import Path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_vector_store(tmp_path):
    """A fresh VectorStore backed by a temp directory."""
    from src.core.vector_store import VectorStore
    return VectorStore(
        persist_directory=str(tmp_path / "vectordb"),
        collection_name="test_collection",
    )


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

class TestVectorStoreInit:
    def test_creates_persist_directory(self, tmp_path):
        from src.core.vector_store import VectorStore
        db_path = tmp_path / "new_db"
        assert not db_path.exists()
        VectorStore(persist_directory=str(db_path))
        assert db_path.exists()

    def test_initial_count_is_zero(self, tmp_vector_store):
        stats = tmp_vector_store.get_stats()
        assert stats["total_chunks"] == 0

    def test_collection_name_in_stats(self, tmp_vector_store):
        stats = tmp_vector_store.get_stats()
        assert stats["collection_name"] == "test_collection"


# ---------------------------------------------------------------------------
# add_chunks
# ---------------------------------------------------------------------------

class TestAddChunks:
    def test_adds_chunks_to_collection(
        self, tmp_vector_store, sample_chunks_with_embeddings
    ):
        tmp_vector_store.add_chunks(sample_chunks_with_embeddings)
        stats = tmp_vector_store.get_stats()
        assert stats["total_chunks"] == len(sample_chunks_with_embeddings)

    def test_adds_in_batches_larger_than_100(self, tmp_vector_store, mock_embedding):
        """Verify batch logic handles >100 chunks."""
        chunks = [
            {
                "text": f"chunk {i}",
                "embedding": mock_embedding,
                "metadata": {"source": "test.docx", "chunk_index": i, "chunk_size": 10},
            }
            for i in range(150)
        ]
        tmp_vector_store.add_chunks(chunks)
        assert tmp_vector_store.get_stats()["total_chunks"] == 150

    def test_metadata_stored_correctly(
        self, tmp_vector_store, sample_chunks_with_embeddings
    ):
        tmp_vector_store.add_chunks(sample_chunks_with_embeddings)
        # Query back and check a metadata field exists
        results = tmp_vector_store.query(
            query_embedding=sample_chunks_with_embeddings[0]["embedding"],
            n_results=1,
        )
        meta = results["metadatas"][0][0]
        assert "source" in meta
        assert "chunk_index" in meta


# ---------------------------------------------------------------------------
# query
# ---------------------------------------------------------------------------

class TestQuery:
    def test_query_returns_expected_keys(
        self, tmp_vector_store, sample_chunks_with_embeddings, mock_embedding
    ):
        tmp_vector_store.add_chunks(sample_chunks_with_embeddings)
        result = tmp_vector_store.query(mock_embedding, n_results=2)
        assert "documents" in result
        assert "metadatas" in result
        assert "distances" in result

    def test_query_respects_n_results(
        self, tmp_vector_store, sample_chunks_with_embeddings, mock_embedding
    ):
        tmp_vector_store.add_chunks(sample_chunks_with_embeddings)
        result = tmp_vector_store.query(mock_embedding, n_results=2)
        assert len(result["documents"][0]) == 2

    def test_query_returns_documents_as_strings(
        self, tmp_vector_store, sample_chunks_with_embeddings, mock_embedding
    ):
        tmp_vector_store.add_chunks(sample_chunks_with_embeddings)
        result = tmp_vector_store.query(mock_embedding, n_results=1)
        doc = result["documents"][0][0]
        assert isinstance(doc, str)
        assert len(doc) > 0


# ---------------------------------------------------------------------------
# clear
# ---------------------------------------------------------------------------

class TestClear:
    def test_clear_removes_all_chunks(
        self, tmp_vector_store, sample_chunks_with_embeddings
    ):
        tmp_vector_store.add_chunks(sample_chunks_with_embeddings)
        assert tmp_vector_store.get_stats()["total_chunks"] > 0
        tmp_vector_store.clear()
        assert tmp_vector_store.get_stats()["total_chunks"] == 0

    def test_can_add_after_clear(
        self, tmp_vector_store, sample_chunks_with_embeddings
    ):
        tmp_vector_store.add_chunks(sample_chunks_with_embeddings)
        tmp_vector_store.clear()
        tmp_vector_store.add_chunks(sample_chunks_with_embeddings[:1])
        assert tmp_vector_store.get_stats()["total_chunks"] == 1

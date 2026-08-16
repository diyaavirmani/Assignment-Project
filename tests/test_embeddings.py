"""
Tests for src/core/embeddings.py (LocalEmbeddingGenerator)

The SentenceTransformer model is mocked so tests run without downloading
any model weights.
"""
import pytest
from unittest.mock import MagicMock, patch
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_generator(dim=384):
    """Return a LocalEmbeddingGenerator with a mocked SentenceTransformer."""
    with patch("src.core.embeddings.SentenceTransformer") as MockST:
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = dim
        mock_model.device = "cpu"
        MockST.return_value = mock_model

        from src.core.embeddings import LocalEmbeddingGenerator
        gen = LocalEmbeddingGenerator(model_name="mini")
        gen.model = mock_model          # keep a handle for assertions
        return gen


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

class TestLocalEmbeddingGeneratorInit:
    def test_known_model_alias_resolves(self):
        gen = _build_generator()
        assert gen.model_name == "mini"

    def test_embedding_dim_stored(self):
        gen = _build_generator(dim=384)
        assert gen.embedding_dim == 384

    def test_unknown_model_alias_passes_through(self):
        with patch("src.core.embeddings.SentenceTransformer") as MockST:
            mock_model = MagicMock()
            mock_model.get_sentence_embedding_dimension.return_value = 768
            mock_model.device = "cpu"
            MockST.return_value = mock_model

            from src.core.embeddings import LocalEmbeddingGenerator
            gen = LocalEmbeddingGenerator(model_name="custom-model-name")
            # Should still construct successfully
            assert gen.model_name == "custom-model-name"

    def test_raises_without_sentence_transformers(self):
        with patch("src.core.embeddings.SentenceTransformer", None):
            from src.core.embeddings import LocalEmbeddingGenerator
            with pytest.raises(ImportError):
                LocalEmbeddingGenerator()


# ---------------------------------------------------------------------------
# generate_embedding
# ---------------------------------------------------------------------------

class TestGenerateEmbedding:
    def test_returns_list_of_floats(self, embedding_dim):
        gen = _build_generator(dim=embedding_dim)
        gen.model.encode.return_value = np.array([0.1] * embedding_dim)

        result = gen.generate_embedding("test text")
        assert isinstance(result, list)
        assert len(result) == embedding_dim
        assert all(isinstance(v, float) for v in result)

    def test_calls_model_encode(self):
        gen = _build_generator()
        gen.model.encode.return_value = np.array([0.0] * 384)
        gen.generate_embedding("hello world")
        gen.model.encode.assert_called_once_with("hello world", convert_to_numpy=True)

    def test_generate_alias_works(self):
        """generate() should behave identically to generate_embedding()."""
        gen = _build_generator()
        gen.model.encode.return_value = np.array([0.5] * 384)
        assert gen.generate("text") == gen.generate_embedding("text")


# ---------------------------------------------------------------------------
# generate_embeddings_batch
# ---------------------------------------------------------------------------

class TestGenerateEmbeddingsBatch:
    def test_returns_list_of_lists(self, embedding_dim):
        gen = _build_generator(dim=embedding_dim)
        texts = ["text one", "text two", "text three"]
        gen.model.encode.return_value = np.array([[0.1] * embedding_dim] * 3)

        result = gen.generate_embeddings_batch(texts)
        assert isinstance(result, list)
        assert len(result) == 3
        assert all(isinstance(row, list) for row in result)

    def test_batch_length_matches_input(self):
        gen = _build_generator()
        texts = ["a", "b", "c", "d", "e"]
        gen.model.encode.return_value = np.zeros((5, 384))
        result = gen.generate_embeddings_batch(texts)
        assert len(result) == 5


# ---------------------------------------------------------------------------
# embed_chunks
# ---------------------------------------------------------------------------

class TestEmbedChunks:
    def test_adds_embedding_field(self, sample_chunks, embedding_dim):
        gen = _build_generator(dim=embedding_dim)
        n = len(sample_chunks)
        gen.model.encode.return_value = np.zeros((n, embedding_dim))

        result = gen.embed_chunks(sample_chunks)
        for chunk in result:
            assert "embedding" in chunk
            assert isinstance(chunk["embedding"], list)
            assert len(chunk["embedding"]) == embedding_dim

    def test_original_fields_preserved(self, sample_chunks, embedding_dim):
        gen = _build_generator(dim=embedding_dim)
        n = len(sample_chunks)
        gen.model.encode.return_value = np.zeros((n, embedding_dim))

        result = gen.embed_chunks(sample_chunks)
        for orig, embedded in zip(sample_chunks, result):
            assert embedded["text"] == orig["text"]
            assert embedded["metadata"] == orig["metadata"]

    def test_output_length_matches_input(self, sample_chunks, embedding_dim):
        gen = _build_generator(dim=embedding_dim)
        gen.model.encode.return_value = np.zeros((len(sample_chunks), embedding_dim))
        result = gen.embed_chunks(sample_chunks)
        assert len(result) == len(sample_chunks)

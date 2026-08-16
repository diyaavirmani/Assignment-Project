"""Tests for src/core/openai_embeddings.py."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.core.openai_embeddings import OpenAIEmbeddingGenerator


def _embedding(values, index=0):
    return SimpleNamespace(embedding=values, index=index)


def _response(rows):
    return SimpleNamespace(data=[_embedding(values, i) for i, values in enumerate(rows)])


@pytest.fixture
def mock_openai_client():
    client = MagicMock()
    client.embeddings.create.return_value = _response([[0.1, 0.2, 0.3]])
    return client


class TestOpenAIEmbeddingGenerator:
    def test_single_embedding(self, mock_openai_client):
        generator = OpenAIEmbeddingGenerator(
            model_name="text-embedding-3-small",
            client=mock_openai_client,
        )

        result = generator.generate_embedding("hello")

        assert result == [0.1, 0.2, 0.3]
        mock_openai_client.embeddings.create.assert_called_once_with(
            model="text-embedding-3-small",
            input=["hello"],
        )

    def test_generate_alias(self, mock_openai_client):
        generator = OpenAIEmbeddingGenerator(client=mock_openai_client)

        assert generator.generate("hello") == [0.1, 0.2, 0.3]

    def test_batch_embeddings(self):
        client = MagicMock()
        client.embeddings.create.side_effect = [
            _response([[0.1], [0.2]]),
            _response([[0.3]]),
        ]
        generator = OpenAIEmbeddingGenerator(
            model_name="text-embedding-3-small",
            batch_size=2,
            client=client,
        )

        result = generator.generate_embeddings_batch(["one", "two", "three"])

        assert result == [[0.1], [0.2], [0.3]]
        assert client.embeddings.create.call_count == 2
        assert client.embeddings.create.call_args_list[0].kwargs["input"] == [
            "one",
            "two",
        ]
        assert client.embeddings.create.call_args_list[1].kwargs["input"] == ["three"]

    def test_empty_batch_returns_empty_list(self, mock_openai_client):
        generator = OpenAIEmbeddingGenerator(client=mock_openai_client)

        assert generator.generate_embeddings_batch([]) == []
        mock_openai_client.embeddings.create.assert_not_called()

    def test_embed_chunks_preserves_original_fields(self, sample_chunks):
        client = MagicMock()
        client.embeddings.create.return_value = _response([[0.1], [0.2], [0.3]])
        generator = OpenAIEmbeddingGenerator(client=client)

        result = generator.embed_chunks(sample_chunks)

        assert len(result) == len(sample_chunks)
        for original, embedded in zip(sample_chunks, result):
            assert embedded["text"] == original["text"]
            assert embedded["metadata"] == original["metadata"]
            assert "embedding" in embedded
        assert result[0]["embedding"] == [0.1]

    def test_missing_api_key_raises_clear_error(self):
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            OpenAIEmbeddingGenerator(
                model_name="text-embedding-3-small",
                api_key="",
            )

    def test_invalid_model_raises_clear_error(self, mock_openai_client):
        with pytest.raises(ValueError, match="OPENAI_EMBEDDING_MODEL"):
            OpenAIEmbeddingGenerator(model_name="", client=mock_openai_client)

    def test_invalid_batch_size_raises_clear_error(self, mock_openai_client):
        with pytest.raises(ValueError, match="batch_size"):
            OpenAIEmbeddingGenerator(batch_size=0, client=mock_openai_client)

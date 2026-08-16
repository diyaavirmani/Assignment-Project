"""
OpenAI embedding generation for the 3GPP RAG pipeline.

This module mirrors the small interface used by LocalEmbeddingGenerator while
keeping OpenAI-specific configuration and clients isolated from the existing
local embedding path. Tests can inject a fake client, so unit tests never need
to call the OpenAI API.
"""

import logging
from typing import Any, Iterable, List, Optional, Sequence

from src.config import settings

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - exercised only when dependency is absent
    OpenAI = None

logger = logging.getLogger(__name__)


class OpenAIEmbeddingGenerator:
    """Generate text embeddings using the OpenAI Python SDK."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        batch_size: int = 100,
        api_key: Optional[str] = None,
        client: Optional[Any] = None,
    ) -> None:
        """
        Args:
            model_name: OpenAI embedding model name. Defaults to
                Settings.openai_embedding_model.
            batch_size: Number of texts to send per embeddings request.
            api_key: OpenAI API key. Defaults to Settings.openai_api_key.
            client: Optional prebuilt OpenAI-compatible client for tests.

        Raises:
            ImportError: If the OpenAI SDK is not installed and no client is injected.
            ValueError: If no API key is configured and no client is injected.
        """
        self.model_name = model_name if model_name is not None else settings.openai_embedding_model
        self.batch_size = batch_size

        if self.batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")
        if not self.model_name:
            raise ValueError("OPENAI_EMBEDDING_MODEL is required")

        if client is not None:
            self.client = client
            return

        resolved_api_key = api_key if api_key is not None else settings.openai_api_key
        if not resolved_api_key:
            raise ValueError("OPENAI_API_KEY is required to use OpenAIEmbeddingGenerator")
        if OpenAI is None:
            raise ImportError("openai is required. Install with: pip install openai")

        self.client = OpenAI(api_key=resolved_api_key)

    def generate(self, text: str) -> List[float]:
        """Alias for generate_embedding to match existing embedding usage."""
        return self.generate_embedding(text)

    def generate_embedding(self, text: str) -> List[float]:
        """Generate an embedding for a single text string."""
        return self.generate_embeddings_batch([text])[0]

    def generate_embeddings_batch(self, texts: Sequence[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in batches.

        Args:
            texts: Text strings to embed.

        Returns:
            A list of embedding vectors in the same order as the input texts.
        """
        if not texts:
            return []

        embeddings: List[List[float]] = []
        for batch in self._batched(texts, self.batch_size):
            response = self.client.embeddings.create(
                model=self.model_name,
                input=list(batch),
            )
            embeddings.extend(self._extract_embeddings(response))

        return embeddings

    def embed_chunks(self, chunks: List[dict]) -> List[dict]:
        """
        Generate embeddings for document chunks.

        Args:
            chunks: Chunk dictionaries containing a ``text`` field.

        Returns:
            Copies of the original chunks with an added ``embedding`` field.
        """
        if not chunks:
            return []

        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.generate_embeddings_batch(texts)

        embedded_chunks = []
        for chunk, embedding in zip(chunks, embeddings):
            chunk_with_embedding = chunk.copy()
            chunk_with_embedding["embedding"] = embedding
            embedded_chunks.append(chunk_with_embedding)

        logger.info("Generated %s OpenAI embeddings", len(embedded_chunks))
        return embedded_chunks

    @staticmethod
    def _batched(texts: Sequence[str], size: int) -> Iterable[Sequence[str]]:
        for i in range(0, len(texts), size):
            yield texts[i : i + size]

    @staticmethod
    def _extract_embeddings(response: Any) -> List[List[float]]:
        data = list(response.data)
        data.sort(key=lambda item: getattr(item, "index", 0))
        return [list(item.embedding) for item in data]


OpenAIEmbeddings = OpenAIEmbeddingGenerator

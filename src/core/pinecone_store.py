"""
Pinecone vector store wrapper for the 3GPP RAG pipeline.

The class intentionally returns query results in the same broad shape as the
existing Chroma-backed VectorStore: ``documents``, ``metadatas``, and
``distances``. That keeps the retriever interface stable while Pinecone is the
production vector-store provider.
"""

import logging
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

from src.config import settings

try:
    from pinecone import Pinecone
except ImportError:  # pragma: no cover - exercised only when dependency is absent
    Pinecone = None

logger = logging.getLogger(__name__)


class PineconeVectorStore:
    """Manage vector database operations with Pinecone."""

    METADATA_FIELDS = (
        "source",
        "chunk_index",
        "chunk_size",
        "domain",
        "generation",
        "spec_number",
        "spec_title",
        "release",
        "version",
        "section",
        "section_title",
        "document_type",
    )

    def __init__(
        self,
        index_name: Optional[str] = None,
        namespace: Optional[str] = None,
        batch_size: int = 100,
        api_key: Optional[str] = None,
        client: Optional[Any] = None,
        index: Optional[Any] = None,
    ) -> None:
        """
        Args:
            index_name: Pinecone index name. Defaults to Settings.pinecone_index_name.
            namespace: Pinecone namespace. Defaults to Settings.pinecone_namespace.
            batch_size: Number of vectors per upsert request.
            api_key: Pinecone API key. Defaults to Settings.pinecone_api_key.
            client: Optional prebuilt Pinecone-compatible client for tests.
            index: Optional prebuilt Pinecone-compatible index for tests.

        Raises:
            ImportError: If the Pinecone SDK is not installed and no client/index is injected.
            ValueError: If required Pinecone configuration is missing.
        """
        self.index_name = index_name if index_name is not None else settings.pinecone_index_name
        self.namespace = (
            namespace if namespace is not None else settings.pinecone_namespace
        ) or "3gpp-specs"
        self.batch_size = batch_size
        self._indexed_spec_cache: Set[str] = set()
        self._indexed_spec_cache_candidates: Set[str] = set()

        if self.batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")

        if index is not None:
            self.index = index
            self.client = client
            return

        if not self.index_name:
            raise ValueError("PINECONE_INDEX_NAME is required to use PineconeVectorStore")

        if client is not None:
            self.client = client
            self.index = client.Index(self.index_name)
            return

        resolved_api_key = api_key if api_key is not None else settings.pinecone_api_key
        if not resolved_api_key:
            raise ValueError("PINECONE_API_KEY is required to use PineconeVectorStore")
        if Pinecone is None:
            raise ImportError("pinecone is required. Install with: pip install pinecone")

        self.client = Pinecone(api_key=resolved_api_key)
        self.index = self.client.Index(self.index_name)

    def add_chunks(self, chunks: List[Dict]) -> None:
        """
        Add chunks with embeddings to the Pinecone namespace.

        Args:
            chunks: List of chunks with ``text``, ``embedding``, and optional
                ``metadata`` fields.
        """
        if not chunks:
            return

        vectors = [
            {
                "id": self._vector_id(chunk, i),
                "values": list(chunk["embedding"]),
                "metadata": self._metadata_for_chunk(chunk, i),
            }
            for i, chunk in enumerate(chunks)
        ]

        for batch in self._batched(vectors, self.batch_size):
            self.index.upsert(vectors=list(batch), namespace=self.namespace)

        self._invalidate_indexed_spec_cache()
        logger.info("Upserted %s chunks into Pinecone namespace %s", len(chunks), self.namespace)

    def query(
        self,
        query_embedding: List[float],
        n_results: int = 5,
        where_filter: Optional[Dict] = None,
    ) -> Dict:
        """
        Query the Pinecone index with optional metadata filtering.

        Returns a Chroma-like shape with cosine scores converted to distances
        so DocumentRetriever can continue to compute ``similarity = 1 - distance``.
        """
        kwargs: Dict[str, Any] = {
            "vector": query_embedding,
            "top_k": n_results,
            "include_metadata": True,
            "namespace": self.namespace,
        }
        if where_filter:
            kwargs["filter"] = where_filter

        response = self.index.query(**kwargs)
        matches = self._read(response, "matches", default=[]) or []

        documents: List[str] = []
        metadatas: List[Dict] = []
        distances: List[float] = []

        for match in matches:
            metadata = dict(self._read(match, "metadata", default={}) or {})
            score = float(self._read(match, "score", default=0.0) or 0.0)

            documents.append(str(metadata.get("text", "")))
            metadata.pop("text", None)
            metadatas.append(metadata)
            distances.append(1.0 - score)

        return {
            "documents": [documents],
            "metadatas": [metadatas],
            "distances": [distances],
        }

    def get_stats(self) -> Dict:
        """Get statistics about the Pinecone index and configured namespace."""
        stats = self.index.describe_index_stats()
        namespaces = self._read(stats, "namespaces", default={}) or {}
        namespace_stats = namespaces.get(self.namespace, {})

        namespace_count = self._read(namespace_stats, "vector_count", default=0) or 0
        total_count = self._read(stats, "total_vector_count", default=namespace_count) or 0

        return {
            "index_name": self.index_name,
            "namespace": self.namespace,
            "total_chunks": namespace_count,
            "total_vector_count": total_count,
        }

    def get_indexed_spec_numbers(self, candidate_spec_numbers: Iterable[str]) -> Set[str]:
        """
        Return candidate 3GPP spec numbers present in this Pinecone namespace.

        Pinecone serverless does not support metadata filtering on
        ``describe_index_stats()``, so this uses one unfiltered stats call to
        determine vector dimension and then probes candidates with
        metadata-filtered ``query`` calls. Query responses request no metadata
        or values; only the presence of a match matters.
        """
        candidates = {spec_number for spec_number in candidate_spec_numbers if spec_number}
        if not candidates:
            return set()

        cached_hits = candidates & self._indexed_spec_cache
        candidates_to_probe = candidates - self._indexed_spec_cache_candidates
        if not candidates_to_probe:
            return cached_hits

        if not self.namespace:
            logger.warning("Cannot detect indexed Pinecone specs without a namespace")
            return cached_hits

        try:
            stats = self.index.describe_index_stats()
        except Exception as e:
            logger.warning("Failed to read Pinecone index stats for catalog detection: %s", e)
            return cached_hits

        dimension = self._dimension_from_stats(stats)
        if dimension is None:
            logger.warning("Cannot detect indexed Pinecone specs without index dimension")
            return cached_hits

        probe = [0.0] * dimension
        probe[0] = 1.0

        indexed: Set[str] = set()
        checked: Set[str] = set()
        for spec_number in candidates_to_probe:
            try:
                response = self.index.query(
                    vector=probe,
                    top_k=1,
                    namespace=self.namespace,
                    filter={"spec_number": {"$eq": spec_number}},
                    include_metadata=False,
                    include_values=False,
                )
            except Exception as e:
                logger.warning(
                    "Failed to probe Pinecone indexed status for TS %s: %s",
                    spec_number,
                    e,
                )
                continue

            checked.add(spec_number)
            matches = self._read(response, "matches", default=[]) or []
            if matches:
                indexed.add(spec_number)

        self._indexed_spec_cache.update(indexed)
        self._indexed_spec_cache_candidates.update(checked)
        return cached_hits | indexed

    def clear(self) -> None:
        """Clear only this store's namespace, leaving the index intact."""
        logger.warning("Clearing Pinecone namespace: %s", self.namespace)
        self.index.delete(delete_all=True, namespace=self.namespace)
        self._invalidate_indexed_spec_cache()

    def _metadata_for_chunk(self, chunk: Dict, position: int) -> Dict:
        text = chunk["text"]
        source_metadata = chunk.get("metadata", {}) or {}

        metadata: Dict[str, Any] = {
            "text": text,
            "source": source_metadata.get("source", "unknown"),
            "chunk_index": source_metadata.get("chunk_index", position),
            "chunk_size": source_metadata.get("chunk_size", len(text)),
            "domain": source_metadata.get("domain", "unknown"),
            "generation": source_metadata.get("generation", "unknown"),
            "spec_number": source_metadata.get("spec_number", "unknown"),
            "spec_title": source_metadata.get("spec_title", "unknown"),
        }

        for field in self.METADATA_FIELDS:
            if field in metadata:
                continue
            value = source_metadata.get(field)
            if value not in (None, ""):
                metadata[field] = value

        for key, value in source_metadata.items():
            if key not in metadata and value not in (None, ""):
                metadata[key] = value

        return {
            key: self._clean_metadata_value(value)
            for key, value in metadata.items()
            if value is not None
        }

    @staticmethod
    def _vector_id(chunk: Dict, position: int) -> str:
        metadata = chunk.get("metadata", {}) or {}
        if chunk.get("id") is not None:
            return str(chunk["id"])
        if metadata.get("source") is not None and metadata.get("chunk_index") is not None:
            return f"{metadata['source']}:{metadata['chunk_index']}"
        if chunk.get("chunk_id") is not None:
            return f"chunk_{chunk['chunk_id']}"
        return f"chunk_{position}"

    @staticmethod
    def _clean_metadata_value(value: Any) -> Any:
        if isinstance(value, (str, bool, int, float)):
            return value
        if isinstance(value, list) and all(isinstance(item, str) for item in value):
            return value
        return str(value)

    @staticmethod
    def _batched(vectors: Sequence[Dict], size: int) -> Iterable[Sequence[Dict]]:
        for i in range(0, len(vectors), size):
            yield vectors[i : i + size]

    def _invalidate_indexed_spec_cache(self) -> None:
        self._indexed_spec_cache.clear()
        self._indexed_spec_cache_candidates.clear()

    @classmethod
    def _dimension_from_stats(cls, stats: Any) -> Optional[int]:
        raw_dimension = cls._read(stats, "dimension")
        try:
            dimension = int(raw_dimension)
        except (TypeError, ValueError):
            return None
        if dimension <= 0:
            return None
        return dimension

    @staticmethod
    def _read(obj: Any, key: str, default: Any = None) -> Any:
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)


PineconeStore = PineconeVectorStore

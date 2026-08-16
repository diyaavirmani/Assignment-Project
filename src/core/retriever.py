"""
Semantic document retrieval for the 3GPP RAG pipeline.

Given a natural language query this module:
  1. Embeds the query using the same model used at index time (bge-small by default)
  2. Performs approximate nearest-neighbour search in the ChromaDB vector store
  3. Applies an optional source-document filter
  4. Returns the top-k most similar chunks with similarity scores

The retriever is intentionally stateless (no caching) so it can be safely
shared across concurrent sessions via the FastAPI app_state.

Example:
    retriever = DocumentRetriever(top_k=5)
    docs = retriever.retrieve("What is the F1 interface?")
    context = retriever.format_context(docs)
"""

from typing import Any, List, Dict, Optional
import logging

from src.config import settings
from src.core.providers import create_embedding_generator, create_vector_store
from src.core.reranker import CrossEncoderReranker
from src.core.retrieval_fusion import fused_retrieve

logger = logging.getLogger(__name__)


class DocumentRetriever:
    """Semantic retriever that embeds queries and searches the vector store.

    Attributes:
        vector_store: ChromaDB-backed store holding all indexed spec chunks.
        embedding_generator: Sentence-transformer model for query encoding.
        top_k: Default number of chunks to return per query.
    """

    def __init__(
        self,
        vector_store: Optional[Any] = None,
        embedding_generator: Optional[Any] = None,
        reranker: Optional[Any] = None,
        reranker_enabled: Optional[bool] = None,
        top_k: int = 5,
    ) -> None:
        """
        Args:
            vector_store: Vector store instance. A provider-configured
                instance is created if not provided.
            embedding_generator: Embedding generator instance. A
                provider-configured instance is created if not provided.
            reranker: Optional reranker instance. Tests may inject a fake.
            reranker_enabled: Override reranker configuration for this retriever.
            top_k: Default number of chunks to return. Can be overridden
                per-query via the top_k parameter of retrieve().
        """
        created_vector_store = vector_store is None
        self.vector_store = vector_store or create_vector_store()
        self.embedding_generator = embedding_generator or create_embedding_generator()
        self.top_k = top_k
        self.reranker_enabled = self._should_enable_reranker(
            explicit=reranker_enabled,
            created_vector_store=created_vector_store,
            reranker=reranker,
        )
        self.reranker = (
            reranker
            if reranker is not None
            else (
                CrossEncoderReranker(model_name=settings.reranker_model)
                if self.reranker_enabled
                else None
            )
        )
        logger.info(
            "Initialized DocumentRetriever (top_k=%s, embeddings=%s, "
            "vector_store=%s, reranker_enabled=%s)",
            top_k,
            settings.embedding_provider,
            settings.vector_store_provider,
            self.reranker_enabled,
        )

    @staticmethod
    def _build_where_filter(
        domain: Optional[str] = None,
        generation: Optional[str] = None,
    ) -> Optional[Dict]:
        """Build a ChromaDB ``where`` clause from domain/generation filters.

        Returns None when no filters are requested (search entire collection).
        """
        conditions = []
        if domain:
            conditions.append({"domain": domain})
        if generation:
            conditions.append({"generation": generation})

        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        source_filter: Optional[str] = None,
        domain: Optional[str] = None,
        generation: Optional[str] = None,
    ) -> List[Dict]:
        """Retrieve the most semantically similar chunks for a query.

        Args:
            query: Natural language question or search string.
            top_k: Override the default number of results for this call.
            source_filter: If set, only return chunks whose source filename
                contains this string (e.g. "38401"). Applied post-retrieval.
            domain: If set, restrict retrieval to "RAN" or "CORE" chunks.
                Applied via ChromaDB metadata filter (pre-retrieval).
            generation: If set, restrict retrieval to "5G" or "LTE" chunks.
                Applied via ChromaDB metadata filter (pre-retrieval).

        Returns:
            List of chunk dicts, each with keys:
                - text        (str)   : chunk content
                - source      (str)   : filename of the source document
                - chunk_index (int)   : position of the chunk in its document
                - similarity  (float) : cosine similarity in [0, 1]
                - domain      (str)   : "RAN" | "CORE" | "unknown"
                - generation  (str)   : "5G" | "LTE" | "unknown"
                - spec_number (str)   : e.g. "38.300"
                - spec_title  (str)   : human-readable spec title
            Sorted descending by similarity, length <= top_k.
        """
        requested_k = top_k or self.top_k
        final_k = (
            top_k
            if top_k is not None
            else settings.reranker_top_k if self.reranker_enabled else requested_k
        )
        candidate_k = (
            max(settings.reranker_candidate_k, final_k) if self.reranker_enabled else final_k
        )

        logger.info(
            f"Retrieving documents for query: '{query[:50]}...' "
            f"[domain={domain}, generation={generation}]"
        )

        where_filter = self._build_where_filter(domain=domain, generation=generation)

        # Bind this call's filters into a search primitive, then delegate the
        # query-rewriting rank fusion to the shared, embedding-agnostic
        # orchestration (also used by the Streamlit Cloud path).
        def search_fn(text: str, n: int) -> List[Dict]:
            return self._search(text, n, where_filter, source_filter)

        retrieved_docs = fused_retrieve(
            query,
            search_fn,
            candidate_k,
            expand=settings.query_expansion,
            decompose=settings.query_decomposition,
        )
        retrieved_docs = self._rerank(query, retrieved_docs, final_k)

        logger.info(f"Retrieved {len(retrieved_docs)} documents")
        return retrieved_docs

    def _search(
        self,
        embed_text: str,
        n: int,
        where_filter: Optional[Dict],
        source_filter: Optional[str],
    ) -> List[Dict]:
        """Embed one query string and return up to n matching chunk dicts.

        Over-fetches when a post-retrieval ``source_filter`` is active so
        that up to n survive the filter.
        """
        query_embedding = self.embedding_generator.generate_embedding(embed_text)

        n_fetch = n * 2 if source_filter else n
        results = self.vector_store.query(
            query_embedding=query_embedding,
            n_results=n_fetch,
            where_filter=where_filter,
        )

        retrieved_docs = []
        for doc, metadata, distance in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            if source_filter and source_filter not in metadata.get("source", ""):
                continue

            retrieved_docs.append(
                {
                    "text": doc,
                    "source": metadata.get("source", "unknown"),
                    "chunk_index": metadata.get("chunk_index", 0),
                    "similarity": 1 - distance,
                    "domain": metadata.get("domain", "unknown"),
                    "generation": metadata.get("generation", "unknown"),
                    "spec_number": metadata.get("spec_number", "unknown"),
                    "spec_title": metadata.get("spec_title", "unknown"),
                }
            )

            if len(retrieved_docs) >= n:
                break

        return retrieved_docs

    def _rerank(self, query: str, documents: List[Dict], final_k: int) -> List[Dict]:
        """Optionally rerank candidate documents with a cross-encoder."""
        if not self.reranker_enabled or self.reranker is None:
            return documents[:final_k]

        candidates = []
        for rank, doc in enumerate(documents, start=1):
            candidate = dict(doc)
            candidate.setdefault("rank_before_reranking", rank)
            candidate.setdefault("vector_similarity", candidate.get("similarity"))
            candidates.append(candidate)
        return self.reranker.rerank(query, candidates, top_n=final_k)

    def _should_enable_reranker(
        self,
        *,
        explicit: Optional[bool],
        created_vector_store: bool,
        reranker: Optional[Any],
    ) -> bool:
        if explicit is not None:
            return explicit
        if reranker is not None:
            return True
        if not settings.reranker_enabled:
            return False
        if settings.vector_store_provider.strip().lower() != "pinecone":
            return False
        return created_vector_store and type(self.vector_store).__name__ == "PineconeVectorStore"

    def format_context(self, documents: List[Dict]) -> str:
        """Format retrieved chunks into a numbered context block for the LLM prompt.

        Each chunk is prefixed with its index, source filename, and similarity
        score so the LLM can cite the right document in its answer.

        Args:
            documents: List of chunk dicts as returned by retrieve().

        Returns:
            Multi-line string ready to be inserted into PROMPT_TEMPLATE.
            Returns an empty string if documents is empty.
        """
        context_parts = []

        for i, doc in enumerate(documents, 1):
            spec_info = ""
            if doc.get("spec_number") and doc["spec_number"] != "unknown":
                spec_info = f", Spec: TS {doc['spec_number']}"
            context_parts.append(
                f"[S{i}]\n"
                f"[Document {i}] (Source ID: S{i}, Source: {doc['source']}{spec_info}, "
                f"Similarity: {doc['similarity']:.3f})\n"
                f"{doc['text']}\n"
            )

        return "\n".join(context_parts)


if __name__ == "__main__":
    # Test the retriever
    import sys
    from src.core.providers import create_vector_store

    logging.basicConfig(level=logging.INFO)

    print("\n" + "=" * 60)
    print("Document Retriever Test")
    print("=" * 60 + "\n")

    # Check if vector store has data
    store = create_vector_store()
    stats = store.get_stats()

    if stats["total_chunks"] == 0:
        print("❌ Error: Vector store is empty!")
        print("Please run: python scripts/build_index.py")
        sys.exit(1)

    print(f"Vector store has {stats['total_chunks']} chunks\n")

    # Initialize retriever
    retriever = DocumentRetriever(top_k=3)

    # Test query
    query = "What is the 5G protocol architecture?"
    docs = retriever.retrieve(query)

    print(f"Query: {query}")
    print("=" * 60)

    for i, doc in enumerate(docs, 1):
        print(f"\nResult {i} (similarity: {doc['similarity']:.3f}):")
        print(f"Source: {doc['source']}")
        print(f"Text: {doc['text'][:200]}...")

    print("\n" + "=" * 60)
    print("Context for LLM:")
    print("=" * 60)
    print(retriever.format_context(docs))
    print("\n" + "=" * 60 + "\n")

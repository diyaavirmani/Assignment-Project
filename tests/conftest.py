"""
Shared pytest fixtures for the 3GPP RAG Assistant test suite
"""

import pytest
from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# Embedding fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def embedding_dim():
    return 384


@pytest.fixture
def mock_embedding(embedding_dim):
    """A single fake embedding vector."""
    return [0.1] * embedding_dim


@pytest.fixture
def mock_embeddings(mock_embedding):
    """A list of three fake embedding vectors."""
    return [mock_embedding, mock_embedding, mock_embedding]


# ---------------------------------------------------------------------------
# Document chunk fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_chunk():
    return {
        "text": "The gNB-CU is a logical node hosting RRC and PDCP protocols.",
        "metadata": {
            "source": "38300-g30.docx",
            "chunk_index": 0,
            "chunk_size": 62,
        },
        "chunk_id": 0,
    }


@pytest.fixture
def sample_chunks():
    return [
        {
            "text": "The gNB-CU is a logical node hosting RRC and PDCP protocols.",
            "metadata": {"source": "38300-g30.docx", "chunk_index": 0, "chunk_size": 62},
            "chunk_id": 0,
        },
        {
            "text": "The gNB-DU is connected to the gNB-CU via the F1 interface.",
            "metadata": {"source": "38401-g30.docx", "chunk_index": 1, "chunk_size": 60},
            "chunk_id": 1,
        },
        {
            "text": "NG-RAN supports both SA and NSA deployment options for 5G.",
            "metadata": {"source": "38300-g30.docx", "chunk_index": 2, "chunk_size": 58},
            "chunk_id": 2,
        },
    ]


@pytest.fixture
def sample_chunks_with_embeddings(sample_chunks, mock_embedding):
    chunks = []
    for c in sample_chunks:
        ec = c.copy()
        ec["embedding"] = mock_embedding
        chunks.append(ec)
    return chunks


# ---------------------------------------------------------------------------
# Retrieved document fixtures (retriever output format)
# ---------------------------------------------------------------------------


@pytest.fixture
def retrieved_docs():
    return [
        {
            "text": "The gNB-CU is a logical node hosting RRC and PDCP protocols.",
            "source": "38300-g30.docx",
            "chunk_index": 0,
            "similarity": 0.92,
            "vector_similarity": 0.92,
            "reranker_score": 0.92,
        },
        {
            "text": "The gNB-DU connects to the gNB-CU via the F1 interface.",
            "source": "38401-g30.docx",
            "chunk_index": 1,
            "similarity": 0.87,
            "vector_similarity": 0.87,
            "reranker_score": 0.87,
        },
    ]


# ---------------------------------------------------------------------------
# Mocked core components
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_embedding_generator(mock_embedding, mock_embeddings):
    gen = MagicMock()
    gen.embedding_dim = 384
    gen.model_name = "mini"
    gen.generate_embedding.return_value = mock_embedding
    gen.generate_embeddings_batch.return_value = mock_embeddings
    gen.embed_chunks.side_effect = lambda chunks: [
        {**c, "embedding": mock_embedding} for c in chunks
    ]
    return gen


@pytest.fixture
def mock_vector_store(mock_embedding):
    store = MagicMock()
    store.get_stats.return_value = {
        "collection_name": "3gpp_specs",
        "total_chunks": 100,
        "persist_directory": "data/vectordb",
    }
    store.query.return_value = {
        "documents": [["chunk text 1", "chunk text 2"]],
        "metadatas": [
            [
                {"source": "38300-g30.docx", "chunk_index": 0},
                {"source": "38401-g30.docx", "chunk_index": 1},
            ]
        ],
        "distances": [[0.08, 0.13]],
    }
    return store


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.model = "llama3.2"
    llm.generate.return_value = "The gNB-CU handles upper-layer protocols."
    llm.stream.return_value = iter(["The ", "gNB-CU ", "handles ", "PDCP."])
    llm.is_available.return_value = True
    return llm


@pytest.fixture
def mock_retriever(retrieved_docs):
    ret = MagicMock()
    ret.retrieve.return_value = retrieved_docs
    ret.format_context.return_value = "\n".join(d["text"] for d in retrieved_docs)
    return ret

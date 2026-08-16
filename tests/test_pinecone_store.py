"""Tests for src/core/pinecone_store.py."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.core.pinecone_store import PineconeVectorStore


@pytest.fixture
def pinecone_index():
    return MagicMock()


@pytest.fixture
def pinecone_store(pinecone_index):
    return PineconeVectorStore(
        index_name="test-index",
        namespace="test-namespace",
        batch_size=2,
        index=pinecone_index,
    )


@pytest.fixture
def pinecone_chunks(mock_embedding):
    return [
        {
            "text": "The gNB-CU hosts RRC and PDCP protocols.",
            "embedding": mock_embedding,
            "metadata": {
                "source": "38300-g30.docx",
                "chunk_index": 0,
                "chunk_size": 42,
                "domain": "RAN",
                "generation": "5G",
                "spec_number": "38.300",
                "spec_title": "NR overall description",
                "release": "Rel-18",
                "version": "18.3.0",
                "section": "4.2",
                "section_title": "Architecture",
                "document_type": "TS",
            },
        },
        {
            "text": "The F1 interface connects gNB-CU and gNB-DU.",
            "embedding": mock_embedding,
            "metadata": {
                "source": "38401-g30.docx",
                "chunk_index": 1,
                "chunk_size": 44,
                "domain": "RAN",
                "generation": "5G",
                "spec_number": "38.401",
                "spec_title": "NG-RAN architecture",
            },
        },
        {
            "text": "NG-RAN supports standalone and non-standalone deployments.",
            "embedding": mock_embedding,
            "metadata": {
                "source": "38300-g30.docx",
                "chunk_index": 2,
                "chunk_size": 61,
                "domain": "RAN",
                "generation": "5G",
                "spec_number": "38.300",
                "spec_title": "NR overall description",
            },
        },
    ]


class TestPineconeAddChunks:
    def test_upsert_payload_batches_and_namespace(
        self, pinecone_store, pinecone_index, pinecone_chunks
    ):
        pinecone_store.add_chunks(pinecone_chunks)

        assert pinecone_index.upsert.call_count == 2
        first_call = pinecone_index.upsert.call_args_list[0]
        assert first_call.kwargs["namespace"] == "test-namespace"
        assert len(first_call.kwargs["vectors"]) == 2

        vector = first_call.kwargs["vectors"][0]
        assert vector["id"] == "38300-g30.docx:0"
        assert vector["values"] == pinecone_chunks[0]["embedding"]
        assert vector["metadata"]["text"] == pinecone_chunks[0]["text"]

    def test_metadata_preservation(self, pinecone_store, pinecone_index, pinecone_chunks):
        pinecone_store.add_chunks(pinecone_chunks[:1])

        metadata = pinecone_index.upsert.call_args.kwargs["vectors"][0]["metadata"]
        assert metadata["source"] == "38300-g30.docx"
        assert metadata["chunk_index"] == 0
        assert metadata["chunk_size"] == 42
        assert metadata["domain"] == "RAN"
        assert metadata["generation"] == "5G"
        assert metadata["spec_number"] == "38.300"
        assert metadata["spec_title"] == "NR overall description"
        assert metadata["release"] == "Rel-18"
        assert metadata["version"] == "18.3.0"
        assert metadata["section"] == "4.2"
        assert metadata["section_title"] == "Architecture"
        assert metadata["document_type"] == "TS"

    def test_empty_add_does_not_call_upsert(self, pinecone_store, pinecone_index):
        pinecone_store.add_chunks([])

        pinecone_index.upsert.assert_not_called()


class TestPineconeQuery:
    def test_query_result_normalization(self, pinecone_store, pinecone_index, mock_embedding):
        pinecone_index.query.return_value = SimpleNamespace(
            matches=[
                SimpleNamespace(
                    id="match-1",
                    score=0.91,
                    metadata={
                        "text": "matched chunk",
                        "source": "38300-g30.docx",
                        "chunk_index": 0,
                        "domain": "RAN",
                    },
                )
            ]
        )

        result = pinecone_store.query(mock_embedding, n_results=1)

        assert result["documents"] == [["matched chunk"]]
        assert result["metadatas"] == [
            [{"source": "38300-g30.docx", "chunk_index": 0, "domain": "RAN"}]
        ]
        assert result["distances"][0][0] == pytest.approx(0.09)
        pinecone_index.query.assert_called_once_with(
            vector=mock_embedding,
            top_k=1,
            include_metadata=True,
            namespace="test-namespace",
        )

    def test_metadata_filter_forwarding(
        self, pinecone_store, pinecone_index, mock_embedding
    ):
        pinecone_index.query.return_value = {"matches": []}
        where_filter = {"domain": {"$eq": "RAN"}}

        pinecone_store.query(mock_embedding, n_results=3, where_filter=where_filter)

        assert pinecone_index.query.call_args.kwargs["filter"] == where_filter


class TestPineconeStatsAndClear:
    def test_get_stats_uses_namespace_count(self, pinecone_store, pinecone_index):
        pinecone_index.describe_index_stats.return_value = {
            "total_vector_count": 25,
            "namespaces": {
                "test-namespace": {"vector_count": 12},
                "other-namespace": {"vector_count": 13},
            },
        }

        stats = pinecone_store.get_stats()

        assert stats == {
            "index_name": "test-index",
            "namespace": "test-namespace",
            "total_chunks": 12,
            "total_vector_count": 25,
        }

    def test_clear_deletes_only_namespace(self, pinecone_store, pinecone_index):
        pinecone_store.clear()

        pinecone_index.delete.assert_called_once_with(
            delete_all=True,
            namespace="test-namespace",
        )


class TestPineconeConfiguration:
    def test_missing_api_key_raises_clear_error(self):
        with pytest.raises(ValueError, match="PINECONE_API_KEY"):
            PineconeVectorStore(index_name="test-index", api_key="")

    def test_missing_index_name_raises_clear_error(self):
        with pytest.raises(ValueError, match="PINECONE_INDEX_NAME"):
            PineconeVectorStore(index_name="", api_key="test-key")

    def test_client_injection_gets_index(self):
        client = MagicMock()

        store = PineconeVectorStore(
            index_name="test-index",
            namespace="test-namespace",
            client=client,
        )

        client.Index.assert_called_once_with("test-index")
        assert store.index == client.Index.return_value

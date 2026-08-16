"""Tests for scripts/build_pinecone_index.py."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from scripts import build_pinecone_index as indexer


def _chunk(text: str, source: str = "38300-h30.docx", chunk_index: int = 0) -> dict:
    return {
        "text": text,
        "metadata": {
            "source": source,
            "chunk_index": chunk_index,
            "chunk_size": len(text),
        },
        "chunk_id": chunk_index,
    }


def _chunks(count: int, source: str = "38300-h30.docx") -> list[dict]:
    return [_chunk(f"chunk text {i}", source=source, chunk_index=i) for i in range(count)]


def _embedding_side_effect(texts):
    return [[float(i)] for i, _ in enumerate(texts)]


class TestPineconeIndexHelpers:
    def test_deterministic_ids_use_source_and_chunk_index(self):
        chunk = _chunk("hello", chunk_index=7)
        prepared_once = indexer.prepare_chunk_for_index(chunk, 7, Path("38300-h30.docx"))
        prepared_twice = indexer.prepare_chunk_for_index(chunk, 7, Path("38300-h30.docx"))

        assert prepared_once["id"] == "38300-h30.docx:7"
        assert prepared_once["id"] == prepared_twice["id"]

    def test_deterministic_id_falls_back_to_content_hash(self):
        chunk = {"text": "same text", "metadata": {}}

        first = indexer.deterministic_vector_id(chunk)
        second = indexer.deterministic_vector_id(chunk)

        assert first.startswith("chunk:")
        assert first == second

    def test_metadata_creation_preserves_required_and_available_optional_fields(self):
        chunk = {
            "text": "body",
            "metadata": {
                "source": "38300-h30.docx",
                "chunk_index": 3,
                "chunk_size": 4,
                "domain": "RAN",
                "generation": "5G",
                "spec_number": "38.300",
                "spec_title": "NR overall description",
                "release": "Rel-17",
                "version": "17.3.0",
                "section": "4.2",
                "section_title": "Functional Split",
            },
        }

        metadata = indexer.build_chunk_metadata(chunk, 3, Path("38300-h30.docx"))

        assert metadata["source"] == "38300-h30.docx"
        assert metadata["chunk_index"] == 3
        assert metadata["chunk_size"] == 4
        assert metadata["domain"] == "RAN"
        assert metadata["generation"] == "5G"
        assert metadata["spec_number"] == "38.300"
        assert metadata["spec_title"] == "NR overall description"
        assert metadata["release"] == "Rel-17"
        assert metadata["version"] == "17.3.0"
        assert metadata["section"] == "4.2"
        assert metadata["section_title"] == "Functional Split"
        assert "document_type" not in metadata

    def test_apply_chunk_limit(self):
        chunks = _chunks(5)

        assert len(indexer.apply_chunk_limit(chunks, limit=3, already_seen=1)) == 2
        assert len(indexer.apply_chunk_limit(chunks, limit=None, already_seen=0)) == 5


class TestPineconeIndexBatching:
    def test_embedding_and_upload_batching(self):
        chunks = [
            indexer.prepare_chunk_for_index(chunk, i, Path("38300-h30.docx"))
            for i, chunk in enumerate(_chunks(5))
        ]
        generator = MagicMock()
        generator.generate_embeddings_batch.side_effect = _embedding_side_effect
        store = MagicMock()

        uploaded = indexer.embed_and_upload_chunks(
            chunks,
            embedding_generator=generator,
            vector_store=store,
            embedding_batch_size=2,
            retry_delay_seconds=0,
        )

        assert uploaded == 5
        assert generator.generate_embeddings_batch.call_count == 3
        assert [len(call.args[0]) for call in generator.generate_embeddings_batch.call_args_list] == [
            2,
            2,
            1,
        ]
        assert [len(call.args[0]) for call in store.add_chunks.call_args_list] == [2, 2, 1]

    def test_retry_behavior_for_transient_embedding_error(self, monkeypatch):
        chunks = [
            indexer.prepare_chunk_for_index(chunk, i, Path("38300-h30.docx"))
            for i, chunk in enumerate(_chunks(1))
        ]
        monkeypatch.setattr(indexer.time, "sleep", MagicMock())
        generator = MagicMock()
        generator.generate_embeddings_batch.side_effect = [
            RuntimeError("temporary"),
            [[0.1]],
        ]
        store = MagicMock()

        uploaded = indexer.embed_and_upload_chunks(
            chunks,
            embedding_generator=generator,
            vector_store=store,
            embedding_batch_size=1,
            retry_delay_seconds=0,
        )

        assert uploaded == 1
        assert generator.generate_embeddings_batch.call_count == 2
        store.add_chunks.assert_called_once()


class TestBuildPineconeIndex:
    def test_dry_run_processes_without_external_calls(self):
        processor = MagicMock()
        processor.process_document.return_value = _chunks(3)
        generator = MagicMock()
        store = MagicMock()

        stats = indexer.build_pinecone_index(
            files=[Path("38300-h30.docx")],
            processor=processor,
            embedding_generator=generator,
            vector_store=store,
            dry_run=True,
        )

        assert stats.processed_documents == 1
        assert stats.chunks_processed == 3
        assert stats.vectors_uploaded == 0
        generator.generate_embeddings_batch.assert_not_called()
        store.add_chunks.assert_not_called()

    def test_limit_behavior_across_documents(self):
        processor = MagicMock()
        processor.process_document.side_effect = [
            _chunks(4, source="38300-h30.docx"),
            _chunks(4, source="38401-h30.docx"),
        ]

        stats = indexer.build_pinecone_index(
            files=[Path("38300-h30.docx"), Path("38401-h30.docx")],
            processor=processor,
            dry_run=True,
            limit=5,
        )

        assert stats.processed_documents == 2
        assert stats.chunks_processed == 5
        assert processor.process_document.call_count == 2

    def test_duplicate_safe_ids_are_stable_across_rebuilds(self):
        first_processor = MagicMock()
        second_processor = MagicMock()
        first_processor.process_document.return_value = _chunks(2)
        second_processor.process_document.return_value = _chunks(2)
        generator = MagicMock()
        generator.generate_embeddings_batch.side_effect = _embedding_side_effect
        store = MagicMock()

        indexer.build_pinecone_index(
            files=[Path("38300-h30.docx")],
            processor=first_processor,
            embedding_generator=generator,
            vector_store=store,
            embedding_batch_size=10,
        )
        first_ids = [chunk["id"] for chunk in store.add_chunks.call_args.args[0]]

        store.reset_mock()
        indexer.build_pinecone_index(
            files=[Path("38300-h30.docx")],
            processor=second_processor,
            embedding_generator=generator,
            vector_store=store,
            embedding_batch_size=10,
        )
        second_ids = [chunk["id"] for chunk in store.add_chunks.call_args.args[0]]

        assert first_ids == second_ids == ["38300-h30.docx:0", "38300-h30.docx:1"]

    def test_upload_uses_mocked_openai_and_pinecone(self):
        processor = MagicMock()
        processor.process_document.return_value = _chunks(2)
        generator = MagicMock()
        generator.generate_embeddings_batch.side_effect = _embedding_side_effect
        store = MagicMock()

        stats = indexer.build_pinecone_index(
            files=[Path("38300-h30.docx")],
            processor=processor,
            embedding_generator=generator,
            vector_store=store,
            embedding_batch_size=2,
            retry_delay_seconds=0,
        )

        assert stats.vectors_uploaded == 2
        generator.generate_embeddings_batch.assert_called_once()
        store.add_chunks.assert_called_once()
        uploaded_chunks = store.add_chunks.call_args.args[0]
        assert uploaded_chunks[0]["embedding"] == [0.0]
        assert uploaded_chunks[0]["metadata"]["spec_number"] == "38.300"

    def test_processing_failure_recorded(self):
        processor = MagicMock()
        processor.process_document.side_effect = RuntimeError("bad doc")

        stats = indexer.build_pinecone_index(
            files=[Path("bad.docx")],
            processor=processor,
            dry_run=True,
        )

        assert stats.failed_documents == 1
        assert "bad doc" in stats.failures[0]


class TestValidation:
    def test_batch_size_must_be_positive(self):
        with pytest.raises(ValueError, match="batch_size"):
            list(indexer.batched(_chunks(1), 0))

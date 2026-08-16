#!/usr/bin/env python3
"""
Build the Pinecone vector index from raw 3GPP spec files.

Pipeline:
    raw 3GPP documents
    -> existing DocumentProcessor chunking
    -> OpenAI embeddings
    -> Pinecone upserts

This script deliberately sits beside scripts/build_index.py so the existing
Chroma indexing path remains intact.
"""

import argparse
import logging
import sys
import time
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence

# Add repo root to path so this script works when run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.build_index import enrich_chunks, find_spec_files
from src.config import settings
from src.core.document_processor import DocumentProcessor
from src.core.openai_embeddings import OpenAIEmbeddingGenerator
from src.core.pinecone_store import PineconeVectorStore
from src.core.spec_catalog import catalog_summary, infer_spec_from_filename

logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")
DEFAULT_EMBEDDING_BATCH_SIZE = 100
DEFAULT_UPSERT_BATCH_SIZE = 100
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY_SECONDS = 2.0

REQUIRED_METADATA_FIELDS = (
    "source",
    "chunk_index",
    "chunk_size",
    "domain",
    "generation",
    "spec_number",
    "spec_title",
)

OPTIONAL_METADATA_FIELDS = (
    "release",
    "version",
    "section",
    "section_title",
    "document_type",
)


@dataclass
class IndexingStats:
    """Summary counters for one indexing run."""

    processed_documents: int = 0
    failed_documents: int = 0
    chunks_processed: int = 0
    vectors_uploaded: int = 0
    failures: List[str] = field(default_factory=list)


def deterministic_vector_id(chunk: dict) -> str:
    """Return a stable Pinecone vector ID for a chunk.

    The normal path uses source + chunk_index, which makes rebuilds replace
    existing vectors instead of creating duplicates. A content hash is only a
    deterministic fallback for malformed chunks that lack those fields.
    """
    metadata = chunk.get("metadata", {}) or {}
    source = metadata.get("source")
    chunk_index = metadata.get("chunk_index")
    if source not in (None, "") and chunk_index not in (None, ""):
        return f"{source}:{chunk_index}"

    digest = sha256(chunk.get("text", "").encode("utf-8")).hexdigest()[:16]
    return f"chunk:{digest}"


def build_chunk_metadata(chunk: dict, position: int, file_path: Optional[Path] = None) -> dict:
    """Create Pinecone metadata for one chunk without inventing optional fields."""
    text = chunk.get("text", "")
    source_metadata = dict(chunk.get("metadata", {}) or {})
    source = source_metadata.get("source") or (file_path.name if file_path else "unknown")
    chunk_index = source_metadata.get("chunk_index", position)

    metadata = {
        "source": source,
        "chunk_index": chunk_index,
        "chunk_size": source_metadata.get("chunk_size", len(text)),
        "domain": source_metadata.get("domain", "unknown"),
        "generation": source_metadata.get("generation", "unknown"),
        "spec_number": source_metadata.get("spec_number", "unknown"),
        "spec_title": source_metadata.get("spec_title", "unknown"),
    }

    for field_name in OPTIONAL_METADATA_FIELDS:
        value = source_metadata.get(field_name)
        if value not in (None, ""):
            metadata[field_name] = value

    return metadata


def prepare_chunk_for_index(chunk: dict, position: int, file_path: Optional[Path] = None) -> dict:
    """Return a chunk copy with normalized metadata and deterministic ID."""
    prepared = {
        "text": chunk["text"],
        "metadata": build_chunk_metadata(chunk, position, file_path),
    }
    if "chunk_id" in chunk:
        prepared["chunk_id"] = chunk["chunk_id"]
    prepared["id"] = deterministic_vector_id(prepared)
    return prepared


def apply_chunk_limit(chunks: List[dict], limit: Optional[int], already_seen: int) -> List[dict]:
    """Apply a run-wide chunk limit to the current document's chunks."""
    if limit is None:
        return chunks
    remaining = max(limit - already_seen, 0)
    return chunks[:remaining]


def batched(items: Sequence[dict], batch_size: int) -> Iterable[Sequence[dict]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def retry_operation(
    operation: Callable[[], object],
    description: str,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_delay_seconds: float = DEFAULT_RETRY_DELAY_SECONDS,
):
    """Retry a transient external operation."""
    for attempt in range(1, max_retries + 1):
        try:
            return operation()
        except Exception as exc:
            if attempt >= max_retries:
                raise
            logger.warning(
                "%s failed: %s (attempt %s/%s); retrying in %.1fs",
                description,
                exc,
                attempt,
                max_retries,
                retry_delay_seconds,
            )
            time.sleep(retry_delay_seconds)


def embed_and_upload_chunks(
    chunks: List[dict],
    embedding_generator,
    vector_store,
    embedding_batch_size: int,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_delay_seconds: float = DEFAULT_RETRY_DELAY_SECONDS,
) -> int:
    """Embed chunks in batches and upload each embedded batch to Pinecone."""
    uploaded = 0
    for batch_number, chunk_batch in enumerate(batched(chunks, embedding_batch_size), 1):
        texts = [chunk["text"] for chunk in chunk_batch]
        embeddings = retry_operation(
            lambda: embedding_generator.generate_embeddings_batch(texts),
            description=f"OpenAI embedding batch {batch_number}",
            max_retries=max_retries,
            retry_delay_seconds=retry_delay_seconds,
        )

        if len(embeddings) != len(chunk_batch):
            raise RuntimeError(
                f"Embedding batch {batch_number} returned {len(embeddings)} vectors "
                f"for {len(chunk_batch)} chunks"
            )

        embedded_batch = []
        for chunk, embedding in zip(chunk_batch, embeddings):
            embedded = chunk.copy()
            embedded["embedding"] = embedding
            embedded_batch.append(embedded)

        retry_operation(
            lambda: vector_store.add_chunks(embedded_batch),
            description=f"Pinecone upsert batch {batch_number}",
            max_retries=max_retries,
            retry_delay_seconds=retry_delay_seconds,
        )
        uploaded += len(embedded_batch)
        print(f"    uploaded batch {batch_number}: {len(embedded_batch)} vectors")

    return uploaded


def build_pinecone_index(
    files: Sequence[Path],
    processor: DocumentProcessor,
    embedding_generator=None,
    vector_store=None,
    dry_run: bool = False,
    limit: Optional[int] = None,
    embedding_batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_delay_seconds: float = DEFAULT_RETRY_DELAY_SECONDS,
) -> IndexingStats:
    """Process files and optionally upload vectors to Pinecone."""
    stats = IndexingStats()

    for i, file_path in enumerate(files, 1):
        if limit is not None and stats.chunks_processed >= limit:
            break

        file_path = Path(file_path)
        entry = infer_spec_from_filename(file_path.name)
        label = f"TS {entry['spec_number']}" if entry else file_path.name
        print(f"[{i}/{len(files)}] Processing {label} ({file_path.name})...")

        try:
            chunks = processor.process_document(file_path)
            chunks = enrich_chunks(chunks, file_path)
            chunks = [
                prepare_chunk_for_index(chunk, position, file_path)
                for position, chunk in enumerate(chunks)
                if chunk.get("text", "").strip()
            ]
            chunks = apply_chunk_limit(chunks, limit, stats.chunks_processed)
        except Exception as exc:
            stats.failed_documents += 1
            message = f"{file_path}: {exc}"
            stats.failures.append(message)
            logger.error("Failed to process %s: %s", file_path, exc)
            continue

        if not chunks:
            print("    no chunks to index")
            stats.processed_documents += 1
            continue

        stats.processed_documents += 1
        stats.chunks_processed += len(chunks)
        print(f"    chunks prepared: {len(chunks)}")

        if dry_run:
            continue

        if embedding_generator is None or vector_store is None:
            raise ValueError("embedding_generator and vector_store are required unless dry_run=True")

        try:
            uploaded = embed_and_upload_chunks(
                chunks,
                embedding_generator=embedding_generator,
                vector_store=vector_store,
                embedding_batch_size=embedding_batch_size,
                max_retries=max_retries,
                retry_delay_seconds=retry_delay_seconds,
            )
            stats.vectors_uploaded += uploaded
        except Exception as exc:
            stats.failed_documents += 1
            message = f"{file_path}: upload failed: {exc}"
            stats.failures.append(message)
            logger.error("Failed to upload chunks for %s: %s", file_path, exc)

    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Pinecone index from 3GPP spec files")
    parser.add_argument("--generation", choices=["5G", "LTE"], help="Only index this generation")
    parser.add_argument("--domain", choices=["RAN", "CORE"], help="Only index this domain")
    parser.add_argument("--file", type=Path, help="Index a single file instead of scanning data/raw/")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR, help="Raw data directory")
    parser.add_argument("--limit", type=int, help="Maximum number of chunks to process")
    parser.add_argument("--dry-run", action="store_true", help="Process and validate without API calls or writes")
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=DEFAULT_EMBEDDING_BATCH_SIZE,
        help="Texts per OpenAI embeddings request",
    )
    parser.add_argument(
        "--upsert-batch-size",
        type=int,
        default=DEFAULT_UPSERT_BATCH_SIZE,
        help="Vectors per Pinecone upsert request",
    )
    parser.add_argument("--index-name", default=settings.pinecone_index_name, help="Pinecone index name")
    parser.add_argument("--namespace", default=settings.pinecone_namespace, help="Pinecone namespace")
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    parser.add_argument("--retry-delay", type=float, default=DEFAULT_RETRY_DELAY_SECONDS)
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be greater than zero")

    print(f"\n{'=' * 60}")
    print("3GPP RAG Assistant - Pinecone Index Builder")
    print(f"{'=' * 60}\n")
    print(catalog_summary())
    print()
    if args.dry_run:
        print("[dry-run] OpenAI and Pinecone clients will not be initialized.")

    if args.file:
        files = [args.file]
    else:
        files = find_spec_files(
            root=args.raw_dir,
            domain=args.domain,
            generation=args.generation,
        )

    if not files:
        print(
            f"No .doc/.docx files found under {args.raw_dir}.\n"
            "Run first: python scripts/download_specs.py"
        )
        sys.exit(1)

    print(f"Files discovered: {len(files)}")
    if args.limit:
        print(f"Chunk limit     : {args.limit}")
    print(f"Namespace       : {args.namespace}")
    print(f"Index name      : {args.index_name or '(from environment)'}")
    print()

    processor = DocumentProcessor(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    embedding_generator = None
    vector_store = None
    if not args.dry_run:
        embedding_generator = OpenAIEmbeddingGenerator(
            model_name=settings.openai_embedding_model,
            batch_size=args.embedding_batch_size,
        )
        vector_store = PineconeVectorStore(
            index_name=args.index_name,
            namespace=args.namespace,
            batch_size=args.upsert_batch_size,
        )

    stats = build_pinecone_index(
        files=files,
        processor=processor,
        embedding_generator=embedding_generator,
        vector_store=vector_store,
        dry_run=args.dry_run,
        limit=args.limit,
        embedding_batch_size=args.embedding_batch_size,
        max_retries=args.max_retries,
        retry_delay_seconds=args.retry_delay,
    )

    print(f"\n{'=' * 60}")
    print("Pinecone Index Build Complete" if not args.dry_run else "Pinecone Dry Run Complete")
    print(f"{'=' * 60}")
    print(f"Documents processed : {stats.processed_documents}")
    print(f"Document failures   : {stats.failed_documents}")
    print(f"Chunks processed    : {stats.chunks_processed}")
    print(f"Vectors uploaded    : {stats.vectors_uploaded}")
    if stats.failures:
        print("Failures:")
        for failure in stats.failures:
            print(f"  - {failure}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()

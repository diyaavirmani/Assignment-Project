#!/usr/bin/env python3
"""
Build the vector index using ChromaDB's built-in ONNX embeddings.

This avoids torch/sentence-transformers entirely — making the deployment
much lighter for Streamlit Cloud (~80MB ONNX model vs ~2GB torch stack).

Usage:
    python scripts/build_index_cloud.py          # build index
    python scripts/build_index_cloud.py --clear   # rebuild from scratch
"""
import os
import sys
import time
import logging
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.document_processor import DocumentProcessor
from src.core.spec_catalog import (
    CATALOG,
    infer_spec_from_filename,
    infer_release_from_filename,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# Where to store the cloud-optimized index
VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "./data/vectordb_cloud")
COLLECTION_NAME = "3gpp_specs"
DATA_DIR = "./data/raw"


def find_spec_files() -> list:
    """Walk data/raw tree and find all .docx/.doc/.pdf files."""
    data_path = Path(DATA_DIR)
    files = []
    for ext in ("*.docx", "*.doc", "*.pdf"):
        files.extend(data_path.rglob(ext))

    # Deduplicate: keep latest version per spec number, skip companion files
    spec_map = {}
    for f in sorted(files):
        name = f.stem.lower()
        if "-cl" in name or "-rm" in name:
            continue
        entry = infer_spec_from_filename(f.name)
        if entry:
            spec_num = entry["spec_number"]
            spec_map[spec_num] = f  # last one wins (sorted = latest)
        else:
            spec_map[f.name] = f

    return list(spec_map.values())


def main():
    parser = argparse.ArgumentParser(description="Build cloud-optimized vector index")
    parser.add_argument("--clear", action="store_true", help="Clear existing index first")
    args = parser.parse_args()

    import chromadb

    # Initialize ChromaDB with its built-in embedding function (ONNX all-MiniLM-L6-v2)
    client = chromadb.PersistentClient(path=VECTOR_DB_PATH)

    if args.clear:
        try:
            client.delete_collection(COLLECTION_NAME)
            logger.info(f"Deleted existing collection '{COLLECTION_NAME}'")
        except Exception:
            pass

    # Create collection — ChromaDB auto-uses its default ONNX embedding function
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    existing = collection.count()
    if existing > 0 and not args.clear:
        logger.info(f"Collection already has {existing} chunks. Resuming (skipping indexed specs)...")

    dp = DocumentProcessor()
    files = find_spec_files()
    logger.info(f"Found {len(files)} spec files to process")

    # Build set of already-indexed spec numbers for resume
    indexed_specs = set()
    if existing > 0 and not args.clear:
        for entry_cat in CATALOG:
            result = collection.get(limit=1, where={"spec_number": entry_cat["spec_number"]}, include=[])
            if result and result.get("ids"):
                indexed_specs.add(entry_cat["spec_number"])
        logger.info(f"Already indexed: {len(indexed_specs)} specs — skipping those")

    total_chunks = 0
    for i, fpath in enumerate(files):
        entry = infer_spec_from_filename(fpath.name)
        # Release parsed from the filename version suffix (h30 -> Rel-17);
        # kept consistent with scripts/build_index.py (see ADR-008).
        release = infer_release_from_filename(fpath.name) or "unknown"
        spec_label = f"TS {entry['spec_number']}" if entry else fpath.name

        # Skip already-indexed specs on resume
        if entry and entry["spec_number"] in indexed_specs:
            logger.info(f"[{i+1}/{len(files)}] Skipping {spec_label} (already indexed)")
            continue

        logger.info(f"[{i+1}/{len(files)}] Processing {spec_label} ({fpath.name})...")

        try:
            chunks = dp.process_document(str(fpath))
        except Exception as e:
            logger.error(f"  Failed to process {fpath.name}: {e}")
            continue

        if not chunks:
            logger.warning(f"  No chunks from {fpath.name}")
            continue

        # Prepare batch data
        ids = []
        documents = []
        metadatas = []

        for j, chunk in enumerate(chunks):
            chunk_id = f"{fpath.stem}_{j}_{total_chunks + j}"
            text = chunk.get("text", "")
            if not text.strip():
                continue

            meta = {
                "source": fpath.name,
                "chunk_index": j,
                "release": release,
                "domain": entry["domain"] if entry else "unknown",
                "generation": entry["generation"] if entry else "unknown",
                "spec_number": entry["spec_number"] if entry else "unknown",
                "spec_title": entry["title"] if entry else "unknown",
            }

            ids.append(chunk_id)
            documents.append(text)
            metadatas.append(meta)

        # Add in batches of 500 (ChromaDB handles embedding internally)
        batch_size = 500
        for b_start in range(0, len(ids), batch_size):
            b_end = min(b_start + batch_size, len(ids))
            collection.add(
                ids=ids[b_start:b_end],
                documents=documents[b_start:b_end],
                metadatas=metadatas[b_start:b_end],
            )
            logger.info(f"  Added batch {b_start//batch_size + 1} ({b_end - b_start} chunks)")

        total_chunks += len(ids)
        logger.info(f"  Indexed {len(ids)} chunks from {spec_label}")

    print(f"\n{'='*60}")
    print(f"Cloud Index Build Complete!")
    print(f"{'='*60}")
    print(f"Files processed : {len(files)}")
    print(f"Total chunks    : {total_chunks}")
    print(f"Total in store  : {collection.count()}")
    print(f"Index path      : {VECTOR_DB_PATH}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

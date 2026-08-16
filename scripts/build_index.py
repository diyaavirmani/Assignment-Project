"""
Build (or rebuild) the ChromaDB vector index from raw 3GPP spec files.

The script scans data/raw/ for .doc and .docx files, infers each file's
catalog metadata (domain, generation, spec_number, spec_title) using
src.core.spec_catalog.infer_spec_from_filename, processes the text into
chunks, embeds them locally, and stores them in ChromaDB with full metadata.

Usage:
    # Index everything in data/raw/
    python scripts/build_index.py

    # Index only 5G RAN specs
    python scripts/build_index.py --generation 5G --domain RAN

    # Wipe the collection and rebuild from scratch
    python scripts/build_index.py --clear

    # Index a single file (useful for incremental updates)
    python scripts/build_index.py --file data/raw/5G/RAN/38300.docx

    # Dry-run: show what would be indexed without writing anything
    python scripts/build_index.py --dry-run
"""
import argparse
import json
import logging
import sys
from pathlib import Path

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.corpus_config import DEFAULT_CORPUS
from src.core.document_processor import DocumentProcessor
from src.core.embeddings import LocalEmbeddingGenerator
from src.core.vector_store import VectorStore
from src.core.spec_catalog import (
    infer_spec_from_filename,
    infer_release_from_filename,
    catalog_summary,
)

RAW_DIR = Path("data/raw")
CHUNKS_CACHE = Path("data/processed/chunks.json")

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Suffixes that indicate a change-log or revision-marks companion file,
# not the actual spec content.
_SKIP_SUFFIXES = ("-cl", "-rm", "-cover", "cover sheet")

# Prefixes used for 3GPP meeting contribution documents (not specs).
_SKIP_PREFIXES = ("r2-", "r3-", "rp-", "r1-", "s2-", "c1-")


def _is_spec_file(path: Path) -> bool:
    """Return True if this file looks like an actual spec, not a companion doc."""
    name = path.name.lower()
    for pfx in _SKIP_PREFIXES:
        if name.startswith(pfx):
            return False
    stem = path.stem.lower()
    for sfx in _SKIP_SUFFIXES:
        if sfx in stem:
            return False
    return True


def find_spec_files(
    root: Path = RAW_DIR,
    domain: str = None,
    generation: str = None,
) -> list[Path]:
    """Recursively find .doc/.docx spec files under root.

    - Skips companion files (change-logs, revision marks, cover sheets).
    - For each spec number, returns only the **latest** file (sorted by name
      descending so the highest version string wins).
    - If domain or generation is set, only files inside a matching
      subdirectory (data/raw/<generation>/<domain>/) are considered.
    """
    if not root.exists():
        logger.warning(f"Raw data directory not found: {root}")
        return []

    candidates: list[Path] = []
    for ext in ("*.docx", "*.doc"):
        for f in root.rglob(ext):
            if not _is_spec_file(f):
                continue
            parts = f.relative_to(root).parts
            if generation and len(parts) >= 1 and parts[0] != generation:
                continue
            if domain and len(parts) >= 2 and parts[1] != domain:
                continue
            candidates.append(f)

    # Group by inferred spec number; keep only the latest file per spec.
    by_spec: dict[str, list[Path]] = {}
    unrecognised: list[Path] = []
    for f in candidates:
        entry = infer_spec_from_filename(f.name)
        if entry:
            key = entry["spec_number"]
            by_spec.setdefault(key, []).append(f)
        else:
            unrecognised.append(f)

    selected: list[Path] = []
    for spec_num, paths in by_spec.items():
        # Sort descending by filename — highest version suffix wins.
        latest = sorted(paths, key=lambda p: p.name.lower(), reverse=True)[0]
        selected.append(latest)

    # Include unrecognised files so nothing is silently dropped.
    selected.extend(unrecognised)

    return sorted(selected)


def enrich_chunks(chunks: list, file_path: Path) -> list:
    """Add catalog metadata to every chunk from a given file.

    Catalog data is sourced from DEFAULT_CORPUS (the 3GPP CorpusConfig), which
    wraps spec_catalog.CATALOG.  This is the production call site where the
    corpus-specific catalog plugs into the indexing pipeline — swap DEFAULT_CORPUS
    for a different CorpusConfig and the enrichment picks up that corpus's fields.
    """
    inferred = infer_spec_from_filename(file_path.name)
    # Route through DEFAULT_CORPUS so the config object is the live source of
    # catalog truth, not a direct reference to spec_catalog.CATALOG.
    corpus_entry = DEFAULT_CORPUS.get_entry(inferred["spec_number"]) if inferred else None
    # Release parsed from the filename's version suffix (e.g. h30 -> Rel-17).
    # Track C groundwork (ADR-008): lets citations name their release, and is
    # the metadata a future version-aware index will filter on. Takes effect
    # at the next index rebuild.
    release = infer_release_from_filename(file_path.name) or "unknown"
    for chunk in chunks:
        meta = chunk.setdefault("metadata", {})
        meta["release"] = release
        if corpus_entry:
            meta["domain"]      = corpus_entry["domain"]
            meta["generation"]  = corpus_entry["generation"]
            meta["spec_number"] = corpus_entry["spec_number"]
            meta["spec_title"]  = corpus_entry["title"]
        else:
            meta.setdefault("domain",      "unknown")
            meta.setdefault("generation",  "unknown")
            meta.setdefault("spec_number", "unknown")
            meta.setdefault("spec_title",  "unknown")
    return chunks


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build ChromaDB index from 3GPP spec files")
    p.add_argument("--generation", choices=["5G", "LTE"],
                   help="Only index specs of this generation")
    p.add_argument("--domain", choices=["RAN", "CORE"],
                   help="Only index specs of this domain")
    p.add_argument("--file", type=Path,
                   help="Index a single file instead of scanning data/raw/")
    p.add_argument("--clear", action="store_true",
                   help="Delete all existing data from the collection before indexing")
    p.add_argument("--dry-run", action="store_true",
                   help="Show what would be indexed without writing to ChromaDB")
    p.add_argument("--save-chunks", action="store_true",
                   help=f"Save all chunks to {CHUNKS_CACHE} (useful for debugging)")
    p.add_argument("--embedding-model", default="bge-small",
                   choices=["mini", "mpnet", "bge-small", "bge-base"],
                   help="Sentence-transformer model for embeddings (default: bge-small)")
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    print(f"\n{'='*60}")
    print("3GPP RAG Assistant — Index Builder")
    print(f"{'='*60}\n")
    # DEFAULT_CORPUS is the CorpusConfig for 3GPP — the seam where corpus-specific
    # catalog and cleaning config enter the pipeline.  See corpus_config.py.
    print(f"Corpus : {DEFAULT_CORPUS.summary()}")
    print(f"Filters: {DEFAULT_CORPUS.filter_dimensions}")
    print()
    print(catalog_summary())
    print()

    # ------------------------------------------------------------------
    # Discover files to index
    # ------------------------------------------------------------------
    if args.file:
        files = [args.file]
    else:
        files = find_spec_files(
            root=RAW_DIR,
            domain=args.domain,
            generation=args.generation,
        )

    if not files:
        print(
            f"No .doc/.docx files found under {RAW_DIR}.\n"
            "Run first: python scripts/download_specs.py"
        )
        sys.exit(1)

    print(f"Files to index: {len(files)}")
    for f in files:
        entry = infer_spec_from_filename(f.name)
        tag = f"{entry['generation']} {entry['domain']} TS {entry['spec_number']}" if entry else "unknown"
        print(f"  {f.name:40s}  [{tag}]")
    print()

    if args.dry_run:
        print("[dry-run] No data will be written.")
        return

    # ------------------------------------------------------------------
    # Initialise components
    # ------------------------------------------------------------------
    processor = DocumentProcessor()
    generator = LocalEmbeddingGenerator(model_name=args.embedding_model)
    store = VectorStore()

    if args.clear:
        print("Clearing existing vector store...")
        store.clear()
        print("✓ Collection cleared\n")

    # ------------------------------------------------------------------
    # Process each file
    # ------------------------------------------------------------------
    all_chunks: list = []
    total_indexed = 0

    for i, file_path in enumerate(files, 1):
        entry = infer_spec_from_filename(file_path.name)
        label = f"TS {entry['spec_number']}" if entry else file_path.name
        print(f"[{i}/{len(files)}] Processing {label} ({file_path.name})...")

        try:
            chunks = processor.process_document(str(file_path))
        except Exception as exc:
            logger.error(f"  ✗ Failed to process {file_path.name}: {exc}")
            continue

        if not chunks:
            logger.warning(f"  ✗ No chunks extracted from {file_path.name}")
            continue

        # Enrich with catalog metadata
        chunks = enrich_chunks(chunks, file_path)
        print(f"    {len(chunks)} chunks extracted")

        # Embed
        print(f"    Generating embeddings...")
        embedded = generator.embed_chunks(chunks)

        # Store
        store.add_chunks(embedded)
        total_indexed += len(embedded)
        print(f"    ✓ {len(embedded)} chunks indexed")

        if args.save_chunks:
            all_chunks.extend(chunks)

    # ------------------------------------------------------------------
    # Optionally save chunk cache
    # ------------------------------------------------------------------
    if args.save_chunks and all_chunks:
        CHUNKS_CACHE.parent.mkdir(parents=True, exist_ok=True)
        with open(CHUNKS_CACHE, "w") as f:
            # Strip embeddings from cache (they're stored in ChromaDB)
            cache = [{k: v for k, v in c.items() if k != "embedding"} for c in all_chunks]
            json.dump(cache, f, indent=2)
        print(f"\nChunk cache saved to {CHUNKS_CACHE}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    stats = store.get_stats()
    print(f"\n{'='*60}")
    print("Index Build Complete!")
    print(f"{'='*60}")
    print(f"Files processed : {len(files)}")
    print(f"Chunks added    : {total_indexed}")
    print(f"Total in store  : {stats['total_chunks']}")
    print(f"Collection      : {stats['collection_name']}")
    print(f"Location        : {stats['persist_directory']}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

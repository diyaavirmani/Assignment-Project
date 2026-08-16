"""Build vector index forcing CPU for stable embedding (avoids MPS stalls)."""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.document_processor import DocumentProcessor
from src.core.embeddings import LocalEmbeddingGenerator
from src.core.vector_store import VectorStore
from src.core.spec_catalog import infer_spec_from_filename

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")

# Same skip logic as build_index.py
_SKIP_SUFFIXES = ("-cl", "-rm", "-cover", "cover sheet")
_SKIP_PREFIXES = ("r2-", "r3-", "rp-", "r1-", "s2-", "c1-")


def _is_spec_file(path: Path) -> bool:
    name = path.name.lower()
    for pfx in _SKIP_PREFIXES:
        if name.startswith(pfx):
            return False
    stem = path.stem.lower()
    for sfx in _SKIP_SUFFIXES:
        if sfx in stem:
            return False
    return True


def find_spec_files(root: Path) -> list[Path]:
    candidates = []
    for ext in ("*.docx", "*.doc"):
        for f in root.rglob(ext):
            if not _is_spec_file(f):
                continue
            candidates.append(f)

    by_spec: dict[str, list[Path]] = {}
    unrecognised: list[Path] = []
    for f in candidates:
        entry = infer_spec_from_filename(f.name)
        if entry:
            key = entry["spec_number"]
            by_spec.setdefault(key, []).append(f)
        else:
            unrecognised.append(f)

    selected = []
    for spec_num, paths in by_spec.items():
        latest = sorted(paths, key=lambda p: p.name.lower(), reverse=True)[0]
        selected.append(latest)
    selected.extend(unrecognised)
    return sorted(selected)


def enrich_chunks(chunks, file_path):
    entry = infer_spec_from_filename(file_path.name)
    for chunk in chunks:
        meta = chunk.setdefault("metadata", {})
        if entry:
            meta["domain"] = entry["domain"]
            meta["generation"] = entry["generation"]
            meta["spec_number"] = entry["spec_number"]
            meta["spec_title"] = entry["title"]
        else:
            meta.setdefault("domain", "unknown")
            meta.setdefault("generation", "unknown")
            meta.setdefault("spec_number", "unknown")
            meta.setdefault("spec_title", "unknown")
    return chunks


def main():
    print(f"\n{'='*60}")
    print("3GPP RAG Assistant — Index Builder (CPU mode)")
    print(f"{'='*60}\n")

    files = find_spec_files(RAW_DIR)
    if not files:
        print(f"No files found in {RAW_DIR}")
        sys.exit(1)

    print(f"Files to index: {len(files)}")
    for f in files:
        entry = infer_spec_from_filename(f.name)
        tag = f"{entry['generation']} {entry['domain']} TS {entry['spec_number']}" if entry else "unknown"
        print(f"  {f.name:40s}  [{tag}]")
    print()

    # Force CPU for stable embedding — avoids MPS stalls on large files
    processor = DocumentProcessor()
    generator = LocalEmbeddingGenerator(model_name="bge-small", device="cpu", batch_size=64)
    store = VectorStore()

    print("Clearing existing vector store...")
    store.clear()
    print("Done.\n")

    total_indexed = 0
    for i, file_path in enumerate(files, 1):
        entry = infer_spec_from_filename(file_path.name)
        label = f"TS {entry['spec_number']}" if entry else file_path.name
        print(f"[{i}/{len(files)}] Processing {label} ({file_path.name})...")

        try:
            chunks = processor.process_document(str(file_path))
        except Exception as exc:
            logger.error(f"  Failed to process {file_path.name}: {exc}")
            continue

        if not chunks:
            logger.warning(f"  No chunks from {file_path.name}")
            continue

        chunks = enrich_chunks(chunks, file_path)
        print(f"    {len(chunks)} chunks extracted")

        # Embed in sub-batches of 500 to keep memory stable
        SUB_BATCH = 500
        for start in range(0, len(chunks), SUB_BATCH):
            batch = chunks[start:start + SUB_BATCH]
            embedded = generator.embed_chunks(batch)
            store.add_chunks(embedded)
            total_indexed += len(embedded)
            print(f"    Indexed {min(start + SUB_BATCH, len(chunks))}/{len(chunks)} chunks")

    stats = store.get_stats()
    print(f"\n{'='*60}")
    print("Index Build Complete!")
    print(f"{'='*60}")
    print(f"Files processed : {len(files)}")
    print(f"Total chunks    : {total_indexed}")
    print(f"Total in store  : {stats['total_chunks']}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

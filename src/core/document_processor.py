"""
Document processing module for parsing and chunking 3GPP specifications

This module is the canonical entry point referenced in the README and CLI.
It delegates to UnifiedDocumentProcessor which handles PDF, DOCX, and DOC.
"""
from pathlib import Path
from typing import List, Dict
import logging

from src.core.document_processor_impl import UnifiedDocumentProcessor, DocumentChunk

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Process 3GPP specification documents (PDF, DOCX, DOC) into RAG chunks."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Args:
            chunk_size: Target character length of each chunk
            chunk_overlap: Character overlap between adjacent chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._processor = UnifiedDocumentProcessor(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def load_pdf(self, file_path: Path) -> str:
        """Load and extract raw text from a PDF file."""
        file_path = Path(file_path)
        text, _ = self._processor._load_pdf(file_path)
        return text

    def chunk_text(self, text: str, metadata: Dict) -> List[Dict]:
        """
        Split text into overlapping chunks.

        Returns:
            List of chunk dicts with 'text', 'metadata', and 'chunk_id' keys
        """
        chunks: List[DocumentChunk] = self._processor.chunk_text(text, metadata)
        return [c.to_dict() for c in chunks]

    def process_directory(self, directory: Path) -> List[Dict]:
        """
        Process all supported documents in a directory.

        Handles .pdf, .docx, and .doc files automatically.

        Returns:
            List of chunk dicts ready for embedding
        """
        directory = Path(directory)
        chunks: List[DocumentChunk] = self._processor.process_directory(directory)
        return [c.to_dict() for c in chunks]

    def process_document(self, file_path: Path) -> List[Dict]:
        """Process a single document file."""
        chunks: List[DocumentChunk] = self._processor.process_document(Path(file_path))
        return [c.to_dict() for c in chunks]

    def save_chunks(self, chunks: List[Dict], output_path: Path) -> None:
        """Save chunks list to a JSON file."""
        import json
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(chunks)} chunks to {output_path}")


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")

    processor = DocumentProcessor()
    data_dir = Path("data/raw")

    if not data_dir.exists():
        print(f"Error: {data_dir} not found. Add 3GPP spec files to data/raw/")
        sys.exit(1)

    print(f"\nProcessing documents in {data_dir} ...")
    chunks = processor.process_directory(data_dir)

    if not chunks:
        print("No documents processed.")
        sys.exit(1)

    output = Path("data/processed/chunks.json")
    processor.save_chunks(chunks, output)

    print(f"\nDone! {len(chunks)} chunks saved to {output}")
    print("Next step: python scripts/build_index.py")

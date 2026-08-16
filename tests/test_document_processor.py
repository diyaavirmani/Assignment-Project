"""
Test suite for document processor
"""
import pytest
from pathlib import Path
from src.core.document_processor import DocumentProcessor


def test_document_processor_init():
    """Test DocumentProcessor initialization"""
    processor = DocumentProcessor(chunk_size=500, chunk_overlap=100)
    assert processor.chunk_size == 500
    assert processor.chunk_overlap == 100


def test_document_processor_defaults():
    """Test DocumentProcessor default values"""
    processor = DocumentProcessor()
    assert processor.chunk_size == 1000
    assert processor.chunk_overlap == 200


# TODO: Add more tests as features are implemented
# - test_load_pdf()
# - test_chunk_text()
# - test_process_directory()

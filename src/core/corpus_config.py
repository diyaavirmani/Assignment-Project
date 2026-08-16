"""
Corpus configuration abstraction for the RAG pipeline.

This module defines the seam between the generic RAG infrastructure and any
specific document corpus.  3GPP is the implemented corpus; this file shows
where the 3GPP-specific choices plug in so a different corpus can be wired
in without forking the pipeline.

Design goals
------------
* Minimal: only the pieces that actually vary between corpora are in the
  config.  Everything else stays in the shared pipeline.
* Backward-compatible: all existing code that imports from spec_catalog or
  document_processor_impl continues to work unchanged.
* Reversible: the abstraction adds no mandatory dependencies; callers that
  don't need corpus configs simply don't import this module.

What belongs here vs stays in the pipeline
-------------------------------------------
Corpus-specific (this module):
  - Document catalog (what docs exist, their metadata schema)
  - Text cleaner (regex patterns that strip corpus-specific artifacts)
  - Metadata dimensions used for filtering (e.g. domain / generation for
    3GPP; chapter / regulation for a legal corpus)
  - FTP/download URL builder (3GPP FTP layout vs an SEC EDGAR path, etc.)

Shared pipeline (unchanged):
  - Chunking algorithm (sentence-boundary splitting, overlap)
  - Embedding model choice
  - Vector store schema (chunk_id, source, chunk_index, chunk_size)
  - LLM prompt templates and retrieval logic
  - Security controls, rate limiting, API surface
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


# ---------------------------------------------------------------------------
# Core types
# ---------------------------------------------------------------------------

CatalogEntry = Dict[str, Any]
"""A dict describing one document in the corpus.

Required keys (pipeline-level — must be present for every corpus):
  ``doc_id``   — unique stable identifier, e.g. "38.300" or "NIST-800-53-AC"
  ``title``    — human-readable title shown in the UI

Optional / corpus-specific keys may be added freely and are stored as
chunk metadata so they can be used as retrieval filters.  For 3GPP the
extra keys are: spec_number, series, domain, generation, description.
"""

TextCleaner = Callable[[str], str]
"""A function that takes raw extracted text and returns cleaned text.

Each corpus has its own artifacts to remove: 3GPP specs have running headers
like "3GPP TS 38.300 version 16.3.0", legal PDFs have Bates numbers, SEC
filings have XBRL tags, etc.  Provide a corpus-specific cleaner here.
"""


# ---------------------------------------------------------------------------
# CorpusConfig dataclass
# ---------------------------------------------------------------------------

@dataclass
class CorpusConfig:
    """
    All corpus-specific configuration in one place.

    To add a new corpus, create a CorpusConfig (or a factory function that
    returns one) and pass it wherever corpus-specific behaviour is needed.
    See ``build_3gpp_corpus_config()`` for the worked example.

    Attributes
    ----------
    name : str
        Short identifier, e.g. ``"3gpp"`` or ``"sec-filings"``.
    description : str
        One-sentence description shown in docs and diagnostics.
    catalog : List[CatalogEntry]
        All documents the system knows about.  Each entry must have at least
        ``doc_id`` and ``title`` keys.  Additional keys become chunk metadata.
    text_cleaner : TextCleaner
        Callable applied to raw extracted text before chunking.  Use the
        default ``generic_text_cleaner`` if the corpus has no special
        artifacts to remove.
    filter_dimensions : List[str]
        Metadata keys that the retriever can use as pre-retrieval filters.
        For 3GPP this is ``["domain", "generation"]``; for a legal corpus it
        might be ``["regulation", "section_type"]``.
    doc_id_field : str
        Which catalog entry key holds the unique document identifier.
        Defaults to ``"doc_id"``.  The 3GPP catalog uses ``"spec_number"``
        historically; ``build_3gpp_corpus_config()`` maps this correctly.
    """

    name: str
    description: str
    catalog: List[CatalogEntry]
    text_cleaner: TextCleaner = field(default_factory=lambda: generic_text_cleaner)
    filter_dimensions: List[str] = field(default_factory=list)
    doc_id_field: str = "doc_id"

    # ------------------------------------------------------------------
    # Catalog lookup helpers (generic — work for any corpus)
    # ------------------------------------------------------------------

    def get_entry(self, doc_id: str) -> Optional[CatalogEntry]:
        """Return the catalog entry whose ``doc_id_field`` matches *doc_id*."""
        for entry in self.catalog:
            if entry.get(self.doc_id_field) == doc_id:
                return entry
        return None

    def filter_catalog(self, **kwargs: str) -> List[CatalogEntry]:
        """Return entries where every key=value in *kwargs* matches."""
        results = self.catalog
        for key, value in kwargs.items():
            results = [e for e in results if e.get(key) == value]
        return results

    def summary(self) -> str:
        """One-line summary for diagnostics."""
        return f"CorpusConfig({self.name!r}, {len(self.catalog)} docs)"


# ---------------------------------------------------------------------------
# Generic (no-op) text cleaner
# ---------------------------------------------------------------------------

def generic_text_cleaner(text: str) -> str:
    """Minimal cleaner: collapse whitespace and remove non-printable chars.

    Use as the ``text_cleaner`` for corpora that don't have known extraction
    artifacts.  The 3GPP corpus uses a stricter cleaner; see
    ``threegpp_text_cleaner``.
    """
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# 3GPP-specific text cleaner
# ---------------------------------------------------------------------------

def threegpp_text_cleaner(text: str) -> str:
    """Clean text extracted from 3GPP specification PDFs/DOCX files.

    Removes artifacts specific to 3GPP document formatting:
    - Running headers/footers like "3GPP TS 38.300 version 16.3.0 Release 16"
    - Repeated whitespace and extra blank lines

    This function encapsulates the 3GPP-specific regex that previously lived
    inline in ``UnifiedDocumentProcessor.clean_text()`` (``document_processor_impl``).  That method still
    works unchanged; this version is exposed here so corpus-agnostic code can
    call ``config.text_cleaner(text)`` without knowing which corpus it has.

    Operation order matters: the 3GPP header pattern requires newlines to be
    intact, so it runs before the whitespace-collapse step.
    """
    # Step 1: strip excess blank lines (preserves single newlines for header detection)
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
    # Step 2: strip 3GPP running header/footer lines, e.g. "3  3GPP TS 38.300 V16.3.0"
    # Pattern: a line that starts with digits, optional spaces, then "3GPP"
    text = re.sub(r"\n\d+\s+3GPP[^\n]*\n", "\n", text)
    # Step 3: collapse remaining whitespace (including now-orphaned spaces)
    text = re.sub(r"\s+", " ", text)
    # Step 4: remove characters outside expected 3GPP document range
    text = re.sub(r"[^\w\s.,;:()\[\]\-/\n#|]", "", text)
    return text.strip()


# ---------------------------------------------------------------------------
# 3GPP corpus factory
# ---------------------------------------------------------------------------

def build_3gpp_corpus_config() -> CorpusConfig:
    """Return the CorpusConfig for the 3GPP specification corpus.

    This is the canonical instantiation of CorpusConfig.  It wraps the
    existing ``spec_catalog.CATALOG`` and 3GPP-specific cleaning logic.
    The 3GPP pipeline continues to work exactly as before; this factory
    just makes the seam explicit.

    Returns
    -------
    CorpusConfig
        Fully configured instance for the 37-spec 3GPP catalog.
    """
    # Import here to avoid circular imports; spec_catalog has no deps on this
    # module so the import is always safe.
    from src.core.spec_catalog import CATALOG as _3GPP_CATALOG

    # Normalize: add the generic ``doc_id`` alias pointing to ``spec_number``
    # so corpus-agnostic helpers (get_entry, filter_catalog) work without
    # knowing 3GPP-specific field names.  Original keys are preserved.
    catalog: List[CatalogEntry] = []
    for entry in _3GPP_CATALOG:
        normalized = dict(entry)
        normalized["doc_id"] = entry["spec_number"]
        catalog.append(normalized)

    return CorpusConfig(
        name="3gpp",
        description=(
            "3GPP Release 16/17 specifications covering 5G NR and LTE. "
            "37 curated specs across RAN and Core domains."
        ),
        catalog=catalog,
        text_cleaner=threegpp_text_cleaner,
        filter_dimensions=["domain", "generation"],
        doc_id_field="spec_number",
    )


# ---------------------------------------------------------------------------
# Module-level default: 3GPP corpus (the only implemented corpus)
# ---------------------------------------------------------------------------

DEFAULT_CORPUS: CorpusConfig = build_3gpp_corpus_config()
"""
The default corpus config used when nothing else is specified.

To use a different corpus, construct a ``CorpusConfig`` and pass it
explicitly — do not mutate this module-level object.

See ``docs/GENERALIZE.md`` for how to add a new corpus.
"""

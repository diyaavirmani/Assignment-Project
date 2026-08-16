"""
Tests for src/core/corpus_config.py

Verifies:
  - CorpusConfig dataclass construction and helpers
  - generic_text_cleaner basic normalisation
  - threegpp_text_cleaner 3GPP-specific artifact removal
  - build_3gpp_corpus_config() returns the right shape / wraps spec_catalog
  - DEFAULT_CORPUS is an alias for the 3GPP config
  - The 3GPP config round-trips correctly through the generic helpers (get_entry,
    filter_catalog, summary) so any future corpus can rely on these methods
"""

import re
import pytest

from src.core.corpus_config import (
    CorpusConfig,
    CatalogEntry,
    TextCleaner,
    generic_text_cleaner,
    threegpp_text_cleaner,
    build_3gpp_corpus_config,
    DEFAULT_CORPUS,
)


# ---------------------------------------------------------------------------
# generic_text_cleaner
# ---------------------------------------------------------------------------

class TestGenericTextCleaner:
    def test_collapses_whitespace(self):
        text = "hello   world\t\there"
        result = generic_text_cleaner(text)
        assert "  " not in result

    def test_collapses_excess_blank_lines(self):
        text = "para one\n\n\n\n\npara two"
        result = generic_text_cleaner(text)
        assert result.count("\n") <= 2

    def test_preserves_content(self):
        text = "The gNB-CU hosts RRC and PDCP."
        result = generic_text_cleaner(text)
        assert "gNB-CU" in result
        assert "RRC" in result

    def test_strips_leading_trailing_whitespace(self):
        text = "   hello world   "
        result = generic_text_cleaner(text)
        assert result == result.strip()


# ---------------------------------------------------------------------------
# threegpp_text_cleaner
# ---------------------------------------------------------------------------

class TestThreeGPPTextCleaner:
    def test_removes_3gpp_running_header(self):
        text = "\n3  3GPP TS 38.300 V16.3.0 (2021-03)\nSome real content here."
        result = threegpp_text_cleaner(text)
        assert "3GPP TS 38.300" not in result
        assert "real content" in result

    def test_removes_another_header_variant(self):
        text = "\n12 3GPP TS 36.331 version 15.4.0 Release 15\nContent after."
        result = threegpp_text_cleaner(text)
        assert "3GPP TS 36.331" not in result
        assert "Content after" in result

    def test_preserves_technical_content(self):
        text = "The F1 interface connects the gNB-CU to the gNB-DU via NGAP."
        result = threegpp_text_cleaner(text)
        assert "F1 interface" in result
        assert "gNB-CU" in result

    def test_collapses_whitespace(self):
        text = "hello   world"
        result = threegpp_text_cleaner(text)
        assert "  " not in result

    def test_content_without_3gpp_headers_unchanged_in_substance(self):
        text = "Physical channel structure for 5G NR is defined in TS 38.211."
        result = threegpp_text_cleaner(text)
        # Technical content should survive
        assert "Physical channel structure" in result
        assert "38.211" in result


# ---------------------------------------------------------------------------
# CorpusConfig dataclass
# ---------------------------------------------------------------------------

@pytest.fixture
def minimal_catalog() -> list:
    return [
        {"doc_id": "DOC-001", "title": "First Document", "category": "A"},
        {"doc_id": "DOC-002", "title": "Second Document", "category": "B"},
        {"doc_id": "DOC-003", "title": "Third Document", "category": "A"},
    ]


@pytest.fixture
def minimal_corpus(minimal_catalog) -> CorpusConfig:
    return CorpusConfig(
        name="test-corpus",
        description="A minimal corpus for testing.",
        catalog=minimal_catalog,
        filter_dimensions=["category"],
    )


class TestCorpusConfigConstruction:
    def test_required_fields(self, minimal_corpus):
        assert minimal_corpus.name == "test-corpus"
        assert minimal_corpus.description == "A minimal corpus for testing."
        assert len(minimal_corpus.catalog) == 3

    def test_default_doc_id_field(self, minimal_corpus):
        assert minimal_corpus.doc_id_field == "doc_id"

    def test_default_text_cleaner_is_generic(self, minimal_corpus):
        # Default cleaner should behave like generic_text_cleaner
        text = "hello   world"
        result = minimal_corpus.text_cleaner(text)
        assert "  " not in result

    def test_filter_dimensions_stored(self, minimal_corpus):
        assert "category" in minimal_corpus.filter_dimensions


class TestCorpusConfigGetEntry:
    def test_returns_matching_entry(self, minimal_corpus):
        entry = minimal_corpus.get_entry("DOC-001")
        assert entry is not None
        assert entry["title"] == "First Document"

    def test_returns_none_for_missing(self, minimal_corpus):
        entry = minimal_corpus.get_entry("DOES-NOT-EXIST")
        assert entry is None

    def test_exact_match_required(self, minimal_corpus):
        # Partial match should not return an entry
        assert minimal_corpus.get_entry("DOC") is None


class TestCorpusConfigFilterCatalog:
    def test_filter_by_single_key(self, minimal_corpus):
        results = minimal_corpus.filter_catalog(category="A")
        assert len(results) == 2
        for entry in results:
            assert entry["category"] == "A"

    def test_filter_returns_empty_when_no_match(self, minimal_corpus):
        results = minimal_corpus.filter_catalog(category="Z")
        assert results == []

    def test_filter_with_no_kwargs_returns_all(self, minimal_corpus):
        results = minimal_corpus.filter_catalog()
        assert len(results) == 3

    def test_filter_multiple_keys(self):
        catalog = [
            {"doc_id": "X", "title": "X", "type": "regulation", "region": "EU"},
            {"doc_id": "Y", "title": "Y", "type": "regulation", "region": "US"},
            {"doc_id": "Z", "title": "Z", "type": "guidance", "region": "EU"},
        ]
        corpus = CorpusConfig(
            name="legal",
            description="Legal docs",
            catalog=catalog,
            filter_dimensions=["type", "region"],
        )
        results = corpus.filter_catalog(type="regulation", region="EU")
        assert len(results) == 1
        assert results[0]["doc_id"] == "X"


class TestCorpusConfigSummary:
    def test_summary_contains_name(self, minimal_corpus):
        assert "test-corpus" in minimal_corpus.summary()

    def test_summary_contains_count(self, minimal_corpus):
        assert "3" in minimal_corpus.summary()


# ---------------------------------------------------------------------------
# build_3gpp_corpus_config()
# ---------------------------------------------------------------------------

class TestBuildThreeGPPCorpusConfig:
    @pytest.fixture(autouse=True)
    def config(self):
        self._config = build_3gpp_corpus_config()

    def test_name_is_3gpp(self):
        assert self._config.name == "3gpp"

    def test_catalog_is_non_empty(self):
        assert len(self._config.catalog) > 0

    def test_catalog_has_37_specs(self):
        # The 3GPP catalog currently has 37 entries; flag if count changes
        assert len(self._config.catalog) == 37

    def test_each_entry_has_doc_id(self):
        for entry in self._config.catalog:
            assert "doc_id" in entry, f"Missing doc_id in {entry}"

    def test_each_entry_has_spec_number(self):
        for entry in self._config.catalog:
            assert "spec_number" in entry

    def test_doc_id_equals_spec_number(self):
        for entry in self._config.catalog:
            assert entry["doc_id"] == entry["spec_number"]

    def test_each_entry_has_title(self):
        for entry in self._config.catalog:
            assert entry.get("title"), f"Missing title in {entry}"

    def test_filter_dimensions_include_domain_and_generation(self):
        assert "domain" in self._config.filter_dimensions
        assert "generation" in self._config.filter_dimensions

    def test_text_cleaner_is_3gpp_cleaner(self):
        # The 3GPP cleaner should strip 3GPP header lines
        text = "\n5  3GPP TS 38.401 V16.0.0 (2020-12)\nActual content."
        result = self._config.text_cleaner(text)
        assert "3GPP TS 38.401" not in result
        assert "Actual content" in result

    def test_get_entry_by_spec_number(self):
        entry = self._config.get_entry("38.300")
        assert entry is not None
        assert "NG-RAN" in entry["title"] or "NR" in entry["title"]

    def test_get_entry_missing_returns_none(self):
        assert self._config.get_entry("99.999") is None

    def test_filter_by_domain_ran(self):
        ran_specs = self._config.filter_catalog(domain="RAN")
        assert len(ran_specs) > 0
        for entry in ran_specs:
            assert entry["domain"] == "RAN"

    def test_filter_by_domain_core(self):
        core_specs = self._config.filter_catalog(domain="CORE")
        assert len(core_specs) > 0
        for entry in core_specs:
            assert entry["domain"] == "CORE"

    def test_filter_by_generation_5g(self):
        g5_specs = self._config.filter_catalog(generation="5G")
        assert len(g5_specs) > 0

    def test_filter_by_generation_lte(self):
        lte_specs = self._config.filter_catalog(generation="LTE")
        assert len(lte_specs) > 0

    def test_filter_ran_5g(self):
        results = self._config.filter_catalog(domain="RAN", generation="5G")
        assert len(results) > 0
        for entry in results:
            assert entry["domain"] == "RAN"
            assert entry["generation"] == "5G"

    def test_original_catalog_not_mutated(self):
        # build_3gpp_corpus_config normalizes entries into new dicts;
        # the original spec_catalog.CATALOG should not have a doc_id key added
        from src.core.spec_catalog import CATALOG
        for entry in CATALOG:
            assert "doc_id" not in entry, (
                "build_3gpp_corpus_config() must not mutate spec_catalog.CATALOG"
            )


# ---------------------------------------------------------------------------
# DEFAULT_CORPUS
# ---------------------------------------------------------------------------

class TestDefaultCorpus:
    def test_is_corpus_config_instance(self):
        assert isinstance(DEFAULT_CORPUS, CorpusConfig)

    def test_is_3gpp(self):
        assert DEFAULT_CORPUS.name == "3gpp"

    def test_has_catalog(self):
        assert len(DEFAULT_CORPUS.catalog) > 0


# ---------------------------------------------------------------------------
# Hypothetical second corpus (verifies the interface is generic)
# ---------------------------------------------------------------------------

class TestHypotheticalSecondCorpus:
    """
    Demonstrate that building a second corpus requires only a CorpusConfig —
    no changes to spec_catalog.py, document_processor, or retriever.

    This test uses a tiny synthetic catalog; it is not a claim that the
    system has been tested against real NIST documents.
    """

    @pytest.fixture
    def nist_like_corpus(self) -> CorpusConfig:
        catalog = [
            {
                "doc_id": "NIST-800-53-rev5",
                "title": "Security and Privacy Controls for Information Systems",
                "category": "control-catalog",
                "family": "AC",
            },
            {
                "doc_id": "NIST-800-37-rev2",
                "title": "Risk Management Framework for Information Systems",
                "category": "framework",
                "family": "RM",
            },
        ]
        return CorpusConfig(
            name="nist-800",
            description="NIST SP 800-series cybersecurity publications.",
            catalog=catalog,
            text_cleaner=generic_text_cleaner,
            filter_dimensions=["category", "family"],
            doc_id_field="doc_id",
        )

    def test_construction(self, nist_like_corpus):
        assert nist_like_corpus.name == "nist-800"
        assert len(nist_like_corpus.catalog) == 2

    def test_get_entry(self, nist_like_corpus):
        entry = nist_like_corpus.get_entry("NIST-800-53-rev5")
        assert entry is not None
        assert "Security" in entry["title"]

    def test_filter_catalog(self, nist_like_corpus):
        results = nist_like_corpus.filter_catalog(category="framework")
        assert len(results) == 1
        assert results[0]["doc_id"] == "NIST-800-37-rev2"

    def test_text_cleaner_callable(self, nist_like_corpus):
        text = "  some   spaced   text  "
        result = nist_like_corpus.text_cleaner(text)
        assert result == result.strip()

    def test_does_not_share_state_with_3gpp(self, nist_like_corpus):
        # The two configs are independent; modifying one should not affect the other
        three_gpp = build_3gpp_corpus_config()
        assert nist_like_corpus.name != three_gpp.name
        assert nist_like_corpus.catalog is not three_gpp.catalog

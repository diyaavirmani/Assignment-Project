"""
Tests for src/core/query_expansion.py — 3GPP vocabulary query expansion.

Pure string logic: no vector store, no LLM, no index required.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.query_expansion import THREEGPP_VOCAB, expand_query, _term_matches


class TestTermMatching:
    def test_uppercase_acronym_matches_on_word_boundary(self):
        assert _term_matches("SA", "difference between SA and NSA options") is True

    def test_uppercase_acronym_is_case_sensitive(self):
        """Short all-caps terms must not match lowercase prose ('sa', 'ran')."""
        assert _term_matches("SA", "the usage of sa is common") is False
        assert _term_matches("RAN", "the process ran quickly") is False

    def test_no_match_inside_longer_token(self):
        """'F1' must not match inside 'F1AP' (alnum-adjacent)."""
        assert _term_matches("F1", "What does F1AP specify?") is False

    def test_hyphen_is_a_boundary(self):
        """'gNB' matches inside 'gNB-CU' — hyphens separate tokens."""
        assert _term_matches("gNB", "the gNB-CU handles RRC") is True

    def test_mixed_case_term_matches_case_insensitively(self):
        assert _term_matches("gNB", "what is a GNB?") is True
        assert _term_matches("NG-RAN", "the ng-ran architecture") is True

    def test_multiword_term(self):
        assert _term_matches("NG interface", "What is the NG interface for?") is True


class TestExpandQuery:
    def test_no_known_terms_returns_query_unchanged(self):
        q = "test query with no telecom vocabulary"
        assert expand_query(q) == q

    def test_expansion_preserves_original_text(self):
        q = "What is SDAP and what role does it play in 5G?"
        out = expand_query(q)
        assert out.startswith(q)
        assert "Service Data Adaptation Protocol" in out

    def test_named_miss_sa_vs_nsa(self):
        """The gs-007 miss query gains both deployment-mode full forms."""
        out = expand_query("What is the difference between SA and NSA 5G deployment options?")
        assert "Standalone deployment" in out
        assert "Non-Standalone deployment" in out

    def test_named_miss_e1_interface(self):
        """The gs-003 miss query gains the E1 definition vocabulary."""
        out = expand_query("What is the E1 interface used for in 5G NG-RAN?")
        assert "gNB-CU-CP" in out
        assert "Next Generation Radio Access Network" in out

    def test_multiple_expansions_joined_in_one_gloss(self):
        out = expand_query("How do RRC and PDCP interact?")
        assert out.count("(") == 1
        assert "Radio Resource Control protocol" in out
        assert "Packet Data Convergence Protocol" in out

    def test_duplicate_full_forms_not_repeated(self):
        vocab = {"NR": "New Radio", "NR2": "New Radio"}
        out = expand_query("NR and NR2", vocab=vocab)
        assert out.count("New Radio") == 1

    def test_custom_vocab_override(self):
        out = expand_query("What is XYZ?", vocab={"XYZ": "example expansion"})
        assert out == "What is XYZ? (example expansion)"

    def test_vocab_entries_are_nonempty(self):
        for term, full_form in THREEGPP_VOCAB.items():
            assert term.strip() and full_form.strip()

    def test_lowercase_prose_not_expanded(self):
        """A query whose words merely resemble acronyms stays unchanged."""
        q = "he ran to the mac store and got a phy book"
        assert expand_query(q) == q

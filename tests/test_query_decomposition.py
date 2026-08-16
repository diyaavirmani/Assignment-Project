"""
Tests for src/core/query_decomposition.py — comparison-query decomposition.

Pure string logic: no vector store, no LLM, no index required.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.query_decomposition import decompose_query


class TestDecomposeQuery:
    def test_non_comparison_query_returns_empty(self):
        assert decompose_query("What is the F1 interface used for?") == []

    def test_named_miss_sa_vs_nsa(self):
        """gs-007: the surviving comparison miss."""
        subs = decompose_query(
            "What is the difference between SA and NSA 5G deployment options?"
        )
        assert subs == [
            "SA 5G deployment options",
            "NSA 5G deployment options",
        ]

    def test_named_miss_nr_vs_lte_phy(self):
        """gs-025: the other surviving comparison miss."""
        subs = decompose_query(
            "What are the differences between NR and LTE physical layer?"
        )
        assert subs == ["NR physical layer", "LTE physical layer"]

    def test_between_with_multiword_left_side(self):
        subs = decompose_query(
            "Explain the difference between the gNB-CU and gNB-DU functions"
        )
        assert len(subs) == 2
        assert "gNB-CU" in subs[0] and "functions" in subs[0]
        assert subs[1] == "gNB-DU functions"

    def test_vs_pattern(self):
        assert decompose_query("NR vs LTE handover procedures?") != []

    def test_versus_pattern(self):
        subs = decompose_query("Compare SA versus NSA deployment")
        assert len(subs) == 2

    def test_interface_between_phrasing_not_decomposed(self):
        """Bare 'between' is not a comparison: interface questions describe
        one concept (the interface), and splitting them harms retrieval."""
        assert decompose_query("What is exchanged between the AMF and SMF?") == []
        assert decompose_query("Describe the F1 interface between gNB-CU and gNB-DU") == []

    def test_identical_sides_rejected(self):
        assert decompose_query("difference between NR and NR?") == []

    def test_no_hint_words_short_circuits(self):
        """The cheap pre-filter rejects before any regex work."""
        assert decompose_query("How does HARQ retransmission work in NR?") == []

    def test_question_punctuation_stripped(self):
        subs = decompose_query("differences between NR and LTE physical layer?")
        for sub in subs:
            assert not sub.endswith("?")

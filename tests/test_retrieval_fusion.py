"""
Tests for src/core/retrieval_fusion.py — the shared, embedding-agnostic
query-rewriting rank fusion used by both the local retriever and the
Streamlit Cloud path.

Pure logic: fusion runs against a fake in-memory search_fn, no vector store,
no embeddings, no LLM.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.retrieval_fusion import (
    rrf_merge,
    cap_per_source,
    interleave_sides,
    fused_retrieve,
)


def _doc(source, idx, sim=0.5, text="t"):
    return {"source": source, "chunk_index": idx, "similarity": sim, "text": text}


# ---------------------------------------------------------------------------
# rrf_merge
# ---------------------------------------------------------------------------

class TestRRFMerge:
    def test_consensus_doc_ranks_first(self):
        a = _doc("x.docx", 0)
        b = _doc("y.docx", 0)
        c = _doc("z.docx", 0)
        # 'a' is rank 1 in both lists (consensus); b and c each appear once.
        merged = rrf_merge([[a, b], [a, c]])
        assert merged[0]["source"] == "x.docx"

    def test_dedup_by_source_and_chunk(self):
        a1 = _doc("x.docx", 0)
        a1_again = _doc("x.docx", 0)
        merged = rrf_merge([[a1], [a1_again]])
        assert len(merged) == 1

    def test_distinct_chunks_same_source_kept(self):
        merged = rrf_merge([[_doc("x.docx", 0), _doc("x.docx", 1)]])
        assert len(merged) == 2

    def test_empty_lists(self):
        assert rrf_merge([[], []]) == []


# ---------------------------------------------------------------------------
# cap_per_source
# ---------------------------------------------------------------------------

class TestCapPerSource:
    def test_dominant_source_capped(self):
        docs = (
            [_doc("a.docx", i) for i in range(5)]
            + [_doc("b.docx", 0), _doc("c.docx", 0)]
        )
        result = cap_per_source(docs, k=5)
        sources = [d["source"] for d in result]
        assert sources.count("a.docx") == 3
        assert "b.docx" in sources
        assert "c.docx" in sources

    def test_backfill_when_no_other_sources(self):
        docs = [_doc("a.docx", i) for i in range(6)]
        result = cap_per_source(docs, k=5)
        assert len(result) == 5

    def test_order_preserved_within_cap(self):
        docs = [_doc("a.docx", 0), _doc("b.docx", 0), _doc("a.docx", 1)]
        assert cap_per_source(docs, k=3) == docs


# ---------------------------------------------------------------------------
# interleave_sides
# ---------------------------------------------------------------------------

class TestInterleaveSides:
    def test_one_chunk_per_source_across_sides(self):
        side_a = [_doc("nr.docx", 0), _doc("nr.docx", 1)]
        side_b = [_doc("lte.docx", 0), _doc("lte.docx", 1)]
        result = interleave_sides([side_a, side_b], backfill=[], k=4)
        sources = [d["source"] for d in result]
        # round-robin, one chunk per source: nr then lte, then exhausted
        assert sources == ["nr.docx", "lte.docx"]

    def test_backfill_fills_remaining_slots(self):
        side_a = [_doc("nr.docx", 0)]
        side_b = [_doc("lte.docx", 0)]
        backfill = [_doc("extra.docx", 0), _doc("nr.docx", 0)]
        result = interleave_sides([side_a, side_b], backfill, k=3)
        assert [d["source"] for d in result] == ["nr.docx", "lte.docx", "extra.docx"]

    def test_respects_k(self):
        side_a = [_doc("a.docx", i) for i in range(10)]
        result = interleave_sides([side_a], backfill=[], k=3)
        assert len(result) <= 3


# ---------------------------------------------------------------------------
# fused_retrieve — orchestration against a fake backend
# ---------------------------------------------------------------------------

class FakeBackend:
    """Records every (text, n) search and returns canned per-text results."""

    def __init__(self, results_by_substr):
        self.results_by_substr = results_by_substr
        self.calls = []

    def search_fn(self, text, n):
        self.calls.append((text, n))
        for substr, docs in self.results_by_substr.items():
            if substr in text:
                return docs[:n]
        return []


class TestFusedRetrieve:
    def test_plain_query_single_search(self):
        backend = FakeBackend({"gNB": [_doc("38300.docx", 0)]})
        out = fused_retrieve("What is a gNB", backend.search_fn, k=5,
                             expand=False, decompose=False)
        assert len(backend.calls) == 1
        assert out[0]["source"] == "38300.docx"

    def test_expansion_triggers_second_search_and_fuses(self):
        backend = FakeBackend({
            "SDAP": [_doc("38300.docx", 0)],
            "Service Data Adaptation": [_doc("38300.docx", 1)],
        })
        fused_retrieve("What is SDAP?", backend.search_fn, k=5,
                       expand=True, decompose=False)
        searched = [t for t, _ in backend.calls]
        assert any("Service Data Adaptation Protocol" in t for t in searched)
        assert "What is SDAP?" in searched  # raw query never discarded

    def test_no_expansion_when_no_known_terms(self):
        backend = FakeBackend({"foo": [_doc("x.docx", 0)]})
        fused_retrieve("tell me about foo", backend.search_fn, k=5,
                       expand=True, decompose=False)
        assert len(backend.calls) == 1  # nothing to expand → single search

    def test_comparison_searches_each_side(self):
        backend = FakeBackend({
            "NR physical layer": [_doc("38211.docx", 0)],
            "LTE physical layer": [_doc("36211.docx", 0)],
            "differences": [_doc("38300.docx", 0)],
        })
        out = fused_retrieve(
            "What are the differences between NR and LTE physical layer?",
            backend.search_fn, k=5, expand=False, decompose=True,
        )
        searched = [t for t, _ in backend.calls]
        assert any("NR physical layer" in t for t in searched)
        assert any("LTE physical layer" in t for t in searched)
        out_sources = [d["source"] for d in out]
        assert "38211.docx" in out_sources and "36211.docx" in out_sources

    def test_decompose_disabled(self):
        backend = FakeBackend({"NR": [_doc("x.docx", 0)]})
        fused_retrieve(
            "differences between NR and LTE physical layer",
            backend.search_fn, k=5, expand=False, decompose=False,
        )
        assert len(backend.calls) == 1

    def test_returns_at_most_k(self):
        backend = FakeBackend({"q": [_doc("a.docx", i) for i in range(20)]})
        out = fused_retrieve("q", backend.search_fn, k=5,
                             expand=False, decompose=False)
        assert len(out) <= 5

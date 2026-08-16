"""Tests for cross-encoder reranking."""

import pytest

from src.core.reranker import CrossEncoderReranker


class FakeCrossEncoder:
    def __init__(self, scores):
        self.scores = scores
        self.seen_pairs = None

    def predict(self, pairs):
        self.seen_pairs = pairs
        return self.scores


def _doc(text, similarity=0.5, **metadata):
    doc = {
        "text": text,
        "source": metadata.get("source", "38300-j10.docx"),
        "chunk_index": metadata.get("chunk_index", 0),
        "similarity": similarity,
        "domain": metadata.get("domain", "RAN"),
        "generation": metadata.get("generation", "5G"),
    }
    doc.update(metadata)
    return doc


def test_higher_model_score_sorts_first():
    model = FakeCrossEncoder([0.1, 3.0, -1.0])
    reranker = CrossEncoderReranker(model=model)

    results = reranker.rerank(
        "what is gNB",
        [_doc("weak"), _doc("strong"), _doc("bad")],
        top_n=3,
    )

    assert [doc["text"] for doc in results] == ["strong", "weak", "bad"]


def test_metadata_preserved():
    model = FakeCrossEncoder([1.0])
    reranker = CrossEncoderReranker(model=model)
    candidate = _doc("gNB text", release="18", section="4.2")

    result = reranker.rerank("query", [candidate], top_n=1)[0]

    assert result["release"] == "18"
    assert result["section"] == "4.2"
    assert result["source"] == "38300-j10.docx"


def test_vector_similarity_preserved():
    model = FakeCrossEncoder([1.0])
    reranker = CrossEncoderReranker(model=model)

    result = reranker.rerank("query", [_doc("text", similarity=0.61)], top_n=1)[0]

    assert result["similarity"] == 0.61
    assert result["vector_similarity"] == 0.61


def test_reranker_score_attached():
    model = FakeCrossEncoder([0.0])
    reranker = CrossEncoderReranker(model=model)

    result = reranker.rerank("query", [_doc("text")], top_n=1)[0]

    assert result["reranker_raw_score"] == 0.0
    assert result["reranker_score"] == pytest.approx(0.5)
    assert result["rank_after_reranking"] == 1


def test_normalized_scores_valid():
    assert 0.0 <= CrossEncoderReranker.normalize_score(-100.0) <= 1.0
    assert 0.0 <= CrossEncoderReranker.normalize_score(100.0) <= 1.0
    assert CrossEncoderReranker.normalize_score(0.0) == pytest.approx(0.5)


def test_top_n_behavior():
    model = FakeCrossEncoder([0.0, 1.0, 2.0])
    reranker = CrossEncoderReranker(model=model)

    results = reranker.rerank("query", [_doc("a"), _doc("b"), _doc("c")], top_n=2)

    assert len(results) == 2
    assert [doc["text"] for doc in results] == ["c", "b"]


def test_empty_candidates():
    reranker = CrossEncoderReranker(model=FakeCrossEncoder([]))

    assert reranker.rerank("query", [], top_n=5) == []


def test_one_candidate():
    reranker = CrossEncoderReranker(model=FakeCrossEncoder([4.0]))

    result = reranker.rerank("query", [_doc("only")], top_n=5)

    assert len(result) == 1
    assert result[0]["text"] == "only"


def test_dependency_injected_fake_model_receives_pairs():
    model = FakeCrossEncoder([1.0])
    reranker = CrossEncoderReranker(model=model)

    reranker.rerank("query text", [_doc("candidate text")], top_n=1)

    assert model.seen_pairs == [("query text", "candidate text")]


def test_disabled_retriever_behavior(monkeypatch, mock_vector_store, mock_embedding_generator):
    from src.core import retriever as retriever_module

    monkeypatch.setattr(retriever_module.settings, "query_expansion", False)
    monkeypatch.setattr(retriever_module.settings, "query_decomposition", False)
    ret = retriever_module.DocumentRetriever(
        vector_store=mock_vector_store,
        embedding_generator=mock_embedding_generator,
        reranker_enabled=False,
        top_k=2,
    )

    results = ret.retrieve("query")

    assert len(results) == 2
    assert "reranker_score" not in results[0]
    assert mock_vector_store.query.call_args.kwargs["n_results"] == 2

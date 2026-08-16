"""Unit tests for the guarded RAG evaluation suite."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from scripts.eval.verified_rag import (
    ANSWERED,
    REFUSED,
    DatasetValidationError,
    RuntimeProviderError,
    assert_production_provider_config,
    assert_runtime_matches_config,
    aggregate_results,
    classify_behavior,
    format_runtime_diagnostics,
    load_dataset,
    render_markdown_report,
    retrieval_metrics,
    run_evaluation,
    runtime_provider_info,
    score_rag_result,
    validate_example,
    write_outputs,
)
from scripts.analyze_reranker_scores import analyze as analyze_reranker_scores


ANSWER_EXAMPLE = {
    "id": "df-001",
    "category": "direct_fact",
    "question": "What is F1?",
    "expected_behavior": "answer",
    "gold_spec_numbers": ["38.401"],
    "gold_sources": ["38401"],
    "gold_answer": "F1 connects gNB-CU and gNB-DU.",
    "required_terms": ["F1", "gNB-CU"],
    "forbidden_terms": ["blockchain"],
    "notes": "",
}

REFUSAL_EXAMPLE = {
    "id": "ood-001",
    "category": "out_of_domain",
    "question": "What is a Big Mac?",
    "expected_behavior": "refuse",
    "gold_spec_numbers": [],
    "gold_sources": [],
    "gold_answer": None,
    "required_terms": [],
    "forbidden_terms": ["calories"],
    "notes": "",
}


def _rag_result(
    *,
    answer="F1 connects gNB-CU and gNB-DU [S1].",
    answer_status="answered",
    evidence=None,
    verification=None,
    sources=None,
):
    return {
        "answer": answer,
        "answer_status": answer_status,
        "sources": (
            sources
            if sources is not None
            else [
                {
                    "source_id": "S1",
                    "source": "38401-j10.docx",
                    "similarity": 0.91,
                    "text": "F1 connects gNB-CU and gNB-DU",
                    "spec_number": "38.401",
                },
                {
                    "source_id": "S2",
                    "source": "38300-j10.docx",
                    "similarity": 0.72,
                    "text": "NG-RAN overview",
                    "spec_number": "38.300",
                },
            ]
        ),
        "evidence": (
            evidence
            if evidence is not None
            else {
                "sufficient": True,
                "reason": "sufficient_evidence",
                "top_score": 0.91,
                "mean_score": 0.82,
                "qualifying_docs": 2,
                "total_docs": 2,
            }
        ),
        "verification": (
            verification
            if verification is not None
            else {
                "passed": True,
                "reason": "verified",
                "total_claims": 1,
                "supported_claims": 1,
                "unsupported_claims": [],
                "contradicted_claims": [],
                "citation_valid": True,
                "cited_sources": ["S1"],
                "invalid_citations": [],
            }
        ),
        "retrieve_time": 0.1,
        "generate_time": 0.5,
        "verify_time": 0.2,
        "query_time": 0.8,
    }


class OpenAIEmbeddingGenerator:
    pass


class LocalEmbeddingGenerator:
    pass


class PineconeVectorStore:
    def __init__(self, namespace="3gpp-specs"):
        self.namespace = namespace


class VectorStore:
    pass


class OpenAILLM:
    pass


class OllamaLLM:
    pass


class AnswerVerifier:
    pass


def _settings(**overrides):
    values = {
        "embedding_provider": "openai",
        "vector_store_provider": "pinecone",
        "llm_provider": "openai",
        "openai_embedding_model": "text-embedding-3-small",
        "embedding_model": "bge-small",
        "pinecone_index_name": "3gpp-rag",
        "pinecone_namespace": "3gpp-specs",
        "openai_model": "gpt-test",
        "openai_verifier_model": "",
        "groq_model": "llama",
        "llm_model": "llama3.2",
        "openai_api_key": "secret-openai",
        "pinecone_api_key": "secret-pinecone",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _chain(embedding=None, vector_store=None, llm=None):
    retriever = SimpleNamespace(
        embedding_generator=embedding or OpenAIEmbeddingGenerator(),
        vector_store=vector_store or PineconeVectorStore(),
    )
    return SimpleNamespace(
        retriever=retriever,
        llm=llm or OpenAILLM(),
        answer_verifier=AnswerVerifier(),
    )


def test_dataset_parsing(tmp_path):
    dataset = tmp_path / "eval.jsonl"
    dataset.write_text(json.dumps(ANSWER_EXAMPLE) + "\n", encoding="utf-8")

    examples = load_dataset(dataset)

    assert examples == [ANSWER_EXAMPLE]


def test_dataset_category_and_limit(tmp_path):
    dataset = tmp_path / "eval.jsonl"
    dataset.write_text(
        json.dumps(ANSWER_EXAMPLE) + "\n" + json.dumps(REFUSAL_EXAMPLE) + "\n",
        encoding="utf-8",
    )

    examples = load_dataset(dataset, category="out_of_domain", limit=1)

    assert examples == [REFUSAL_EXAMPLE]


def test_schema_validation_rejects_bad_expected_behavior():
    bad = {**ANSWER_EXAMPLE, "expected_behavior": "maybe"}

    with pytest.raises(DatasetValidationError):
        validate_example(bad)


def test_schema_validation_requires_gold_for_answerable():
    bad = {**ANSWER_EXAMPLE, "gold_spec_numbers": [], "gold_sources": []}

    with pytest.raises(DatasetValidationError):
        validate_example(bad)


def test_answer_refusal_classification():
    assert classify_behavior(_rag_result(answer_status="answered")) == ANSWERED
    assert classify_behavior(_rag_result(answer_status="refused_evidence")) == REFUSED


def test_hit_at_k_and_mrr():
    metrics = retrieval_metrics(_rag_result()["sources"], ANSWER_EXAMPLE)

    assert metrics["hit_at_1"] == 1.0
    assert metrics["hit_at_3"] == 1.0
    assert metrics["hit_at_5"] == 1.0
    assert metrics["mrr"] == 1.0


def test_reranker_fields_are_recorded_and_aggregated():
    result = score_rag_result(
        ANSWER_EXAMPLE,
        _rag_result(
            sources=[
                {
                    "source_id": "S1",
                    "source": "38401-j10.docx",
                    "similarity": 0.7,
                    "vector_similarity": 0.7,
                    "reranker_score": 0.8,
                    "reranker_raw_score": 1.4,
                    "rank_before_reranking": 2,
                    "rank_after_reranking": 1,
                    "spec_number": "38.401",
                },
                {
                    "source_id": "S2",
                    "source": "38300-j10.docx",
                    "similarity": 0.9,
                    "vector_similarity": 0.9,
                    "reranker_score": 0.4,
                    "reranker_raw_score": -0.4,
                    "rank_before_reranking": 1,
                    "rank_after_reranking": 2,
                    "spec_number": "38.300",
                },
            ]
        ),
    )

    summary = aggregate_results([result])

    assert result["retrieved_results"][0]["rank_before_reranking"] == 2
    assert result["vector_similarity_scores"] == [0.7, 0.9]
    assert result["reranker_scores"] == [0.8, 0.4]
    assert summary["reranker"]["examples_with_scores"] == 1
    assert summary["reranked_retrieval"]["hit_at_1"] == 1.0
    assert summary["vector_retrieval"]["hit_at_1"] == 0.0


def test_hit_at_k_miss():
    result = _rag_result(
        sources=[
            {
                "source_id": "S1",
                "source": "38300-j10.docx",
                "similarity": 0.9,
                "spec_number": "38.300",
            }
        ]
    )

    metrics = retrieval_metrics(result["sources"], ANSWER_EXAMPLE)

    assert metrics["hit_at_1"] == 0.0
    assert metrics["mrr"] == 0.0


def test_false_refusal_calculation():
    refused_answerable = score_rag_result(
        ANSWER_EXAMPLE,
        _rag_result(
            answer="I could not find sufficient supporting evidence.",
            answer_status="refused_evidence",
            evidence={"sufficient": False, "reason": "top_score_below_threshold"},
            verification=None,
        ),
    )

    summary = aggregate_results([refused_answerable])

    assert summary["behavior"]["false_refusal_rate"] == 1.0
    assert summary["failure_analysis"]["false_refusal_evidence_gate"] == ["df-001"]


def test_correct_refusal_calculation():
    refusal = score_rag_result(
        REFUSAL_EXAMPLE,
        _rag_result(
            answer="I could not find sufficient supporting evidence.",
            answer_status="refused_evidence",
            evidence={"sufficient": False, "reason": "top_score_below_threshold"},
            verification=None,
            sources=[],
        ),
    )

    summary = aggregate_results([refusal])

    assert summary["behavior"]["correct_refusal_rate"] == 1.0


def test_unsafe_answer_calculation():
    unsafe = score_rag_result(REFUSAL_EXAMPLE, _rag_result())

    summary = aggregate_results([unsafe])

    assert summary["behavior"]["unsafe_answer_rate"] == 1.0
    assert summary["failure_analysis"]["unsafe_answer"] == ["ood-001"]


def test_evidence_reason_aggregation():
    result = score_rag_result(
        ANSWER_EXAMPLE,
        _rag_result(evidence={"sufficient": False, "reason": "no_documents"}),
    )

    summary = aggregate_results([result])

    assert summary["evidence"]["reason_distribution"]["no_documents"] == 1


def test_verifier_reason_aggregation():
    result = score_rag_result(
        ANSWER_EXAMPLE,
        _rag_result(
            answer="I could not verify a fully grounded answer.",
            answer_status="refused_verification",
            verification={
                "passed": False,
                "reason": "unsupported_claim",
                "citation_valid": True,
                "cited_sources": ["S1"],
                "invalid_citations": [],
            },
        ),
    )

    summary = aggregate_results([result])

    assert summary["verification"]["reason_distribution"]["unsupported_claim"] == 1
    assert summary["verification"]["failure_after_successful_evidence_gate_rate"] == 1.0


def test_citation_validity_calculation():
    result = score_rag_result(ANSWER_EXAMPLE, _rag_result())

    summary = aggregate_results([result])

    assert summary["citations"]["citation_validity_rate"] == 1.0
    assert summary["citations"]["answer_citation_coverage"] == 1.0
    assert summary["citations"]["source_precision"] == 1.0


def test_category_grouping_and_latency():
    result = score_rag_result(ANSWER_EXAMPLE, _rag_result())

    summary = aggregate_results([result])

    assert "direct_fact" in summary["categories"]
    assert summary["latency"]["retrieve_mean"] == 0.1
    assert summary["latency"]["total_median"] == 0.8


def test_error_handling_records_api_error():
    rag = MagicMock()
    rag.query.side_effect = RuntimeError("boom")

    run = run_evaluation(rag, [ANSWER_EXAMPLE])

    case = run["cases"][0]
    assert case["actual_behavior"] == "error"
    assert case["failure_category"] == "api_error"
    assert case["error"] == "boom"


def test_unsupported_escape_rate_is_measured():
    escaped = score_rag_result(
        REFUSAL_EXAMPLE,
        _rag_result(
            verification={
                "passed": False,
                "reason": "unsupported_claim",
                "citation_valid": True,
                "cited_sources": ["S1"],
                "invalid_citations": [],
            }
        ),
    )

    summary = aggregate_results([escaped])

    assert summary["groundedness"]["unsupported_answer_escape_count"] == 1
    assert summary["groundedness"]["unsupported_answer_escape_rate"] == 1.0


def test_report_serialization_contains_core_sections():
    run = run_evaluation(MagicMock(query=MagicMock(return_value=_rag_result())), [ANSWER_EXAMPLE])

    report = render_markdown_report(run)

    assert "# 3GPP Verified RAG Evaluation" in report
    assert "## Core Metrics" in report
    assert "## Failure Analysis" in report


def test_write_outputs(tmp_path):
    run = run_evaluation(MagicMock(query=MagicMock(return_value=_rag_result())), [ANSWER_EXAMPLE])

    files = write_outputs(run, output_dir=tmp_path)

    for path in (files.raw_json, files.csv, files.summary_json, files.report_md):
        assert Path(path).exists()


def test_analyze_reranker_scores_offline(tmp_path):
    artifact = tmp_path / "run.json"
    artifact.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "expected_behavior": "answer",
                        "reranker_scores": [0.9, 0.4],
                    },
                    {
                        "expected_behavior": "answer",
                        "reranker_scores": [0.7],
                    },
                    {
                        "expected_behavior": "refuse",
                        "reranker_scores": [0.8],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    analysis = analyze_reranker_scores(artifact)

    assert analysis["answerable_top_reranker_score"]["count"] == 2
    assert analysis["expected_refusal_top_reranker_score"]["max"] == 0.8
    assert analysis["overlap"]["answerable_at_or_below_refusal_max"] == 1


def test_openai_pinecone_runtime_matches_config():
    settings = _settings()
    chain = _chain()

    assert_runtime_matches_config(settings, chain)
    info = runtime_provider_info(settings, chain)

    assert info["embedding_class"] == "OpenAIEmbeddingGenerator"
    assert info["vector_store_class"] == "PineconeVectorStore"
    assert info["llm_class"] == "OpenAILLM"


def test_openai_config_rejects_local_embedding_runtime():
    settings = _settings(embedding_provider="openai")
    chain = _chain(embedding=LocalEmbeddingGenerator())

    with pytest.raises(RuntimeProviderError, match="LocalEmbeddingGenerator"):
        assert_runtime_matches_config(settings, chain)


def test_pinecone_config_rejects_chroma_runtime():
    settings = _settings(vector_store_provider="pinecone")
    chain = _chain(vector_store=VectorStore())

    with pytest.raises(RuntimeProviderError, match="VectorStore"):
        assert_runtime_matches_config(settings, chain)


def test_namespace_configuration_is_checked():
    settings = _settings(pinecone_namespace="3gpp-specs")
    chain = _chain(vector_store=PineconeVectorStore(namespace="other-namespace"))

    with pytest.raises(RuntimeProviderError, match="PINECONE_NAMESPACE mismatch"):
        assert_runtime_matches_config(settings, chain)


def test_provider_diagnostics_do_not_include_secrets():
    settings = _settings()
    text = format_runtime_diagnostics(runtime_provider_info(settings, _chain()))

    assert "secret-openai" not in text
    assert "secret-pinecone" not in text
    assert "openai / text-embedding-3-small" in text
    assert "pinecone / 3gpp-rag / 3gpp-specs" in text


def test_misconfigured_provider_config_fails_before_runtime_construction():
    settings = _settings(embedding_provider="local", vector_store_provider="chroma")

    with pytest.raises(RuntimeProviderError, match="requires OpenAI/Pinecone"):
        assert_production_provider_config(settings)


def test_production_provider_config_passes():
    assert_production_provider_config(_settings())

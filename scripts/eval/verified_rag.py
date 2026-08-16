"""Quantitative evaluation helpers for the guarded 3GPP RAG pipeline.

This module is intentionally importable and unit-test friendly. The CLI in
``scripts/run_evaluation.py`` is the only place that constructs the real
RAGChain, so pytest can exercise all metrics without OpenAI or Pinecone calls.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
import json
import re
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from scripts.eval.metrics import hit_rate_at_k, mrr


DEFAULT_DATASET = Path("data/eval/3gpp_verified_rag_eval.jsonl")
DEFAULT_OUTPUT_DIR = Path("eval_results")
ANSWER = "answer"
REFUSE = "refuse"
ANSWERED = "answered"
REFUSED = "refused"
ERROR = "error"
SUPPORTED_ESCAPE_REASONS = {"unsupported_claim", "contradiction"}
REFUSAL_STATUSES = {"refused_evidence", "refused_verification", "verifier_error"}
PRODUCTION_PROVIDER_REQUIREMENTS = {
    "embedding_provider": "openai",
    "vector_store_provider": "pinecone",
    "llm_provider": "openai",
}
EXPECTED_RUNTIME_CLASSES = {
    "embedding_provider": {
        "local": "LocalEmbeddingGenerator",
        "openai": "OpenAIEmbeddingGenerator",
    },
    "vector_store_provider": {
        "chroma": "VectorStore",
        "pinecone": "PineconeVectorStore",
    },
    "llm_provider": {
        "ollama": "OllamaLLM",
        "groq": "GroqLLM",
        "openai": "OpenAILLM",
    },
}


class DatasetValidationError(ValueError):
    """Raised when an evaluation example does not match the schema."""


class RuntimeProviderError(RuntimeError):
    """Raised when evaluation provider configuration/runtime is unsafe."""


@dataclass
class EvaluationRunFiles:
    """Paths written by one evaluation run."""

    raw_json: Path
    csv: Path
    summary_json: Path
    report_md: Path


def assert_production_provider_config(settings: Any) -> None:
    """Fail before evaluation if provider settings are not OpenAI/Pinecone.

    This evaluation suite measures the guarded production stack. Failing before
    constructing RAGChain avoids accidental local embedding model downloads when
    `.env` is missing provider switches.
    """
    mismatches = []
    for field, expected in PRODUCTION_PROVIDER_REQUIREMENTS.items():
        actual = _normalize_provider(getattr(settings, field, ""))
        if actual != expected:
            env_name = field.upper()
            mismatches.append(f"{env_name}={actual or '<missing>'} (expected {expected})")
    if mismatches:
        raise RuntimeProviderError(
            "Real evaluation requires OpenAI/Pinecone provider configuration: "
            + "; ".join(mismatches)
        )


def runtime_provider_info(settings: Any, rag_chain: Any) -> Dict[str, Any]:
    """Collect safe provider/runtime diagnostics without secrets."""
    retriever = getattr(rag_chain, "retriever", None)
    embedding = getattr(retriever, "embedding_generator", None)
    vector_store = getattr(retriever, "vector_store", None)
    llm = getattr(rag_chain, "llm", None)
    verifier = getattr(rag_chain, "answer_verifier", None)
    reranker = getattr(retriever, "reranker", None)

    return {
        "embedding_provider": getattr(settings, "embedding_provider", None),
        "embedding_model": (
            getattr(settings, "openai_embedding_model", None)
            if _normalize_provider(getattr(settings, "embedding_provider", "")) == "openai"
            else getattr(settings, "embedding_model", None)
        ),
        "embedding_class": type(embedding).__name__ if embedding is not None else None,
        "vector_store_provider": getattr(settings, "vector_store_provider", None),
        "vector_store_class": type(vector_store).__name__ if vector_store is not None else None,
        "pinecone_index_name": getattr(settings, "pinecone_index_name", None),
        "pinecone_namespace": getattr(settings, "pinecone_namespace", None),
        "runtime_namespace": getattr(vector_store, "namespace", None),
        "llm_provider": getattr(settings, "llm_provider", None),
        "llm_model": _configured_llm_model(settings),
        "llm_class": type(llm).__name__ if llm is not None else None,
        "verifier_model": (
            getattr(settings, "openai_verifier_model", None)
            or getattr(settings, "openai_model", None)
        ),
        "verifier_class": type(verifier).__name__ if verifier is not None else None,
        "reranker_enabled": getattr(retriever, "reranker_enabled", None),
        "reranker_model": getattr(settings, "reranker_model", None),
        "reranker_class": type(reranker).__name__ if reranker is not None else None,
    }


def assert_runtime_matches_config(settings: Any, rag_chain: Any) -> None:
    """Ensure instantiated providers match provider configuration."""
    info = runtime_provider_info(settings, rag_chain)
    errors = []

    for field, class_key in (
        ("embedding_provider", "embedding_class"),
        ("vector_store_provider", "vector_store_class"),
        ("llm_provider", "llm_class"),
    ):
        provider = _normalize_provider(info.get(field))
        expected_class = EXPECTED_RUNTIME_CLASSES.get(field, {}).get(provider)
        actual_class = info.get(class_key)
        if expected_class and actual_class != expected_class:
            errors.append(
                f"{field.upper()}={provider} instantiated {actual_class}, "
                f"expected {expected_class}"
            )

    if _normalize_provider(info.get("vector_store_provider")) == "pinecone":
        configured_namespace = info.get("pinecone_namespace")
        runtime_namespace = info.get("runtime_namespace")
        if configured_namespace != runtime_namespace:
            errors.append(
                "PINECONE_NAMESPACE mismatch: "
                f"configured {configured_namespace}, runtime {runtime_namespace}"
            )

    if errors:
        raise RuntimeProviderError("; ".join(errors))


def format_runtime_diagnostics(info: Dict[str, Any]) -> str:
    """Format safe runtime diagnostics for CLI output."""
    return "\n".join(
        [
            "Runtime RAG providers:",
            f"  Embeddings: {info.get('embedding_provider')} / "
            f"{info.get('embedding_model')} / {info.get('embedding_class')}",
            f"  Vector store: {info.get('vector_store_provider')} / "
            f"{info.get('pinecone_index_name')} / {info.get('runtime_namespace')} / "
            f"{info.get('vector_store_class')}",
            f"  LLM: {info.get('llm_provider')} / {info.get('llm_model')} / "
            f"{info.get('llm_class')}",
            f"  Verifier: {info.get('verifier_model')} / {info.get('verifier_class')}",
            f"  Reranker: {info.get('reranker_enabled')} / "
            f"{info.get('reranker_model')} / {info.get('reranker_class')}",
        ]
    )


def load_dataset(
    path: Path | str = DEFAULT_DATASET,
    *,
    category: Optional[str] = None,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Load and validate a JSONL evaluation dataset."""
    dataset_path = Path(path)
    examples: List[Dict[str, Any]] = []
    with dataset_path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                example = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise DatasetValidationError(
                    f"{dataset_path}:{line_no}: invalid JSON: {exc}"
                ) from exc
            validate_example(example, source=f"{dataset_path}:{line_no}")
            if category and example["category"] != category:
                continue
            examples.append(example)
            if limit is not None and len(examples) >= limit:
                break
    return examples


def validate_example(example: Dict[str, Any], *, source: str = "example") -> None:
    """Validate one dataset row."""
    required_string_fields = ("id", "category", "question", "expected_behavior")
    for field in required_string_fields:
        if not isinstance(example.get(field), str) or not example[field].strip():
            raise DatasetValidationError(f"{source}: '{field}' must be a non-empty string")

    if example["expected_behavior"] not in {ANSWER, REFUSE}:
        raise DatasetValidationError(f"{source}: expected_behavior must be 'answer' or 'refuse'")

    for field in (
        "gold_spec_numbers",
        "gold_sources",
        "required_terms",
        "forbidden_terms",
    ):
        if field not in example:
            raise DatasetValidationError(f"{source}: missing '{field}'")
        if not isinstance(example[field], list) or not all(
            isinstance(value, str) for value in example[field]
        ):
            raise DatasetValidationError(f"{source}: '{field}' must be a list of strings")

    if "gold_answer" in example and not (
        example["gold_answer"] is None or isinstance(example["gold_answer"], str)
    ):
        raise DatasetValidationError(f"{source}: 'gold_answer' must be string or null")

    if example["expected_behavior"] == ANSWER:
        if not example["gold_spec_numbers"] and not example["gold_sources"]:
            raise DatasetValidationError(
                f"{source}: answerable examples need gold specs or gold sources"
            )
    else:
        if example.get("gold_answer") is not None:
            raise DatasetValidationError(f"{source}: refusal examples must use null gold_answer")


def run_evaluation(
    rag_chain: Any,
    examples: Sequence[Dict[str, Any]],
    *,
    run_name: str = "guarded-current",
) -> Dict[str, Any]:
    """Run examples against a RAGChain-like object and aggregate metrics."""
    started = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    results = [evaluate_example(rag_chain, example) for example in examples]
    summary = aggregate_results(results)
    return {
        "run_name": run_name,
        "evaluated_at": started,
        "dataset_size": len(examples),
        "summary": summary,
        "cases": results,
    }


def evaluate_example(rag_chain: Any, example: Dict[str, Any]) -> Dict[str, Any]:
    """Execute and score one example. Exceptions become per-case errors."""
    try:
        result = rag_chain.query(example["question"])
        return score_rag_result(example, result)
    except Exception as exc:
        return score_error(example, exc)


def score_error(example: Dict[str, Any], exc: Exception) -> Dict[str, Any]:
    """Build an error result without aborting the whole run."""
    return {
        **_base_result(example),
        "actual_behavior": ERROR,
        "answer": "",
        "retrieved_sources": [],
        "retrieved_spec_numbers": [],
        "retrieved_results": [],
        "retrieval_similarity_scores": [],
        "vector_similarity_scores": [],
        "reranker_scores": [],
        "evidence": None,
        "evidence_reason": None,
        "verification": None,
        "verification_reason": None,
        "citations": [],
        "retrieval": _empty_retrieval_metrics(),
        "vector_retrieval": _empty_retrieval_metrics(),
        "reranked_retrieval": _empty_retrieval_metrics(),
        "lexical": _lexical_checks("", example),
        "retrieve_time": 0.0,
        "generate_time": 0.0,
        "verify_time": 0.0,
        "total_latency": 0.0,
        "error": str(exc),
        "failure_category": "api_error",
    }


def score_rag_result(example: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
    """Convert one RAGChain result into a complete per-example eval record."""
    sources = result.get("sources", []) or []
    answer = result.get("answer", "") or ""
    evidence = result.get("evidence")
    verification = result.get("verification")
    actual_behavior = classify_behavior(result)
    retrieval = retrieval_metrics(sources, example)
    vector_sources = _sources_in_vector_order(sources)
    vector_retrieval = (
        retrieval_metrics(vector_sources, example)
        if vector_sources
        else retrieval_metrics(sources, example)
    )
    lexical = _lexical_checks(answer, example)
    citations = list((verification or {}).get("cited_sources", []))
    failure_category = classify_failure(
        example=example,
        actual_behavior=actual_behavior,
        retrieval=retrieval,
        evidence=evidence,
        verification=verification,
    )

    return {
        **_base_result(example),
        "actual_behavior": actual_behavior,
        "answer": answer,
        "answer_status": result.get("answer_status"),
        "retrieved_sources": [source.get("source") for source in sources],
        "retrieved_spec_numbers": [
            source.get("spec_number") for source in sources if source.get("spec_number")
        ],
        "retrieved_results": [_retrieved_result(source) for source in sources],
        "retrieval_similarity_scores": [
            source.get("similarity") for source in sources if source.get("similarity") is not None
        ],
        "vector_similarity_scores": [
            source.get("vector_similarity")
            for source in sources
            if source.get("vector_similarity") is not None
        ],
        "reranker_scores": [
            source.get("reranker_score")
            for source in sources
            if source.get("reranker_score") is not None
        ],
        "evidence": evidence,
        "evidence_reason": (evidence or {}).get("reason"),
        "verification": verification,
        "verification_reason": (verification or {}).get("reason"),
        "citations": citations,
        "retrieval": retrieval,
        "lexical": lexical,
        "retrieve_time": float(result.get("retrieve_time") or 0.0),
        "generate_time": float(result.get("generate_time") or 0.0),
        "verify_time": float(result.get("verify_time") or 0.0),
        "total_latency": float(result.get("query_time") or 0.0),
        "error": None,
        "failure_category": failure_category,
        "vector_retrieval": vector_retrieval,
        "reranked_retrieval": retrieval,
    }


def classify_behavior(result: Dict[str, Any]) -> str:
    """Classify final user-visible behavior from RAG metadata."""
    status = result.get("answer_status")
    if status == ANSWERED:
        return ANSWERED
    if status in REFUSAL_STATUSES:
        return REFUSED
    verification = result.get("verification") or {}
    if verification.get("passed") is True:
        return ANSWERED
    answer = (result.get("answer") or "").lower()
    refusal_signals = (
        "could not find sufficient supporting evidence",
        "could not verify a fully grounded answer",
        "cannot answer",
        "no relevant",
        "not enough information",
    )
    if any(signal in answer for signal in refusal_signals):
        return REFUSED
    return ANSWERED if answer.strip() else ERROR


def retrieval_metrics(sources: Sequence[Dict[str, Any]], example: Dict[str, Any]) -> Dict[str, Any]:
    """Compute retrieval Hit@K and MRR when gold labels are available."""
    gold_tokens = _gold_tokens(example)
    if not gold_tokens:
        metrics = _empty_retrieval_metrics()
        metrics["has_gold_labels"] = False
        return metrics

    docs = [_source_as_doc(source) for source in sources]
    return {
        "has_gold_labels": True,
        "hit_at_1": hit_rate_at_k(docs, gold_tokens, k=1),
        "hit_at_3": hit_rate_at_k(docs, gold_tokens, k=3),
        "hit_at_5": hit_rate_at_k(docs, gold_tokens, k=5),
        "mrr": mrr(docs, gold_tokens),
        "first_match_rank": _first_match_rank(docs, gold_tokens),
    }


def classify_failure(
    *,
    example: Dict[str, Any],
    actual_behavior: str,
    retrieval: Dict[str, Any],
    evidence: Optional[Dict[str, Any]],
    verification: Optional[Dict[str, Any]],
) -> Optional[str]:
    """Assign one primary failure bucket for Prompt 8 analysis."""
    expected = example["expected_behavior"]
    if actual_behavior == ERROR:
        return "api_error"

    if expected == REFUSE and actual_behavior == ANSWERED:
        return "unsafe_answer"

    if expected == ANSWER and actual_behavior == REFUSED:
        if evidence and evidence.get("sufficient") is False:
            return "false_refusal_evidence_gate"
        if verification and verification.get("passed") is False:
            reason = verification.get("reason")
            if reason == "invalid_citation":
                return "invalid_citation"
            if reason == "missing_citations":
                return "missing_citation"
            return "verification_rejection"
        return "false_refusal"

    if expected == ANSWER and retrieval.get("has_gold_labels") and retrieval.get("hit_at_5") == 0.0:
        return "retrieval_miss"

    if verification and actual_behavior == ANSWERED:
        if verification.get("reason") in SUPPORTED_ESCAPE_REASONS:
            return "unsafe_answer"

    return None


def aggregate_results(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-example records into run-level and category metrics."""
    total = len(results)
    answer_expected = [r for r in results if r["expected_behavior"] == ANSWER]
    refusal_expected = [r for r in results if r["expected_behavior"] == REFUSE]
    answered = [r for r in results if r["actual_behavior"] == ANSWERED]
    refused = [r for r in results if r["actual_behavior"] == REFUSED]
    errors = [r for r in results if r["actual_behavior"] == ERROR]

    summary = {
        "dataset": {
            "total": total,
            "answerable": len(answer_expected),
            "expected_refusals": len(refusal_expected),
            "errors": len(errors),
        },
        "retrieval": aggregate_retrieval(results),
        "vector_retrieval": aggregate_metric_group(results, "vector_retrieval"),
        "reranked_retrieval": aggregate_metric_group(results, "reranked_retrieval"),
        "reranker": aggregate_reranker(results),
        "behavior": {
            "answer_rate": _rate(len(answered), total),
            "refusal_rate": _rate(len(refused), total),
            "correct_answer_attempt_rate": _rate(
                sum(1 for r in answer_expected if r["actual_behavior"] == ANSWERED),
                len(answer_expected),
            ),
            "correct_refusal_rate": _rate(
                sum(1 for r in refusal_expected if r["actual_behavior"] == REFUSED),
                len(refusal_expected),
            ),
            "false_refusal_rate": _rate(
                sum(1 for r in answer_expected if r["actual_behavior"] == REFUSED),
                len(answer_expected),
            ),
            "unsafe_answer_rate": _rate(
                sum(1 for r in refusal_expected if r["actual_behavior"] == ANSWERED),
                len(refusal_expected),
            ),
        },
        "evidence": aggregate_evidence(results),
        "verification": aggregate_verification(results),
        "citations": aggregate_citations(results),
        "groundedness": aggregate_groundedness(results),
        "latency": aggregate_latency(results),
        "categories": {},
        "failure_analysis": aggregate_failures(results),
    }
    for category in sorted({r["category"] for r in results}):
        category_results = [r for r in results if r["category"] == category]
        summary["categories"][category] = {
            "count": len(category_results),
            "behavior": (
                aggregate_results(category_results)["behavior"]
                if len(category_results) != total
                else summary["behavior"]
            ),
            "evidence": aggregate_evidence(category_results),
            "verification": aggregate_verification(category_results),
            "latency": aggregate_latency(category_results),
            "failures": aggregate_failures(category_results),
        }
    return summary


def aggregate_retrieval(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return aggregate_metric_group(results, "retrieval")


def aggregate_metric_group(results: Sequence[Dict[str, Any]], field: str) -> Dict[str, Any]:
    labeled = [r for r in results if (r.get(field) or {}).get("has_gold_labels")]
    return {
        "labeled_examples": len(labeled),
        "hit_at_1": _mean_metric(labeled, (field, "hit_at_1")),
        "hit_at_3": _mean_metric(labeled, (field, "hit_at_3")),
        "hit_at_5": _mean_metric(labeled, (field, "hit_at_5")),
        "mrr": _mean_metric(labeled, (field, "mrr")),
    }


def aggregate_reranker(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    with_scores = [r for r in results if r.get("reranker_scores")]
    return {
        "examples_with_scores": len(with_scores),
        "top_score_mean": _mean(r["reranker_scores"][0] for r in with_scores),
        "top_score_median": _median(r["reranker_scores"][0] for r in with_scores),
        "gold_hit_at_1_after_reranking": _mean_metric(
            with_scores, ("reranked_retrieval", "hit_at_1")
        ),
        "gold_hit_at_3_after_reranking": _mean_metric(
            with_scores, ("reranked_retrieval", "hit_at_3")
        ),
        "gold_mrr_after_reranking": _mean_metric(with_scores, ("reranked_retrieval", "mrr")),
    }


def aggregate_evidence(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    with_evidence = [r for r in results if r.get("evidence")]
    passed = [r for r in with_evidence if r["evidence"].get("sufficient") is True]
    return {
        "pass_rate": _rate(len(passed), len(with_evidence)),
        "rejection_rate": _rate(len(with_evidence) - len(passed), len(with_evidence)),
        "reason_distribution": _distribution(r.get("evidence_reason") for r in with_evidence),
        "by_expected_behavior": _reason_split(with_evidence, "evidence_reason"),
    }


def aggregate_verification(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    with_verification = [r for r in results if r.get("verification")]
    passed = [r for r in with_verification if r["verification"].get("passed") is True]
    evidence_passed = [r for r in results if (r.get("evidence") or {}).get("sufficient") is True]
    failures_after_gate = [
        r for r in evidence_passed if (r.get("verification") or {}).get("passed") is False
    ]
    return {
        "pass_rate": _rate(len(passed), len(with_verification)),
        "rejection_rate": _rate(len(with_verification) - len(passed), len(with_verification)),
        "reason_distribution": _distribution(
            r.get("verification_reason") for r in with_verification
        ),
        "failure_after_successful_evidence_gate_rate": _rate(
            len(failures_after_gate), len(evidence_passed)
        ),
        "by_expected_behavior": _reason_split(with_verification, "verification_reason"),
    }


def aggregate_citations(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    verified_answers = [
        r
        for r in results
        if r["actual_behavior"] == ANSWERED and (r.get("verification") or {}).get("passed") is True
    ]
    cited_total = 0
    invalid_total = 0
    with_valid_citation = 0
    source_precision_values = []
    for result in verified_answers:
        verification = result.get("verification") or {}
        citations = verification.get("cited_sources", []) or []
        invalid = verification.get("invalid_citations", []) or []
        cited_total += len(citations)
        invalid_total += len(invalid)
        if citations and verification.get("citation_valid") is True:
            with_valid_citation += 1
        precision = source_precision(result)
        if precision is not None:
            source_precision_values.append(precision)
    valid_total = max(0, cited_total - invalid_total)
    return {
        "citation_validity_rate": _rate(valid_total, cited_total),
        "answer_citation_coverage": _rate(with_valid_citation, len(verified_answers)),
        "source_precision": _mean(source_precision_values),
    }


def aggregate_groundedness(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    released = [r for r in results if r["actual_behavior"] == ANSWERED]
    verified_released = [r for r in released if (r.get("verification") or {}).get("passed") is True]
    escapes = [
        r
        for r in released
        if (r.get("verification") or {}).get("reason") in SUPPORTED_ESCAPE_REASONS
    ]
    return {
        "verified_grounded_answer_rate": _rate(len(verified_released), len(released)),
        "unsupported_answer_escape_rate": _rate(len(escapes), len(results)),
        "unsupported_answer_escape_count": len(escapes),
    }


def aggregate_latency(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "retrieve_mean": _mean(r["retrieve_time"] for r in results),
        "retrieve_median": _median(r["retrieve_time"] for r in results),
        "generate_mean": _mean(r["generate_time"] for r in results),
        "verify_mean": _mean(r["verify_time"] for r in results),
        "verify_median": _median(r["verify_time"] for r in results),
        "total_mean": _mean(r["total_latency"] for r in results),
        "total_median": _median(r["total_latency"] for r in results),
    }


def aggregate_failures(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    grouped: Dict[str, List[str]] = {}
    for result in results:
        category = result.get("failure_category")
        if category:
            grouped.setdefault(category, []).append(result["id"])
    return grouped


def source_precision(result: Dict[str, Any]) -> Optional[float]:
    """Measure how many cited source IDs map to expected gold sources."""
    citations = (result.get("verification") or {}).get("cited_sources", []) or []
    if not citations:
        return None
    gold_tokens = _gold_tokens(result)
    if not gold_tokens:
        return None
    sources = result.get("retrieved_sources", []) or []
    hits = 0
    for citation in citations:
        match = re.fullmatch(r"S(\d+)", citation)
        if not match:
            continue
        index = int(match.group(1)) - 1
        if 0 <= index < len(sources):
            if _matches_any_token(str(sources[index]), gold_tokens):
                hits += 1
    return _rate(hits, len(citations))


def write_outputs(
    run: Dict[str, Any],
    *,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
) -> EvaluationRunFiles:
    """Write timestamped JSON, CSV, summary JSON, and Markdown report."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    prefix = out_dir / f"run_{stamp}"
    files = EvaluationRunFiles(
        raw_json=prefix.with_suffix(".json"),
        csv=prefix.with_suffix(".csv"),
        summary_json=prefix.with_name(prefix.name + "_summary.json"),
        report_md=prefix.with_name(prefix.name + "_report.md"),
    )
    files.raw_json.write_text(json.dumps(run, indent=2), encoding="utf-8")
    files.summary_json.write_text(json.dumps(run["summary"], indent=2), encoding="utf-8")
    write_csv(run["cases"], files.csv)
    files.report_md.write_text(render_markdown_report(run), encoding="utf-8")
    return files


def write_csv(cases: Sequence[Dict[str, Any]], path: Path) -> None:
    """Write a flat inspection-friendly CSV."""
    fields = [
        "id",
        "category",
        "question",
        "expected_behavior",
        "actual_behavior",
        "answer_status",
        "failure_category",
        "evidence_reason",
        "verification_reason",
        "retrieve_time",
        "generate_time",
        "verify_time",
        "total_latency",
        "retrieved_sources",
        "retrieved_results",
        "retrieval_similarity_scores",
        "vector_similarity_scores",
        "reranker_scores",
        "citations",
        "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for case in cases:
            row = {field: case.get(field) for field in fields}
            for field in (
                "retrieved_sources",
                "retrieved_results",
                "retrieval_similarity_scores",
                "vector_similarity_scores",
                "reranker_scores",
                "citations",
            ):
                row[field] = json.dumps(row[field])
            writer.writerow(row)


def render_markdown_report(run: Dict[str, Any]) -> str:
    """Render the human-readable Markdown report."""
    summary = run["summary"]
    retrieval = summary["retrieval"]
    vector_retrieval = summary.get("vector_retrieval", {})
    reranked_retrieval = summary.get("reranked_retrieval", {})
    behavior = summary["behavior"]
    evidence = summary["evidence"]
    verification = summary["verification"]
    citations = summary["citations"]
    groundedness = summary["groundedness"]
    latency = summary["latency"]
    lines = [
        "# 3GPP Verified RAG Evaluation",
        "",
        f"Run name: {run['run_name']}",
        f"Evaluated at: {run['evaluated_at']}",
        f"Dataset size: {summary['dataset']['total']}",
        f"Answerable: {summary['dataset']['answerable']}",
        f"Expected refusals: {summary['dataset']['expected_refusals']}",
        "",
        "## Core Metrics",
        "",
        f"- Retrieval Hit@1: {_fmt(retrieval['hit_at_1'])}",
        f"- Retrieval Hit@3: {_fmt(retrieval['hit_at_3'])}",
        f"- Retrieval Hit@5: {_fmt(retrieval['hit_at_5'])}",
        f"- Retrieval MRR: {_fmt(retrieval['mrr'])}",
        f"- Vector-order Hit@1: {_fmt(vector_retrieval.get('hit_at_1'))}",
        f"- Reranked Hit@1: {_fmt(reranked_retrieval.get('hit_at_1'))}",
        f"- Reranked Hit@3: {_fmt(reranked_retrieval.get('hit_at_3'))}",
        f"- Reranked MRR: {_fmt(reranked_retrieval.get('mrr'))}",
        f"- Correct Answer/Attempt Rate: {_fmt(behavior['correct_answer_attempt_rate'])}",
        f"- Correct Refusal Rate: {_fmt(behavior['correct_refusal_rate'])}",
        f"- False Refusal Rate: {_fmt(behavior['false_refusal_rate'])}",
        f"- Unsafe Answer Rate: {_fmt(behavior['unsafe_answer_rate'])}",
        f"- Evidence Pass Rate: {_fmt(evidence['pass_rate'])}",
        f"- Verifier Pass Rate: {_fmt(verification['pass_rate'])}",
        f"- Citation Validity Rate: {_fmt(citations['citation_validity_rate'])}",
        f"- Citation Coverage: {_fmt(citations['answer_citation_coverage'])}",
        f"- Unsupported Answer Escape Rate: {_fmt(groundedness['unsupported_answer_escape_rate'])}",
        f"- Mean Retrieval Latency: {latency['retrieve_mean']:.3f}s",
        f"- Median Retrieval Latency: {latency['retrieve_median']:.3f}s",
        f"- Mean Total Latency: {latency['total_mean']:.3f}s",
        f"- Median Total Latency: {latency['total_median']:.3f}s",
        "",
        "## Category Breakdown",
        "",
    ]
    for category, category_summary in summary["categories"].items():
        category_behavior = category_summary["behavior"]
        lines.extend(
            [
                f"### {category}",
                "",
                f"- Count: {category_summary['count']}",
                f"- Correct Answer/Attempt Rate: {_fmt(category_behavior['correct_answer_attempt_rate'])}",
                f"- Correct Refusal Rate: {_fmt(category_behavior['correct_refusal_rate'])}",
                f"- False Refusal Rate: {_fmt(category_behavior['false_refusal_rate'])}",
                f"- Unsafe Answer Rate: {_fmt(category_behavior['unsafe_answer_rate'])}",
                "",
            ]
        )
    lines.extend(["## Failure Analysis", ""])
    failures = summary["failure_analysis"]
    if failures:
        for reason, case_ids in sorted(failures.items()):
            lines.append(f"- {reason}: {', '.join(case_ids)}")
    else:
        lines.append("- No failures recorded.")
    lines.append("")
    return "\n".join(lines)


def _base_result(example: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": example["id"],
        "category": example["category"],
        "question": example["question"],
        "expected_behavior": example["expected_behavior"],
        "gold_spec_numbers": example.get("gold_spec_numbers", []),
        "gold_sources": example.get("gold_sources", []),
        "required_terms": example.get("required_terms", []),
        "forbidden_terms": example.get("forbidden_terms", []),
        "notes": example.get("notes", ""),
    }


def _empty_retrieval_metrics() -> Dict[str, Any]:
    return {
        "has_gold_labels": True,
        "hit_at_1": None,
        "hit_at_3": None,
        "hit_at_5": None,
        "mrr": None,
        "first_match_rank": None,
    }


def _gold_tokens(example: Dict[str, Any]) -> List[str]:
    tokens = list(example.get("gold_sources", []) or [])
    for spec in example.get("gold_spec_numbers", []) or []:
        token = spec.replace(".", "")
        if token:
            tokens.append(token)
    deduped = []
    for token in tokens:
        if token not in deduped:
            deduped.append(token)
    return deduped


def _source_as_doc(source: Dict[str, Any]) -> Dict[str, Any]:
    spec_number = source.get("spec_number")
    spec_token = spec_number.replace(".", "") if isinstance(spec_number, str) else ""
    source_text = " ".join(
        str(value) for value in (source.get("source"), spec_token) if value and value != "unknown"
    )
    return {"source": source_text, "similarity": source.get("similarity", 0.0)}


def _retrieved_result(source: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "source": source.get("source"),
        "spec_number": source.get("spec_number"),
        "similarity": source.get("similarity"),
        "vector_similarity": source.get("vector_similarity"),
        "reranker_score": source.get("reranker_score"),
        "reranker_raw_score": source.get("reranker_raw_score"),
        "rank_before_reranking": source.get("rank_before_reranking"),
        "rank_after_reranking": source.get("rank_after_reranking"),
    }


def _sources_in_vector_order(sources: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ranked = [source for source in sources if isinstance(source.get("rank_before_reranking"), int)]
    if not ranked:
        return []
    return sorted(ranked, key=lambda source: source["rank_before_reranking"])


def _first_match_rank(docs: Sequence[Dict[str, Any]], gold_tokens: Sequence[str]) -> Optional[int]:
    for rank, doc in enumerate(docs, 1):
        if _matches_any_token(doc.get("source", ""), gold_tokens):
            return rank
    return None


def _matches_any_token(text: str, tokens: Sequence[str]) -> bool:
    for token in tokens:
        pattern = r"(?<!\d)" + re.escape(token) + r"(?!\d)"
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False


def _lexical_checks(answer: str, example: Dict[str, Any]) -> Dict[str, Any]:
    lower_answer = answer.lower()
    required = example.get("required_terms", []) or []
    forbidden = example.get("forbidden_terms", []) or []
    required_hits = [term for term in required if term.lower() in lower_answer]
    forbidden_hits = [term for term in forbidden if term.lower() in lower_answer]
    spec_hits = [
        spec for spec in example.get("gold_spec_numbers", []) if spec.lower() in lower_answer
    ]
    return {
        "required_term_coverage": _rate(len(required_hits), len(required)),
        "required_terms_found": required_hits,
        "forbidden_terms_found": forbidden_hits,
        "expected_spec_reference_present": (
            bool(spec_hits) if example.get("gold_spec_numbers") else None
        ),
    }


def _reason_split(
    results: Sequence[Dict[str, Any]], reason_field: str
) -> Dict[str, Dict[str, int]]:
    split = {}
    for expected in (ANSWER, REFUSE):
        subset = [r for r in results if r["expected_behavior"] == expected]
        split[expected] = _distribution(r.get(reason_field) for r in subset)
    return split


def _distribution(values: Iterable[Optional[str]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for value in values:
        if not value:
            continue
        counts[value] = counts.get(value, 0) + 1
    return counts


def _mean_metric(results: Sequence[Dict[str, Any]], path: Sequence[str]) -> Optional[float]:
    values = []
    for result in results:
        value: Any = result
        for key in path:
            value = value.get(key)
            if value is None:
                break
        if isinstance(value, (int, float)):
            values.append(float(value))
    return _mean(values)


def _rate(numerator: int, denominator: int) -> Optional[float]:
    if denominator == 0:
        return None
    return round(numerator / denominator, 4)


def _mean(values: Iterable[float]) -> Optional[float]:
    data = [float(value) for value in values]
    if not data:
        return None
    return round(sum(data) / len(data), 4)


def _median(values: Iterable[float]) -> Optional[float]:
    data = [float(value) for value in values]
    if not data:
        return None
    return round(statistics.median(data), 4)


def _fmt(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    return f"{value:.1%}"


def _normalize_provider(value: Optional[str]) -> str:
    return (value or "").strip().lower()


def _configured_llm_model(settings: Any) -> Optional[str]:
    provider = _normalize_provider(getattr(settings, "llm_provider", ""))
    if provider == "openai":
        return getattr(settings, "openai_model", None)
    if provider == "groq":
        return getattr(settings, "groq_model", None)
    return getattr(settings, "llm_model", None)

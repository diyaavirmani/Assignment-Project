"""Run retrieval-only reranker benchmarking for the verified 3GPP dataset.

This script executes only:

    OpenAI embeddings -> Pinecone retrieval -> cross-encoder reranking

It does not invoke OpenAI generation or AnswerVerifier. The output artifact is
intended for offline evidence-gate calibration and score-distribution analysis.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.eval.verified_rag import (  # noqa: E402
    DEFAULT_DATASET,
    DEFAULT_OUTPUT_DIR,
    assert_production_provider_config,
    load_dataset,
)
from src.config import settings  # noqa: E402
from src.core.reranker import CrossEncoderReranker  # noqa: E402
from src.core.retriever import DocumentRetriever  # noqa: E402


DEFAULT_SPLIT = Path("data/eval/eval_split.json")


@dataclass
class RetrievalOnlyFiles:
    raw_json: Path
    summary_json: Path
    report_md: Path


class RecordingReranker:
    """Capture full reranked candidate lists while returning the requested top-N."""

    def __init__(self, reranker: CrossEncoderReranker) -> None:
        self.reranker = reranker
        self.last_candidates: List[Dict[str, Any]] = []
        self.latencies: List[float] = []

    def rerank(self, query: str, candidates: Sequence[Dict], top_n: int = 5) -> List[Dict]:
        start = time.perf_counter()
        ranked = self.reranker.rerank(query, candidates, top_n=len(candidates))
        self.latencies.append(time.perf_counter() - start)
        self.last_candidates = [dict(candidate) for candidate in ranked]
        return ranked[:top_n]


def run_benchmark(
    examples: Sequence[Dict[str, Any]],
    *,
    split: Dict[str, List[str]],
    run_name: str,
    top_k: Optional[int] = None,
) -> Dict[str, Any]:
    started = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    base_reranker = CrossEncoderReranker(model_name=settings.reranker_model, device="cpu")
    recording_reranker = RecordingReranker(base_reranker)
    retriever = DocumentRetriever(
        reranker=recording_reranker,
        reranker_enabled=True,
        top_k=top_k or settings.reranker_top_k,
    )

    cases = []
    for index, example in enumerate(examples, start=1):
        print(f"[{index}/{len(examples)}] {example['id']} {example['question'][:80]}")
        start = time.perf_counter()
        retriever.retrieve(example["question"], top_k=top_k)
        latency = time.perf_counter() - start
        candidates = recording_reranker.last_candidates
        cases.append(
            build_case_record(
                example,
                candidates,
                split_name=split_name_for(example["id"], split),
                retrieve_latency=latency,
                rerank_latency=(
                    recording_reranker.latencies[-1] if recording_reranker.latencies else 0.0
                ),
            )
        )

    summary = summarize_cases(cases, recording_reranker.latencies)
    return {
        "run_name": run_name,
        "evaluated_at": started,
        "dataset": str(DEFAULT_DATASET),
        "split": str(DEFAULT_SPLIT),
        "dataset_size": len(cases),
        "config": {
            "embedding_provider": settings.embedding_provider,
            "embedding_model": settings.openai_embedding_model,
            "vector_store_provider": settings.vector_store_provider,
            "pinecone_index_name": settings.pinecone_index_name,
            "pinecone_namespace": settings.pinecone_namespace,
            "reranker_model": settings.reranker_model,
            "candidate_k": settings.reranker_candidate_k,
            "top_k": top_k or settings.reranker_top_k,
        },
        "summary": summary,
        "cases": cases,
    }


def build_case_record(
    example: Dict[str, Any],
    candidates: Sequence[Dict[str, Any]],
    *,
    split_name: Optional[str],
    retrieve_latency: float,
    rerank_latency: float,
) -> Dict[str, Any]:
    by_vector = sorted(
        candidates,
        key=lambda doc: int(doc.get("rank_before_reranking") or 10**9),
    )
    by_rerank = sorted(
        candidates,
        key=lambda doc: int(doc.get("rank_after_reranking") or 10**9),
    )
    vector_scores = [_score(doc, "vector_similarity") for doc in by_vector]
    reranker_scores = [_score(doc, "reranker_score") for doc in by_rerank]
    top_reranker = reranker_scores[0] if reranker_scores else 0.0
    second_best = reranker_scores[1] if len(reranker_scores) > 1 else None

    return {
        "id": example["id"],
        "category": example["category"],
        "question": example["question"],
        "expected_behavior": example["expected_behavior"],
        "split": split_name,
        "gold_spec_numbers": example.get("gold_spec_numbers", []),
        "gold_sources": example.get("gold_sources", []),
        "top_vector_score": vector_scores[0] if vector_scores else 0.0,
        "top_reranker_score": top_reranker,
        "mean_top3_vector_score": mean(vector_scores[:3]),
        "mean_top3_reranker_score": mean(reranker_scores[:3]),
        "second_best_reranker_score": second_best,
        "reranker_margin": top_reranker - second_best if second_best is not None else None,
        "vector_scores": vector_scores,
        "reranker_scores": reranker_scores,
        "retrieve_latency": round(retrieve_latency, 4),
        "rerank_latency": round(rerank_latency, 4),
        "candidates": [candidate_record(doc) for doc in by_rerank],
    }


def candidate_record(doc: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "source": doc.get("source"),
        "spec_number": doc.get("spec_number"),
        "vector_rank": doc.get("rank_before_reranking"),
        "vector_similarity": doc.get("vector_similarity", doc.get("similarity")),
        "reranked_rank": doc.get("rank_after_reranking"),
        "reranker_raw_score": doc.get("reranker_raw_score"),
        "reranker_score": doc.get("reranker_score"),
    }


def summarize_cases(
    cases: Sequence[Dict[str, Any]],
    rerank_latencies: Sequence[float],
) -> Dict[str, Any]:
    return {
        "dataset": {
            "total": len(cases),
            "answerable": sum(1 for case in cases if case["expected_behavior"] == "answer"),
            "expected_refusals": sum(1 for case in cases if case["expected_behavior"] == "refuse"),
        },
        "top_vector_score": separation_summary(cases, "top_vector_score"),
        "top_reranker_score": separation_summary(cases, "top_reranker_score"),
        "mean_top3_vector_score": separation_summary(cases, "mean_top3_vector_score"),
        "mean_top3_reranker_score": separation_summary(cases, "mean_top3_reranker_score"),
        "reranker_margin": separation_summary(cases, "reranker_margin"),
        "latency": {
            "rerank_mean": round(mean(rerank_latencies), 4),
            "rerank_median": round(median(rerank_latencies), 4),
            "retrieve_mean": round(mean(case["retrieve_latency"] for case in cases), 4),
        },
    }


def separation_summary(cases: Sequence[Dict[str, Any]], field: str) -> Dict[str, Any]:
    answerable = [
        float(case[field])
        for case in cases
        if case["expected_behavior"] == "answer" and case.get(field) is not None
    ]
    refusals = [
        float(case[field])
        for case in cases
        if case["expected_behavior"] == "refuse" and case.get(field) is not None
    ]
    return {
        "answer": describe(answerable),
        "refuse": describe(refusals),
        "overlap": overlap(answerable, refusals),
    }


def describe(values: Sequence[float]) -> Dict[str, Any]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "p25": None,
            "p50": None,
            "p75": None,
            "p90": None,
        }
    return {
        "count": len(ordered),
        "min": round(ordered[0], 4),
        "max": round(ordered[-1], 4),
        "mean": round(mean(ordered), 4),
        "median": round(statistics.median(ordered), 4),
        "p25": round(percentile(ordered, 0.25), 4),
        "p50": round(percentile(ordered, 0.50), 4),
        "p75": round(percentile(ordered, 0.75), 4),
        "p90": round(percentile(ordered, 0.90), 4),
    }


def overlap(answerable: Sequence[float], refusals: Sequence[float]) -> Dict[str, Any]:
    if not answerable or not refusals:
        return {
            "answerable_range": None,
            "refusal_range": None,
            "answerable_at_or_below_refusal_max": 0,
            "refusals_at_or_above_answerable_min": 0,
        }
    answer_min = min(answerable)
    answer_max = max(answerable)
    refusal_min = min(refusals)
    refusal_max = max(refusals)
    return {
        "answerable_range": [round(answer_min, 4), round(answer_max, 4)],
        "refusal_range": [round(refusal_min, 4), round(refusal_max, 4)],
        "ranges_overlap": answer_min <= refusal_max,
        "answerable_at_or_below_refusal_max": sum(score <= refusal_max for score in answerable),
        "refusals_at_or_above_answerable_min": sum(score >= answer_min for score in refusals),
    }


def write_outputs(run: Dict[str, Any], output_dir: Path | str) -> RetrievalOnlyFiles:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    prefix = out_dir / f"reranker_retrieval_{stamp}"
    files = RetrievalOnlyFiles(
        raw_json=prefix.with_suffix(".json"),
        summary_json=prefix.with_name(prefix.name + "_summary.json"),
        report_md=prefix.with_name(prefix.name + "_report.md"),
    )
    files.raw_json.write_text(json.dumps(run, indent=2), encoding="utf-8")
    files.summary_json.write_text(json.dumps(run["summary"], indent=2), encoding="utf-8")
    files.report_md.write_text(render_report(run), encoding="utf-8")
    return files


def render_report(run: Dict[str, Any]) -> str:
    summary = run["summary"]
    top_vector = summary["top_vector_score"]["overlap"]
    top_reranker = summary["top_reranker_score"]["overlap"]
    mean_reranker = summary["mean_top3_reranker_score"]["overlap"]
    margin = summary["reranker_margin"]["overlap"]
    return "\n".join(
        [
            "# Reranker Retrieval-Only Benchmark",
            "",
            f"Run name: {run['run_name']}",
            f"Evaluated at: {run['evaluated_at']}",
            f"Dataset size: {run['dataset_size']}",
            f"Reranker model: `{run['config']['reranker_model']}`",
            f"Candidate K: {run['config']['candidate_k']}",
            f"Top K: {run['config']['top_k']}",
            "",
            "## Overlap",
            "",
            f"- Top vector answerable range: {top_vector['answerable_range']}",
            f"- Top vector refusal range: {top_vector['refusal_range']}",
            f"- Top vector answerable <= refusal max: "
            f"{top_vector['answerable_at_or_below_refusal_max']}",
            f"- Top vector refusals >= answerable min: "
            f"{top_vector['refusals_at_or_above_answerable_min']}",
            f"- Top reranker answerable range: {top_reranker['answerable_range']}",
            f"- Top reranker refusal range: {top_reranker['refusal_range']}",
            f"- Top reranker answerable <= refusal max: "
            f"{top_reranker['answerable_at_or_below_refusal_max']}",
            f"- Top reranker refusals >= answerable min: "
            f"{top_reranker['refusals_at_or_above_answerable_min']}",
            f"- Mean top-3 reranker overlap answerable <= refusal max: "
            f"{mean_reranker['answerable_at_or_below_refusal_max']}",
            f"- Reranker margin overlap answerable <= refusal max: "
            f"{margin['answerable_at_or_below_refusal_max']}",
            "",
            "## Latency",
            "",
            f"- Mean retrieval latency: {summary['latency']['retrieve_mean']:.4f}s",
            f"- Mean warm rerank latency: {summary['latency']['rerank_mean']:.4f}s",
            f"- Median warm rerank latency: {summary['latency']['rerank_median']:.4f}s",
            "",
        ]
    )


def load_split(path: Path | str) -> Dict[str, List[str]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {
        "calibration": list(data["calibration"]),
        "validation": list(data["validation"]),
    }


def split_name_for(case_id: str, split: Dict[str, List[str]]) -> Optional[str]:
    if case_id in split.get("calibration", []):
        return "calibration"
    if case_id in split.get("validation", []):
        return "validation"
    return None


def validate_split_coverage(
    examples: Sequence[Dict[str, Any]], split: Dict[str, List[str]]
) -> None:
    example_ids = {example["id"] for example in examples}
    split_ids = set(split.get("calibration", [])) | set(split.get("validation", []))
    if example_ids != split_ids:
        raise ValueError("Split file does not cover the retrieval benchmark dataset exactly")


def mean(values: Iterable[float]) -> float:
    data = [float(value) for value in values]
    return sum(data) / len(data) if data else 0.0


def median(values: Iterable[float]) -> float:
    data = [float(value) for value in values]
    return statistics.median(data) if data else 0.0


def percentile(values: Sequence[float], q: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _score(doc: Dict[str, Any], field: str) -> float:
    value = doc.get(field)
    if value is None and field == "vector_similarity":
        value = doc.get("similarity")
    return float(value or 0.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run retrieval-only reranker benchmark")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--split-file", default=str(DEFAULT_SPLIT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--run-name", default="reranker-retrieval-only")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print("WARNING: this retrieval-only benchmark uses OpenAI embeddings and Pinecone.")
    print("It does not call OpenAI generation or AnswerVerifier.\n")
    assert_production_provider_config(settings)
    examples = load_dataset(args.dataset, limit=args.limit)
    split = load_split(args.split_file)
    if args.limit is None:
        validate_split_coverage(examples, split)
    run = run_benchmark(examples, split=split, run_name=args.run_name)
    files = write_outputs(run, args.output_dir)
    print("\nRetrieval-only benchmark complete.")
    print(f"Raw JSON:     {files.raw_json}")
    print(f"Summary JSON: {files.summary_json}")
    print(f"Report:       {files.report_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

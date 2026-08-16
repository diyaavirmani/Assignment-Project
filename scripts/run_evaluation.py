"""Run the guarded 3GPP RAG evaluation suite.

This script executes the real configured RAG stack. It may call OpenAI and
Pinecone and may incur external API costs. Unit tests import
``scripts.eval.verified_rag`` instead and never run this CLI.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.eval.verified_rag import (  # noqa: E402
    DEFAULT_DATASET,
    DEFAULT_OUTPUT_DIR,
    assert_production_provider_config,
    assert_runtime_matches_config,
    format_runtime_diagnostics,
    load_dataset,
    run_evaluation,
    runtime_provider_info,
    write_outputs,
)
from src.config import settings  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run quantitative evaluation for the guarded 3GPP RAG stack."
    )
    parser.add_argument(
        "--dataset",
        default=str(DEFAULT_DATASET),
        help=f"JSONL dataset path (default: {DEFAULT_DATASET})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N matching examples.",
    )
    parser.add_argument(
        "--category",
        default=None,
        help="Evaluate only one category, e.g. direct_fact or out_of_domain.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Directory for timestamped outputs (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--run-name",
        default="guarded-current",
        help="Human-readable run label for later comparisons.",
    )
    parser.add_argument(
        "--show-runtime",
        action="store_true",
        help="Print safe provider diagnostics and exit before running queries.",
    )
    parser.add_argument(
        "--split",
        choices=["calibration", "validation"],
        default=None,
        help="Evaluate only IDs from data/eval/eval_split.json.",
    )
    parser.add_argument(
        "--split-file",
        default="data/eval/eval_split.json",
        help="Fixed calibration/validation split definition.",
    )
    parser.add_argument("--evidence-min-top-score", type=float, default=None)
    parser.add_argument("--evidence-min-doc-score", type=float, default=None)
    parser.add_argument("--evidence-min-mean-score", type=float, default=None)
    parser.add_argument("--evidence-min-docs", type=int, default=None)
    parser.add_argument("--evidence-score-source", choices=["vector", "reranker"], default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print("WARNING: this evaluation uses the real configured RAG stack.")
    print("It may call OpenAI and Pinecone and may incur external API costs.")
    print("No API keys or hidden prompts are written to evaluation outputs.\n")

    assert_production_provider_config(settings)

    examples = load_dataset(args.dataset, category=args.category, limit=None)
    if args.split:
        split_ids = _load_split_ids(args.split_file, args.split)
        examples = [example for example in examples if example["id"] in split_ids]
    if args.limit is not None:
        examples = examples[: args.limit]
    if not examples:
        raise SystemExit("No evaluation examples matched the requested filters.")

    print(f"Loaded {len(examples)} examples from {args.dataset}")
    if args.category:
        print(f"Category filter: {args.category}")
    if args.split:
        print(f"Split filter: {args.split}")
    if args.limit:
        print(f"Limit: {args.limit}")

    evidence_gate = _build_evidence_gate(args)
    if evidence_gate:
        print(
            "Evidence override: "
            f"top={evidence_gate.min_top_score:.3f} "
            f"doc={evidence_gate.min_doc_score:.3f} "
            f"mean={evidence_gate.min_mean_score:.3f}"
        )

    from src.core.rag_chain import RAGChain

    chain = RAGChain(evidence_gate=evidence_gate) if evidence_gate else RAGChain()
    assert_runtime_matches_config(settings, chain)
    print(format_runtime_diagnostics(runtime_provider_info(settings, chain)))
    if args.show_runtime:
        return 0

    run = run_evaluation(chain, examples, run_name=args.run_name)
    files = write_outputs(run, output_dir=args.output_dir)

    print("\nEvaluation complete.")
    print(f"Raw JSON:     {files.raw_json}")
    print(f"CSV:          {files.csv}")
    print(f"Summary JSON: {files.summary_json}")
    print(f"Report:       {files.report_md}")
    return 0


def _load_split_ids(path: str, split_name: str) -> set[str]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return set(data[split_name])


def _build_evidence_gate(args: argparse.Namespace):
    if (
        args.evidence_min_top_score is None
        and args.evidence_min_doc_score is None
        and args.evidence_min_mean_score is None
        and getattr(args, "evidence_min_docs", None) is None
        and getattr(args, "evidence_score_source", None) is None
    ):
        return None
    from src.core.evidence_gate import EvidenceGate

    return EvidenceGate(
        min_top_score=args.evidence_min_top_score,
        min_doc_score=args.evidence_min_doc_score,
        min_docs=getattr(args, "evidence_min_docs", None),
        min_mean_score=args.evidence_min_mean_score,
        score_source=getattr(args, "evidence_score_source", None),
    )


if __name__ == "__main__":
    raise SystemExit(main())

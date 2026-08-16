"""
Self-contained sample eval — runs the full harness end-to-end over a tiny
fixture corpus without requiring the full 43,121-chunk index or any LLM.

PURPOSE
-------
This script exists solely to prove the eval harness runs start-to-finish and
produces real, reproducible numbers. It does NOT measure actual retrieval
quality on the 3GPP index.

LABEL: "Sample fixture (10 chunks, 6 queries)" — results are for harness
verification only.  Full-index numbers remain [RUN REQUIRED] and require the
built ChromaDB index plus a live embedding model.

HOW IT WORKS
------------
1. Loads 10 hand-built chunks from tests/fixtures/sample_corpus.py.
2. Builds an in-memory TF-IDF similarity index (no vector DB, no embeddings).
3. For each of 6 fixture queries, retrieves the top-k chunks by TF-IDF cosine.
4. Computes all metrics (hit-rate@k, Recall@k, MRR, nDCG@k binary, nDCG@k
   graded, heuristic context precision/recall) using scripts/eval/metrics.py.
5. Prints a results table and writes output JSON (default: data/eval/sample_results.json).
6. Optionally saves as data/eval/baseline.json (--save-baseline flag).

GUARDRAILS
----------
- No src/ imports. Does not depend on DocumentRetriever, document_processor,
  or ChromaDB. Stream D is renaming a file in document_processor; this script
  is immune to that change.
- Uses only stdlib + math. TF-IDF is re-implemented in ~30 lines for isolation.
- Every number in the output is genuinely produced by running this code.

Usage
-----
    # Run and print results (no file output)
    python scripts/eval/run_sample_eval.py

    # Write results JSON
    python scripts/eval/run_sample_eval.py --output data/eval/sample_results.json

    # Write results AND save as baseline for --regression comparisons
    python scripts/eval/run_sample_eval.py --output data/eval/sample_results.json --save-baseline

    # Quiet mode (summary only, no per-query rows)
    python scripts/eval/run_sample_eval.py --quiet
"""

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

# Make repo root importable (script may be run from repo root or scripts/eval/)
_REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from scripts.eval.metrics import (
    compute_retrieval_metrics,
    hit_rate_at_k,
    recall_at_k,
    mrr,
    ndcg_at_k,
    context_recall_keywords,
    context_precision,
)
from scripts.eval.regression import save_baseline


# ---------------------------------------------------------------------------
# Lightweight TF-IDF retriever (no external deps, no src/ imports)
# ---------------------------------------------------------------------------

def _tokenise(text: str) -> List[str]:
    """Lowercase, split on non-alphanumeric characters."""
    import re
    return re.findall(r'[a-z0-9]+', text.lower())


def _build_tfidf_index(chunks: List[Dict]) -> Tuple[List[Dict], Dict[str, List[float]]]:
    """Build a TF-IDF index over the chunk corpus.

    Returns:
        (chunks, idf_table)  where idf_table maps term -> IDF weight
    """
    n = len(chunks)
    df: Dict[str, int] = defaultdict(int)

    # Build term frequencies per chunk and document frequencies
    tf_lists: List[Dict[str, float]] = []
    for chunk in chunks:
        tokens = _tokenise(chunk["text"])
        tf: Dict[str, int] = defaultdict(int)
        for tok in tokens:
            tf[tok] += 1
        tf_norm = {t: cnt / len(tokens) for t, cnt in tf.items()} if tokens else {}
        tf_lists.append(tf_norm)
        for t in tf_norm:
            df[t] += 1

    idf = {t: math.log((n + 1) / (cnt + 1)) + 1 for t, cnt in df.items()}
    return tf_lists, idf


def _cosine_similarity(vec_a: Dict[str, float], vec_b: Dict[str, float]) -> float:
    """Cosine similarity between two sparse TF-IDF vectors."""
    common = set(vec_a) & set(vec_b)
    if not common:
        return 0.0
    dot = sum(vec_a[t] * vec_b[t] for t in common)
    norm_a = math.sqrt(sum(v * v for v in vec_a.values()))
    norm_b = math.sqrt(sum(v * v for v in vec_b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _tfidf_vector(tokens: List[str], idf: Dict[str, float]) -> Dict[str, float]:
    """Compute a TF-IDF vector for a list of tokens."""
    if not tokens:
        return {}
    tf: Dict[str, int] = defaultdict(int)
    for tok in tokens:
        tf[tok] += 1
    tf_norm = {t: cnt / len(tokens) for t, cnt in tf.items()}
    return {t: tf_norm[t] * idf.get(t, 1.0) for t in tf_norm}


def retrieve_tfidf(
    query: str,
    chunks: List[Dict],
    tf_lists: List[Dict[str, float]],
    idf: Dict[str, float],
    top_k: int = 5,
) -> List[Dict]:
    """Retrieve top_k chunks for a query using TF-IDF cosine similarity."""
    q_tokens = _tokenise(query)
    q_vec = _tfidf_vector(q_tokens, idf)

    scored = []
    for i, chunk in enumerate(chunks):
        sim = _cosine_similarity(q_vec, tf_lists[i])
        scored.append({**chunk, "similarity": round(sim, 6)})

    scored.sort(key=lambda d: d["similarity"], reverse=True)
    return scored[:top_k]


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_sample_eval(top_k: int = 5, quiet: bool = False) -> Dict:
    """Run the sample eval and return the structured results dict."""
    # Import fixture data
    sys.path.insert(0, str(_REPO_ROOT / "tests"))
    from fixtures.sample_corpus import SAMPLE_CHUNKS, SAMPLE_QUERIES

    # Build TF-IDF index over fixture chunks
    tf_lists, idf = _build_tfidf_index(SAMPLE_CHUNKS)

    per_case_results = []
    latencies = []

    for case in SAMPLE_QUERIES:
        t0 = time.perf_counter()
        docs = retrieve_tfidf(case["query"], SAMPLE_CHUNKS, tf_lists, idf, top_k=top_k)
        elapsed = time.perf_counter() - t0
        latencies.append(elapsed)

        # Binary metrics
        metrics = compute_retrieval_metrics(
            docs=docs,
            relevant_sources=case["relevant_sources"],
            expected_keywords=case["expected_keywords"],
            k=top_k,
        )

        # Graded nDCG (additional signal)
        graded_ndcg = ndcg_at_k(
            docs,
            case["relevant_sources"],
            k=top_k,
            graded_relevance=case.get("graded_relevance"),
        )

        per_case_results.append({
            "id": case["id"],
            "query": case["query"],
            "relevant_sources": case["relevant_sources"],
            "top_retrieved": [
                {"source": d["source"], "similarity": d["similarity"]}
                for d in docs
            ],
            "metrics": metrics,
            "ndcg_at_k_graded": graded_ndcg,
            "retrieve_latency_s": round(elapsed, 6),
        })

    # Aggregate summary
    def _avg(key: str) -> float:
        vals = [r["metrics"][key] for r in per_case_results]
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    latencies_sorted = sorted(latencies)
    n = len(latencies_sorted)
    p50 = latencies_sorted[n // 2]
    p95 = latencies_sorted[min(int(n * 0.95), n - 1)]

    graded_ndcg_scores = [r["ndcg_at_k_graded"] for r in per_case_results]
    avg_graded_ndcg = round(sum(graded_ndcg_scores) / len(graded_ndcg_scores), 4)

    summary = {
        "meta": {
            "fixture": "sample (10 chunks, 6 queries)",
            "label": "Sample fixture baseline — demonstrates harness runs start-to-finish; "
                     "NOT representative of full-corpus quality. "
                     "Full-index numbers remain [RUN REQUIRED].",
            "top_k": top_k,
            "num_queries": len(SAMPLE_QUERIES),
            "num_chunks": len(SAMPLE_CHUNKS),
        },
        "retrieval": {
            "avg_hit_rate_at_k":     _avg("hit_rate_at_k"),
            "avg_recall_at_k":       _avg("recall_at_k"),
            "avg_mrr":               _avg("mrr"),
            "avg_ndcg_at_k":         _avg("ndcg_at_k"),
            "avg_ndcg_at_k_graded":  avg_graded_ndcg,
            "avg_context_precision": _avg("context_precision"),
            "avg_context_recall":    _avg("context_recall_keywords"),
        },
        "latency": {
            "retrieve": {
                "p50":  round(p50, 6),
                "p95":  round(p95, 6),
                "mean": round(sum(latencies) / len(latencies), 6),
            }
        },
        "per_case": per_case_results,
    }

    if not quiet:
        _print_results(summary)

    return summary


def _print_results(summary: Dict) -> None:
    """Print a human-readable results table."""
    meta = summary["meta"]
    ret = summary["retrieval"]
    lat = summary["latency"]["retrieve"]

    print()
    print("=" * 70)
    print("SAMPLE FIXTURE EVAL — harness smoke test (NOT full-corpus quality)")
    print(f"  Fixture size: {meta['num_chunks']} chunks, {meta['num_queries']} queries, k={meta['top_k']}")
    print("=" * 70)
    print()
    print(f"  {'Metric':<35}  {'Value':>8}")
    print(f"  {'-'*35}  {'-'*8}")
    print(f"  {'Hit-rate@k':<35}  {ret['avg_hit_rate_at_k']:>8.4f}")
    print(f"  {'Recall@k':<35}  {ret['avg_recall_at_k']:>8.4f}")
    print(f"  {'MRR':<35}  {ret['avg_mrr']:>8.4f}")
    print(f"  {'nDCG@k (binary)':<35}  {ret['avg_ndcg_at_k']:>8.4f}")
    print(f"  {'nDCG@k (graded 0/1/2)':<35}  {ret['avg_ndcg_at_k_graded']:>8.4f}")
    print(f"  {'Context precision (heuristic)':<35}  {ret['avg_context_precision']:>8.4f}")
    print(f"  {'Context recall (keyword, heuristic)':<35}  {ret['avg_context_recall']:>8.4f}")
    print(f"  {'Retrieve latency p50 (s)':<35}  {lat['p50']:>8.6f}")
    print(f"  {'Retrieve latency p95 (s)':<35}  {lat['p95']:>8.6f}")
    print()
    print("  Per-query breakdown:")
    print(f"  {'ID':<10} {'HR@k':>6} {'Rec@k':>6} {'MRR':>6} {'nDCG':>6} {'nDCG-G':>7}  Top source")
    print(f"  {'-'*10} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*7}  {'-'*30}")
    for r in summary["per_case"]:
        m = r["metrics"]
        top_src = r["top_retrieved"][0]["source"] if r["top_retrieved"] else "—"
        print(
            f"  {r['id']:<10} {m['hit_rate_at_k']:>6.3f} {m['recall_at_k']:>6.3f} "
            f"{m['mrr']:>6.3f} {m['ndcg_at_k']:>6.4f} {r['ndcg_at_k_graded']:>7.4f}  {top_src}"
        )
    print()
    print("  [NOTE] These numbers are from a 10-chunk fixture, not the full 43,121-chunk")
    print("         index. They confirm the harness runs correctly. Full-index numbers")
    print("         require the built ChromaDB index: see EVAL_REPORT.md §5.")
    print()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run sample fixture eval to verify the harness end-to-end."
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Write results JSON to this path (e.g. data/eval/sample_results.json)",
    )
    parser.add_argument(
        "--save-baseline",
        action="store_true",
        help="Also save results as data/eval/baseline.json for --regression comparisons.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Top-k window for @k metrics (default: 5).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-query table; print summary only.",
    )
    args = parser.parse_args()

    summary = run_sample_eval(top_k=args.k, quiet=args.quiet)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Results written to {out_path}")

    if args.save_baseline:
        baseline_path = str(_REPO_ROOT / "data" / "eval" / "baseline.json")
        save_baseline(summary, baseline_path=baseline_path)
        print(f"Baseline saved to {baseline_path}")
        print("NOTE: This baseline is from the 10-chunk sample fixture.")
        print("      Replace with a full-index run when the index is available.")


if __name__ == "__main__":
    main()

"""
Evaluation harness for the 3GPP RAG pipeline.

Measures three layers of quality:

  1. Retrieval quality — are the right chunks being fetched?

     STANDARD IR METRICS (primary — suitable for enterprise-RAG audiences):
       - Recall@k    : fraction of expected source specs found in top-k
       - MRR         : mean reciprocal rank of first relevant doc
       - nDCG@k      : normalised discounted cumulative gain
       - Hit-rate@k  : binary — was any relevant doc in top-k?

     HEURISTIC METRICS (secondary — carried forward from original harness):
       - Context precision  : fraction of retrieved chunks from a relevant source
       - Context recall     : fraction of expected keywords in top-3 chunks
       Note: these are RAGAS-inspired keyword heuristics, NOT IR-standard
       precision/recall.  They measure surface-level vocabulary coverage only.

     Latency: p50/p95 retrieve time

  2. Answer quality (requires Ollama or Groq) [RUN REQUIRED if not executed]

     HEURISTIC answer metrics (no LLM judge):
       - Answer relevance  : expected answer-keywords in the response
       - Faithfulness      : heuristic grounding check (NOT LLM-judge)
       - Answer length / refusal detection

     LLM-JUDGE path (faithfulness + answer-correctness, configurable):
       - Uses Groq (default) or Ollama as judge model
       - Requires --judge-provider and GROQ_API_KEY or running Ollama
       - Results marked [RUN REQUIRED] if not executed this session

  3. Regression mode (--regression)
       - Compares current run to a stored baseline
       - Exits non-zero if any tracked metric regresses beyond tolerance
       - Gate CI on: python scripts/eval_retrieval.py --regression

Usage
-----
    # Retrieval only (fast, no LLM — requires built vector index)
    python scripts/eval_retrieval.py

    # Use expanded golden set (default; uses data/eval/golden_set.jsonl)
    python scripts/eval_retrieval.py --golden data/eval/golden_set.jsonl

    # Use legacy 10-query inline test cases
    python scripts/eval_retrieval.py --no-golden

    # Full eval including heuristic answer quality (requires Ollama)
    python scripts/eval_retrieval.py --full

    # Full eval with LLM judge (requires GROQ_API_KEY or running Ollama)
    python scripts/eval_retrieval.py --full --judge --judge-provider groq

    # Save results
    python scripts/eval_retrieval.py --output data/eval_results.json

    # Regression check against stored baseline (CI gate)
    python scripts/eval_retrieval.py --regression

    # Save current results as new baseline
    python scripts/eval_retrieval.py --save-baseline

    # One-shot reproduce command documented in EVAL_REPORT.md:
    python scripts/eval_retrieval.py --output data/eval_results.json
"""
import sys
import json
import argparse
import time
import statistics
from pathlib import Path
from typing import List, Dict, Any, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.retriever import DocumentRetriever
from scripts.eval.metrics import compute_retrieval_metrics
from scripts.eval.regression import compare_to_baseline, save_baseline, print_comparison_table

# ---------------------------------------------------------------------------
# Backward-compatibility aliases (imported by tests/test_eval.py)
# ---------------------------------------------------------------------------

from scripts.eval.metrics import (
    context_precision as _context_precision,
    context_recall_keywords as _context_recall,
)
from scripts.eval.metrics import is_refusal as _is_answer_refusal
from scripts.eval.metrics import pass_sim_threshold, REFUSAL_SIM_THRESHOLD
from src.config import settings


# ---------------------------------------------------------------------------
# Inline test cases (legacy — 10 queries)
# These match the cases used in the 2026-02-27 baseline run in
# data/eval_results.json and are kept for backward-compat / fast smoke tests.
# For the full evaluation use --golden data/eval/golden_set.jsonl.
# ---------------------------------------------------------------------------

LEGACY_TEST_CASES: List[Dict[str, Any]] = [
    {
        "id": "legacy-01",
        "query": "What is gNB?",
        "expected_keywords": ["gnb", "base station", "node", "ran"],
        "answer_keywords": ["gnb", "base station", "radio"],
        "relevant_sources": ["38300", "38401"],
        "min_similarity": 0.50,
    },
    {
        "id": "legacy-02",
        "query": "Explain the 5G protocol stack",
        "expected_keywords": ["protocol", "stack", "layer", "nr"],
        "answer_keywords": ["protocol", "layer", "stack"],
        "relevant_sources": ["38300"],
        "min_similarity": 0.50,
    },
    {
        "id": "legacy-03",
        "query": "What are the NG-RAN functions?",
        "expected_keywords": ["ng-ran", "function", "ran"],
        "answer_keywords": ["ng-ran", "function"],
        "relevant_sources": ["38300", "38401"],
        "min_similarity": 0.50,
    },
    {
        "id": "legacy-04",
        "query": "How does handover work in 5G?",
        "expected_keywords": ["handover", "mobility", "cell"],
        "answer_keywords": ["handover", "cell", "ue"],
        "relevant_sources": ["38300"],
        "min_similarity": 0.45,
    },
    {
        "id": "legacy-05",
        "query": "What is the NR physical layer?",
        "expected_keywords": ["physical", "layer", "nr"],
        "answer_keywords": ["physical", "layer", "channel"],
        "relevant_sources": ["38300"],
        "min_similarity": 0.50,
    },
    {
        "id": "legacy-06",
        "query": "What is the difference between SA and NSA 5G deployments?",
        "expected_keywords": ["standalone", "non-standalone", "sa", "nsa"],
        "answer_keywords": ["sa", "nsa", "standalone"],
        "relevant_sources": ["38300"],
        "min_similarity": 0.45,
    },
    {
        "id": "legacy-07",
        "query": "Describe the F1 interface between gNB-CU and gNB-DU",
        "expected_keywords": ["f1", "cu", "du", "interface"],
        "answer_keywords": ["f1", "cu", "du"],
        "relevant_sources": ["38401"],
        "min_similarity": 0.45,
    },
    {
        "id": "legacy-08",
        "query": "What is E1 interface used for in 5G?",
        "expected_keywords": ["e1", "interface", "cu-cp", "cu-up"],
        "answer_keywords": ["e1", "cu-cp", "cu-up"],
        "relevant_sources": ["38401"],
        "min_similarity": 0.45,
    },
    {
        "id": "legacy-09",
        "query": "How is QoS managed in NG-RAN?",
        "expected_keywords": ["qos", "quality", "service", "bearer"],
        "answer_keywords": ["qos", "service", "flow"],
        "relevant_sources": ["38300"],
        "min_similarity": 0.45,
    },
    {
        "id": "legacy-10",
        "query": "What is Xn interface?",
        "expected_keywords": ["xn", "interface", "gnb"],
        "answer_keywords": ["xn", "interface", "gnb"],
        "relevant_sources": ["38300", "38401"],
        "min_similarity": 0.45,
    },
]

# Backward-compat alias: test_eval.py imports TEST_CASES from this module.
TEST_CASES = LEGACY_TEST_CASES


# ---------------------------------------------------------------------------
# Golden-set loader
# ---------------------------------------------------------------------------

def load_golden_set(path: str) -> List[Dict[str, Any]]:
    """Load the versioned golden dataset from a JSONL file.

    Each line is one test case with at minimum:
        id, query, relevant_sources, expected_keywords

    Raises FileNotFoundError if the path doesn't exist.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Golden set not found: {path}")
    cases = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


# ---------------------------------------------------------------------------
# Retrieval evaluation (single case)
# ---------------------------------------------------------------------------

def evaluate_retrieval_case(retriever: DocumentRetriever, case: Dict) -> Dict:
    """Run retrieval evaluation for a single test case.

    Returns a result dict that includes both standard IR metrics and the
    legacy heuristic metrics, plus latency and pass/fail.
    """
    query = case["query"]
    relevant_sources = case.get("relevant_sources", [])
    expected_keywords = case.get("expected_keywords", [])
    graded_relevance = case.get("graded_relevance")  # {spec: 0/1/2} where labeled
    # Pass threshold: a per-case override wins (the legacy inline cases pin
    # 0.50 for baseline continuity); otherwise use the per-embedding-model
    # calibrated threshold (see scripts/eval/metrics.py PASS_SIM_THRESHOLDS
    # and scripts/eval/calibrate_threshold.py for the derivation).
    min_sim = case.get("min_similarity")
    if min_sim is None:
        min_sim = pass_sim_threshold(settings.embedding_model)
    k = case.get("k", 5)

    t0 = time.time()
    docs = retriever.retrieve(query)
    elapsed = round(time.time() - t0, 4)

    is_ooc = case.get("is_out_of_corpus", False)

    if not docs:
        return {
            "id":               case.get("id", query[:30]),
            "query":            query,
            "is_out_of_corpus": is_ooc,
            # For an out-of-corpus query, returning nothing is the correct
            # (refusing) behaviour; for an in-corpus query it is a miss.
            "refusal_correct":  True if is_ooc else None,
            "retrieval_pass":   False,
            # Standard IR
            "hit_rate_at_k":    0.0,
            "recall_at_k":      0.0,
            "mrr":              0.0,
            "ndcg_at_k":        0.0,
            # Heuristic
            "context_precision":       0.0,
            "context_recall":          0.0,
            # Legacy compat
            "context_recall_keywords": 0.0,
            "avg_similarity":   0.0,
            "top_source":       "N/A",
            "num_results":      0,
            "retrieve_time":    elapsed,
        }

    metrics = compute_retrieval_metrics(
        docs=docs,
        relevant_sources=relevant_sources,
        expected_keywords=expected_keywords,
        k=k,
        graded_relevance=graded_relevance,  # graded nDCG where labeled; binary fallback
    )

    avg_sim = sum(d["similarity"] for d in docs[:3]) / min(3, len(docs))

    # Legacy pass criterion (keyword recall >= 50% AND avg similarity >= threshold)
    passed = metrics["context_recall_keywords"] >= 0.5 and avg_sim >= min_sim

    # For an out-of-corpus query the correct behaviour is to NOT retrieve a
    # confident match: refusal is "correct" when top-3 avg similarity stays
    # below the refusal boundary. This is deliberately a separate dial from
    # the calibrated pass threshold (see metrics.REFUSAL_SIM_THRESHOLD).
    refusal_correct = (avg_sim < REFUSAL_SIM_THRESHOLD) if is_ooc else None

    return {
        "id":             case.get("id", query[:30]),
        "query":          query,
        "is_out_of_corpus": is_ooc,
        "refusal_correct":  refusal_correct,
        "retrieval_pass": passed,
        # Standard IR metrics
        "hit_rate_at_k":  metrics["hit_rate_at_k"],
        "recall_at_k":    metrics["recall_at_k"],
        "mrr":            metrics["mrr"],
        "ndcg_at_k":      metrics["ndcg_at_k"],
        # Heuristic metrics (RAGAS-inspired; NOT IR-standard precision/recall)
        "context_precision":       metrics["context_precision"],
        "context_recall":          metrics["context_recall_keywords"],    # alias for API compat
        "context_recall_keywords": metrics["context_recall_keywords"],    # explicit
        # Other
        "avg_similarity": round(avg_sim, 3),
        "min_similarity": min_sim,
        "top_source":     docs[0]["source"],
        "num_results":    len(docs),
        "retrieve_time":  elapsed,
        "docs":           docs,   # stripped before saving
    }


def _print_retrieval_result(i: int, result: Dict) -> None:
    status = "PASS" if result["retrieval_pass"] else "FAIL"
    icon = "✅" if result["retrieval_pass"] else "❌"
    print(f"\nTest {i} [{result.get('id', '')}]: {result['query']}")
    print("-" * 60)
    print(f"  Hit-rate@k:         {result['hit_rate_at_k']:.1%}")
    print(f"  Recall@k:           {result['recall_at_k']:.1%}")
    print(f"  MRR:                {result['mrr']:.3f}")
    print(f"  nDCG@k:             {result['ndcg_at_k']:.3f}")
    print(f"  Context precision:  {result['context_precision']:.1%}  [heuristic]")
    print(f"  Context recall kw:  {result['context_recall']:.1%}       [heuristic]")
    print(f"  Avg similarity:     {result['avg_similarity']:.3f}  "
          f"(min={result.get('min_similarity', 0.50):.3f})")
    print(f"  Top source:         {result['top_source']}")
    print(f"  Retrieve time:      {result['retrieve_time']}s")
    print(f"  Status:             {icon} {status}")


# ---------------------------------------------------------------------------
# Answer quality evaluation (heuristic — no LLM judge)
# ---------------------------------------------------------------------------

def _answer_relevance(answer: str, case: Dict) -> float:
    """HEURISTIC — fraction of expected answer keywords present in the response.

    Falls back to ``expected_keywords`` when a case has no ``answer_keywords``
    field: the golden set (data/eval/golden_set.jsonl) defines only
    ``expected_keywords``, and without the fallback every golden-set case
    scored 0.0 (the 2026-07-02 run's flat avg_answer_relevance).
    """
    kw = case.get("answer_keywords") or case.get("expected_keywords") or []
    if not kw:
        return 0.0
    hits = sum(1 for k in kw if k.lower() in answer.lower())
    return hits / len(kw)


def _faithfulness(answer: str, docs: List[Dict]) -> float:
    """HEURISTIC — estimate of grounding in retrieved context.

    Uses two proxy signals:
      1. Presence of grounding vocabulary (e.g. "3gpp", "according to")
      2. Content overlap between answer and top retrieved chunk

    Limitation: this is a surface heuristic, NOT a true faithfulness score.
    For production use, replace with the LLM-judge path in scripts/eval/judge.py.
    """
    if not docs:
        return 0.0

    answer_lower = answer.lower()
    grounding_signals = [
        "3gpp", "ts ", "specification", "according", "document",
        "source:", "based on", "the spec", "release",
    ]
    grounding_score = min(
        sum(1 for s in grounding_signals if s in answer_lower) / 3, 1.0
    )

    stop = {
        "the", "a", "an", "is", "are", "in", "of", "to", "and", "or",
        "that", "it", "this", "for", "be", "was", "with", "as", "by",
    }
    top_words = set(docs[0]["text"].lower().split()) - stop
    answer_words = set(answer_lower.split()) - stop
    if top_words:
        overlap = len(top_words & answer_words) / min(len(top_words), 20)
        overlap_score = min(overlap, 1.0)
    else:
        overlap_score = 0.0

    return round((grounding_score * 0.5) + (overlap_score * 0.5), 3)


def evaluate_answer_quality(answer: str, case: Dict, docs: List[Dict]) -> Dict:
    """Score the quality of a generated answer against a test case (heuristic)."""
    relevance = _answer_relevance(answer, case)
    faithfulness_score = _faithfulness(answer, docs)

    is_substantive = len(answer.strip()) >= 50
    # Single source of truth for refusal detection: scripts/eval/metrics.py.
    # (A narrower inline phrase list here previously missed all five real
    # refusals in the 2026-07-02 run.)
    refused = _is_answer_refusal(answer)

    score = round(
        (relevance * 0.4)
        + (faithfulness_score * 0.4)
        + (0.2 if is_substantive and not refused else 0.0),
        3,
    )

    return {
        "answer_relevance":   round(relevance, 3),
        "faithfulness":       faithfulness_score,
        "answer_score":       score,
        "is_substantive":     is_substantive,
        "is_refusal":         refused,
        "answer_pass":        score >= 0.40,
        "answer_preview":     answer[:200].replace("\n", " "),
    }


def run_answer_eval(
    retriever: DocumentRetriever, retrieval_results: List[Dict], cases: List[Dict]
) -> List[Dict]:
    """Enhance retrieval results with heuristic answer quality scores."""
    try:
        from src.core.llm import OllamaLLM
        from src.core.rag_chain import RAGChain
    except ImportError as e:
        print(f"\n  Could not import LLM modules: {e}")
        return retrieval_results

    llm = OllamaLLM()
    if not llm.is_available():
        print(f"\n  Ollama not available — skipping answer quality evaluation")
        print(f"  Start Ollama: ollama serve  |  Pull model: ollama pull {llm.model}")
        return retrieval_results

    chain = RAGChain(retriever=retriever, llm=llm)
    print(f"\n{'='*60}")
    print(f"Answer Quality Evaluation (heuristic)  (model: {llm.model})")
    print("=" * 60)

    enriched = []
    for i, (result, case) in enumerate(zip(retrieval_results, cases[:len(retrieval_results)]), 1):
        print(f"\nTest {i}: {result['query']}")
        print("-" * 60)

        t0 = time.time()
        rag_result = chain.query(result["query"])
        gen_time = round(time.time() - t0, 4)

        docs = result.get("docs", [])
        aq = evaluate_answer_quality(rag_result["answer"], case, docs)
        aq["generate_time"] = gen_time

        icon = "✅" if aq["answer_pass"] else "❌"
        print(f"  Answer relevance:  {aq['answer_relevance']:.1%}")
        print(f"  Faithfulness:      {aq['faithfulness']:.1%}  [heuristic]")
        print(f"  Score:             {aq['answer_score']:.2f}/1.00")
        print(f"  Generate time:     {gen_time}s")
        print(f"  Status:            {icon} {'PASS' if aq['answer_pass'] else 'FAIL'}")
        print(f"  Preview:           {aq['answer_preview'][:120]}...")

        enriched.append({**result, **aq})

    return enriched


def run_llm_judge_eval(
    retrieval_results: List[Dict],
    cases: List[Dict],
    provider: str = "groq",
    model: Optional[str] = None,
) -> List[Dict]:
    """[RUN REQUIRED] Enrich retrieval results with LLM-judge faithfulness + correctness.

    Requires a live LLM (Groq API key or running Ollama).  Results are
    marked [RUN REQUIRED] in EVAL_REPORT.md when this has not been executed.

    Args:
        retrieval_results: Output of the retrieval eval phase (must include 'docs').
        cases:             Corresponding golden-set cases.
        provider:          "groq" | "ollama"
        model:             Override judge model name.

    Returns:
        Enriched results with "llm_judge" key per case.
    """
    from scripts.eval.judge import LLMJudge

    judge = LLMJudge(provider=provider, model=model)
    if not judge.is_available():
        print(f"\n  LLM judge ({provider}) not available — skipping judge evaluation")
        if provider == "groq":
            print("  Set GROQ_API_KEY environment variable to enable.")
        else:
            print("  Start Ollama: ollama serve")
        return retrieval_results

    print(f"\n{'='*60}")
    print(f"LLM-Judge Evaluation  (provider: {provider}, model: {judge.model})")
    print("=" * 60)

    enriched = []
    for i, (result, case) in enumerate(zip(retrieval_results, cases[:len(retrieval_results)]), 1):
        print(f"\n  Judging {i}/{len(retrieval_results)}: {result['query'][:60]}")

        # Reconstruct context from docs
        docs = result.get("docs", [])
        context = "\n".join(
            f"[{j+1}] (Source: {d['source']}) {d['text'][:400]}"
            for j, d in enumerate(docs[:5])
        )
        # We need an answer to judge — skip if answer not available
        answer = result.get("answer_preview", "")
        if not answer:
            enriched.append(result)
            continue

        scores = judge.judge(
            question=result["query"],
            context=context,
            answer=answer,
        )
        icon = "✅" if scores.get("parse_ok") else "❌"
        if scores.get("parse_ok"):
            print(f"    Faithfulness:       {scores['faithfulness']:.2f}  "
                  f"(raw {scores['faithfulness_raw']}/5)")
            print(f"    Answer correctness: {scores['answer_correctness']:.2f}  "
                  f"(raw {scores['answer_correctness_raw']}/5)")
        else:
            print(f"    {icon} Parse error: {scores.get('notes', '')}")

        enriched.append({**result, "llm_judge": scores})

    return enriched


# ---------------------------------------------------------------------------
# Latency benchmarks
# ---------------------------------------------------------------------------

def _percentile(values: List[float], pct: int) -> float:
    """Return the given percentile value from a list of floats."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = max(0, int(len(sorted_vals) * pct / 100) - 1)
    return round(sorted_vals[idx], 4)


def compute_latency_benchmarks(results: List[Dict]) -> Dict:
    """Compute p50/p95 latency stats for retrieve and generate times."""
    retrieve_times = [r["retrieve_time"] for r in results if "retrieve_time" in r]
    generate_times = [r["generate_time"] for r in results if r.get("generate_time")]

    benchmarks: Dict[str, Any] = {
        "retrieve": {
            "p50":  _percentile(retrieve_times, 50),
            "p95":  _percentile(retrieve_times, 95),
            "mean": round(statistics.mean(retrieve_times), 4) if retrieve_times else 0,
        }
    }
    if generate_times:
        total_times = [
            r.get("retrieve_time", 0) + r.get("generate_time", 0)
            for r in results
            if r.get("generate_time")
        ]
        benchmarks["generate"] = {
            "p50":  _percentile(generate_times, 50),
            "p95":  _percentile(generate_times, 95),
            "mean": round(statistics.mean(generate_times), 4),
        }
        benchmarks["total"] = {
            "p50":  _percentile(total_times, 50),
            "p95":  _percentile(total_times, 95),
            "mean": round(statistics.mean(total_times), 4),
        }

    return benchmarks


# ---------------------------------------------------------------------------
# Summary & reporting
# ---------------------------------------------------------------------------

def print_summary(results: List[Dict], full_eval: bool) -> Dict:
    """Print evaluation summary and return a structured summary dict."""
    print("\n" + "=" * 60)
    print("Evaluation Summary")
    print("=" * 60)

    # Split in-corpus (answerable) queries from out-of-corpus (refusal) probes.
    # IR metrics only make sense on answerable queries; mixing the deliberate
    # negatives into the averages would understate retrieval quality and
    # mis-score the negatives (a negative "passes" by retrieving nothing
    # confident, not by matching a relevant source).
    in_corpus = [r for r in results if not r.get("is_out_of_corpus")]
    ooc       = [r for r in results if r.get("is_out_of_corpus")]

    n_in = len(in_corpus)
    r_passed = sum(1 for r in in_corpus if r["retrieval_pass"])
    r_rate = r_passed / n_in if n_in else 0

    # Standard IR aggregates — in-corpus only
    avg_hit_rate   = sum(r.get("hit_rate_at_k", 0) for r in in_corpus) / n_in if n_in else 0
    avg_recall     = sum(r.get("recall_at_k", 0) for r in in_corpus) / n_in if n_in else 0
    avg_mrr        = sum(r.get("mrr", 0) for r in in_corpus) / n_in if n_in else 0
    avg_ndcg       = sum(r.get("ndcg_at_k", 0) for r in in_corpus) / n_in if n_in else 0
    # Heuristic aggregates — in-corpus only
    avg_precision  = sum(r.get("context_precision", 0) for r in in_corpus) / n_in if n_in else 0
    avg_recall_kw  = sum(r.get("context_recall", 0) for r in in_corpus) / n_in if n_in else 0
    avg_sim        = sum(r.get("avg_similarity", 0) for r in in_corpus) / n_in if n_in else 0

    print(f"\nRetrieval Quality — Standard IR Metrics (in-corpus, N={n_in})")
    print(f"  Hit-rate@k:            {avg_hit_rate:.1%}")
    print(f"  Recall@k:              {avg_recall:.1%}")
    print(f"  MRR:                   {avg_mrr:.3f}")
    print(f"  nDCG@k:                {avg_ndcg:.3f}")

    print(f"\nRetrieval Quality — Heuristic Metrics [RAGAS-inspired; supplementary]")
    print(f"  Legacy pass rate:      {r_passed}/{n_in} ({r_rate:.1%})")
    print(f"  Avg context precision: {avg_precision:.1%}")
    print(f"  Avg context recall kw: {avg_recall_kw:.1%}")
    print(f"  Avg similarity:        {avg_sim:.3f}")

    if avg_ndcg >= 0.80:
        print("  → EXCELLENT retrieval quality (nDCG@k)")
    elif avg_ndcg >= 0.60:
        print("  → GOOD retrieval quality (nDCG@k)")
    elif avg_ndcg >= 0.40:
        print("  → FAIR — consider tuning chunk size or embedding model")
    else:
        print("  → POOR — check vector store and embedding model")

    # Refusal axis — out-of-corpus probes. Two layers:
    #   Retrieval layer: correct behaviour is to NOT surface a confident match
    #   (top-3 avg similarity below the relevance threshold).
    #   Answer layer: the generated answer should decline — scored by the
    #   heuristic detector in scripts/eval/metrics.py when --full ran.
    # A judge-based refusal check remains [RUN REQUIRED] (not implemented).
    refusal_summary: Dict = {}
    if ooc:
        n_ooc = len(ooc)
        refused_ok = sum(1 for r in ooc if r.get("refusal_correct"))
        ooc_rate = refused_ok / n_ooc
        ooc_sim = sum(r.get("avg_similarity", 0) for r in ooc) / n_ooc
        print(f"\nRefusal Axis — Out-of-Corpus Probes (N={n_ooc})")
        print(f"  Retrieval-refusal rate: {refused_ok}/{n_ooc} ({ooc_rate:.1%})  "
              f"[top-3 sim below threshold]")
        print(f"  Avg OOC similarity:     {ooc_sim:.3f}  (lower is better)")

        answered = [r for r in ooc if "is_refusal" in r]
        if answered:
            ans_refused = sum(1 for r in answered if r.get("is_refusal"))
            ans_rate = ans_refused / len(answered)
            print(f"  Answer-refusal rate:    {ans_refused}/{len(answered)} ({ans_rate:.1%})  "
                  f"[heuristic detector on generated answers]")
            answer_refusal_rate: Any = round(ans_rate, 3)
        else:
            print(f"  Answer-refusal rate:    [RUN REQUIRED]  (needs --full)")
            answer_refusal_rate = "[RUN REQUIRED]"
        print(f"  LLM-judged refusal rate: [RUN REQUIRED]  (judge-based refusal check not implemented)")

        refusal_summary = {
            "out_of_corpus_probes":    n_ooc,
            "retrieval_refusal_rate":  round(ooc_rate, 3),
            "retrieval_refused_ok":    refused_ok,
            "avg_ooc_similarity":      round(ooc_sim, 3),
            "answer_refusal_rate":     answer_refusal_rate,
            "llm_judged_refusal_rate": "[RUN REQUIRED]",
        }

    answer_summary: Dict = {}
    if full_eval and any("answer_score" in r for r in in_corpus) and n_in:
        a_passed = sum(1 for r in in_corpus if r.get("answer_pass", False))
        avg_ans_relevance  = sum(r.get("answer_relevance", 0) for r in in_corpus) / n_in
        avg_faithfulness   = sum(r.get("faithfulness", 0) for r in in_corpus) / n_in
        avg_score          = sum(r.get("answer_score", 0) for r in in_corpus) / n_in

        print(f"\nAnswer Quality [heuristic — NOT LLM-judge] (in-corpus, N={n_in})")
        print(f"  Pass rate:            {a_passed}/{n_in} ({a_passed/n_in:.1%})")
        print(f"  Avg answer relevance: {avg_ans_relevance:.1%}")
        print(f"  Avg faithfulness:     {avg_faithfulness:.1%}  [heuristic]")
        print(f"  Avg score:            {avg_score:.2f}/1.00")

        answer_summary = {
            "pass_rate":            round(a_passed / n_in, 3),
            "avg_answer_relevance": round(avg_ans_relevance, 3),
            "avg_faithfulness":     round(avg_faithfulness, 3),
            "avg_score":            round(avg_score, 3),
        }

    # LLM-judge aggregate (only if scores present)
    judge_scores = [r["llm_judge"] for r in results if r.get("llm_judge", {}).get("parse_ok")]
    judge_summary: Dict = {}
    if judge_scores:
        avg_judge_faith = sum(s["faithfulness"] for s in judge_scores) / len(judge_scores)
        avg_judge_corr  = sum(s["answer_correctness"] for s in judge_scores) / len(judge_scores)
        print(f"\nAnswer Quality [LLM-judge: {judge_scores[0].get('judge_model', '?')}]")
        print(f"  Avg faithfulness:     {avg_judge_faith:.3f}")
        print(f"  Avg answer correct:   {avg_judge_corr:.3f}")
        judge_summary = {
            "judged_cases":        len(judge_scores),
            "avg_faithfulness":    round(avg_judge_faith, 3),
            "avg_answer_correct":  round(avg_judge_corr, 3),
            "judge_model":         judge_scores[0].get("judge_model", "unknown"),
        }

    benchmarks = compute_latency_benchmarks(results)
    print(f"\nLatency Benchmarks")
    print(f"  Retrieve p50: {benchmarks['retrieve']['p50']}s  "
          f"p95: {benchmarks['retrieve']['p95']}s")
    if "generate" in benchmarks:
        print(f"  Generate p50: {benchmarks['generate']['p50']}s  "
              f"p95: {benchmarks['generate']['p95']}s")
        print(f"  Total    p50: {benchmarks['total']['p50']}s  "
              f"p95: {benchmarks['total']['p95']}s")

    print("=" * 60 + "\n")

    return {
        "retrieval": {
            # Standard IR — in-corpus (answerable) queries only
            "scope":             "in_corpus",
            "avg_hit_rate_at_k": round(avg_hit_rate, 3),
            "avg_recall_at_k":   round(avg_recall, 3),
            "avg_mrr":           round(avg_mrr, 3),
            "avg_ndcg_at_k":     round(avg_ndcg, 3),
            # Heuristic (legacy compat)
            "pass_rate":         round(r_rate, 3),
            "passed":            r_passed,
            "total":             n_in,
            "avg_context_precision": round(avg_precision, 3),
            "avg_context_recall":    round(avg_recall_kw, 3),
            "avg_similarity":        round(avg_sim, 3),
        },
        "refusal":    refusal_summary,
        "answer":     answer_summary,
        "llm_judge":  judge_summary,
        "latency":    benchmarks,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate 3GPP RAG pipeline quality",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Run heuristic answer quality evaluation (requires Ollama)",
    )
    parser.add_argument(
        "--judge", action="store_true",
        help="Run LLM-judge faithfulness + answer-correctness [RUN REQUIRED]",
    )
    parser.add_argument(
        "--judge-provider", default="groq", choices=["groq", "ollama"],
        help="LLM provider for the judge (default: groq)",
    )
    parser.add_argument(
        "--judge-model", default=None,
        help="Override judge model name",
    )
    parser.add_argument(
        "--golden", default="data/eval/golden_set.jsonl",
        help="Path to golden JSONL dataset (default: data/eval/golden_set.jsonl)",
    )
    parser.add_argument(
        "--no-golden", action="store_true",
        help="Use legacy 10-query inline test cases instead of golden set",
    )
    parser.add_argument(
        "--output", default=None,
        help="Save results to this JSON file (default: data/eval_results.json; "
             "in --regression mode results are NOT saved unless --output is "
             "given explicitly, so the gate never overwrites the artifact)",
    )
    parser.add_argument(
        "--top-k", type=int, default=5,
        help="Chunks to retrieve per query (default: 5)",
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="Skip saving results to disk",
    )
    parser.add_argument(
        "--regression", action="store_true",
        help="Compare to stored baseline; exit non-zero on regression (CI gate)",
    )
    parser.add_argument(
        "--save-baseline", action="store_true",
        help="Save this run's summary as the new regression baseline",
    )
    parser.add_argument(
        "--baseline", default="data/eval/baseline.json",
        help="Path to regression baseline file (default: data/eval/baseline.json)",
    )
    args = parser.parse_args()

    from src.core.vector_store import VectorStore
    vs_stats = VectorStore().get_stats()
    if vs_stats["total_chunks"] == 0:
        print("\n  Vector store is empty. Run: python scripts/build_index.py\n")
        sys.exit(1)

    # Decide which test cases to use
    if args.no_golden:
        test_cases = LEGACY_TEST_CASES
        dataset_label = "legacy 10-query inline cases"
    else:
        try:
            test_cases = load_golden_set(args.golden)
            dataset_label = f"golden set ({args.golden}, {len(test_cases)} cases)"
        except FileNotFoundError:
            print(f"\n  Golden set not found at {args.golden}. "
                  f"Falling back to legacy test cases. "
                  f"Run with --no-golden to suppress this warning.\n")
            test_cases = LEGACY_TEST_CASES
            dataset_label = "legacy 10-query inline cases (fallback)"

    print(f"\n{'='*60}")
    print("3GPP RAG Pipeline Evaluation")
    print(f"{'='*60}")
    print(f"Vector store:  {vs_stats['total_chunks']} chunks")
    print(f"Dataset:       {dataset_label}")
    print(f"Top-k:         {args.top_k}")
    mode_parts = ["Retrieval"]
    if args.full:
        mode_parts.append("heuristic-answer")
    if args.judge:
        mode_parts.append(f"LLM-judge ({args.judge_provider})")
    print(f"Mode:          {' + '.join(mode_parts)}")

    retriever = DocumentRetriever(top_k=args.top_k)

    # Retrieval evaluation
    print(f"\n{'='*60}")
    print("Retrieval Quality Evaluation")
    print("=" * 60)

    retrieval_results = []
    for i, case in enumerate(test_cases, 1):
        result = evaluate_retrieval_case(retriever, case)
        _print_retrieval_result(i, result)
        retrieval_results.append(result)

    # Heuristic answer quality (optional)
    final_results = retrieval_results
    if args.full:
        final_results = run_answer_eval(retriever, retrieval_results, test_cases)

    # LLM-judge path (optional, [RUN REQUIRED] without live LLM)
    if args.judge:
        final_results = run_llm_judge_eval(
            final_results, test_cases,
            provider=args.judge_provider,
            model=args.judge_model,
        )

    summary = print_summary(final_results, full_eval=args.full)

    # Save results. Regression mode is a read-only gate: it must not
    # overwrite the committed results artifact as a side effect (that
    # clobbered a judge run on 2026-07-02). Saving in regression mode
    # requires an explicit --output.
    save_results = not args.no_save and (args.output is not None or not args.regression)
    if save_results:
        import datetime
        save_data = {
            "evaluated_at":  datetime.datetime.utcnow().isoformat() + "Z",
            "config": {
                "top_k":         args.top_k,
                "full_eval":     args.full,
                "judge_eval":    args.judge,
                "judge_provider": args.judge_provider if args.judge else None,
                "total_chunks":  vs_stats["total_chunks"],
                "dataset":       dataset_label,
            },
            "summary": summary,
            "cases": [
                {k: v for k, v in r.items() if k != "docs"}
                for r in final_results
            ],
        }
        out_path = Path(args.output or "data/eval_results.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(save_data, f, indent=2)
        print(f"Results saved to {out_path}")

    # Save baseline
    if args.save_baseline:
        save_baseline(summary, baseline_path=args.baseline)

    # Regression check
    if args.regression:
        print(f"\n{'='*60}")
        print("Regression Check")
        print("=" * 60)
        print_comparison_table(summary, baseline_path=args.baseline)
        regressions = compare_to_baseline(summary, baseline_path=args.baseline)
        if regressions:
            print("\nREGRESSIONS DETECTED:")
            for r in regressions:
                print(f"  {r}")
            print(f"\nTotal regressions: {len(regressions)}")
            sys.exit(1)
        else:
            print("No regressions detected.")


if __name__ == "__main__":
    main()

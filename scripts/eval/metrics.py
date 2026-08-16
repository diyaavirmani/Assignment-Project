"""
Standard information-retrieval metrics for the 3GPP RAG retrieval harness.

All functions operate on the same "docs" list format returned by
DocumentRetriever.retrieve() — a list of dicts with at minimum:
    {"source": str, "text": str, "similarity": float}

Metric taxonomy
---------------
STANDARD IR (these are the primary signals for a Layer 2 audience):
  - hit_rate_at_k     : binary; was any relevant doc in top-k?
  - recall_at_k       : fraction of relevant sources found in top-k
  - mrr               : mean reciprocal rank of the first relevant doc
  - ndcg_at_k         : normalised discounted cumulative gain

HEURISTIC (carried forward from the original harness, clearly labelled):
  - context_precision : fraction of retrieved chunks from a relevant source
  - context_recall    : fraction of expected keywords found in top-3 chunks
  Note: these are RAGAS-inspired keyword heuristics, NOT the IR-standard
  definitions of precision/recall.  They measure surface-level coverage,
  not true relevance judgements.  Treat them as supplementary signals.

Usage
-----
    from scripts.eval.metrics import recall_at_k, mrr, ndcg_at_k, hit_rate_at_k

    relevant_sources = ["38300", "38401"]
    docs = retriever.retrieve(query, top_k=10)

    r5  = recall_at_k(docs, relevant_sources, k=5)
    mrr_ = mrr(docs, relevant_sources)
    h5  = hit_rate_at_k(docs, relevant_sources, k=5)
    n5  = ndcg_at_k(docs, relevant_sources, k=5)
"""

import math
import re
from typing import List, Dict, Optional


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _is_relevant(doc: Dict, relevant_sources: List[str]) -> bool:
    """Return True when the doc's source filename contains any relevant spec token.

    Matching rules (tightened from plain substring). The corpus uses bare-digit
    spec filenames such as "38300-g10.docx" and "38401.docx":
      1. Token match on spec number: "38300" matches "38300-g10.docx" because the
         digits appear as a contiguous spec-number token, not as an accidental
         substring of a longer number. Each relevant_source string is matched
         against the source using a word-boundary-aware regex: the token must be
         preceded and followed by a non-digit character (or the start/end of the
         string). So "38300" does NOT match "383001-h10.docx", and "300" does NOT
         match "38300-g10.docx".
      2. Case-insensitive on the source path.

    This prevents "300" matching "38300" or "38001" matching "38300".

    Args:
        doc:              A retrieved document dict with at least a "source" key.
        relevant_sources: List of spec-number tokens (e.g. ["38300", "38401"]).

    Returns:
        True if the doc source matches any token, False otherwise.
    """
    source = doc.get("source", "")
    if not source or not relevant_sources:
        return False
    for rs in relevant_sources:
        # Require that the spec number is not flanked by other digits.
        # Handles: "38300-g10.docx", "38300.pdf", "TS38300-h20.docx"
        pattern = r'(?<!\d)' + re.escape(rs) + r'(?!\d)'
        if re.search(pattern, source, re.IGNORECASE):
            return True
    return False


# ---------------------------------------------------------------------------
# Standard IR metrics
# ---------------------------------------------------------------------------

def hit_rate_at_k(docs: List[Dict], relevant_sources: List[str], k: int = 5) -> float:
    """Hit-rate@k — 1.0 if any of the top-k docs is relevant, else 0.0.

    The simplest, most intuitive metric: did the retriever surface at least
    one useful chunk?  Analogous to Precision@1 across the binary relevant/
    irrelevant judgement but evaluated over the top-k window.

    Limitations: ignores rank position and the number of relevant docs
    retrieved; a single lucky hit scores the same as all-relevant top-k.

    Args:
        docs: Ranked list of retrieved docs (index 0 = most similar).
        relevant_sources: List of source-filename substrings marking a doc
            as ground-truth relevant (e.g. ["38300", "38401"]).
        k: Window size.

    Returns:
        1.0 or 0.0.
    """
    if not docs or not relevant_sources:
        return 0.0
    for doc in docs[:k]:
        if _is_relevant(doc, relevant_sources):
            return 1.0
    return 0.0


def recall_at_k(docs: List[Dict], relevant_sources: List[str], k: int = 5) -> float:
    """Recall@k — fraction of distinct relevant source specs found in top-k.

    Counts how many of the *expected* source spec numbers appear at least
    once anywhere in the top-k results.

    Example: expected = ["38300", "38401"], top-5 contains one chunk from
    38300 and none from 38401 → Recall@5 = 0.5.

    Limitations: treats each source spec as a single binary signal; does not
    account for how many chunks per spec were retrieved or whether the
    retrieved chunks contain the specific expected information.

    Args:
        docs: Ranked list of retrieved docs.
        relevant_sources: Ground-truth source spec numbers (e.g. ["38300"]).
        k: Window size.

    Returns:
        Float in [0, 1].
    """
    if not docs or not relevant_sources:
        return 0.0
    found = set()
    for doc in docs[:k]:
        for rs in relevant_sources:
            # Route through the same word-boundary matcher every other metric
            # uses, so "300" can't match "38300" and "383001" can't match "38300".
            if _is_relevant(doc, [rs]):
                found.add(rs)
    return len(found) / len(relevant_sources)


def mrr(docs: List[Dict], relevant_sources: List[str]) -> float:
    """Mean Reciprocal Rank — 1/rank of the first relevant doc.

    MRR rewards retrieval systems that put a relevant chunk at the top.
    A first-position hit scores 1.0; second position 0.5; tenth 0.1.

    For a single query this is simply Reciprocal Rank (RR); "Mean" applies
    when averaged across the golden set in the calling harness.

    Limitations: only the rank of the *first* relevant doc matters; all
    subsequent relevant docs are ignored.  Can be misleading for queries that
    need multiple relevant chunks for a complete answer.

    Args:
        docs: Ranked list of retrieved docs (index 0 = highest similarity).
        relevant_sources: Ground-truth source spec numbers.

    Returns:
        Float in (0, 1] if a relevant doc is found; 0.0 otherwise.
    """
    if not docs or not relevant_sources:
        return 0.0
    for rank, doc in enumerate(docs, start=1):
        if _is_relevant(doc, relevant_sources):
            return 1.0 / rank
    return 0.0


def ndcg_at_k(
    docs: List[Dict],
    relevant_sources: List[str],
    k: int = 5,
    graded_relevance: Optional[Dict[str, int]] = None,
) -> float:
    """nDCG@k — Normalised Discounted Cumulative Gain at depth k.

    nDCG is the gold standard retrieval metric.  It rewards:
      - relevance (binary or graded gain per retrieved doc)
      - rank position (a relevant doc at position 1 scores log2(2)=1;
        at position 2 it scores log2(3)≈0.63; etc.)
    and normalises by the ideal DCG (all relevant docs at the top).

    Binary mode (default):
        Each retrieved doc receives gain 1 if its source matches any entry in
        ``relevant_sources``, else 0.  The ideal DCG places all relevant docs
        first using gain = 1.

    Graded mode (when ``graded_relevance`` is provided):
        ``graded_relevance`` is a dict mapping source spec token → integer grade
        (e.g. {"38300": 2, "38401": 1}).  Each doc's gain is looked up from
        the dict; docs whose source doesn't match any key get gain 0.
        The ideal DCG places all docs in descending grade order.
        Typical grade scale: 2 = highly relevant, 1 = partially relevant,
        0 = not relevant.  The binary path is the special case where all grades
        are 1.

    Limitations: when using binary relevance, nDCG cannot distinguish between
    a highly relevant chunk and a marginally relevant one.  With graded labels
    the distinction is visible and the metric is more discriminating.

    Args:
        docs:              Ranked list of retrieved docs.
        relevant_sources:  Ground-truth source spec numbers (used in binary mode
                           and as a fallback allowlist in graded mode).
        k:                 Window size.
        graded_relevance:  Optional dict of {spec_token: grade_int} for graded mode.

    Returns:
        Float in [0, 1].  Returns 0.0 when relevant_sources is empty.
    """
    if not docs or not relevant_sources:
        return 0.0

    def _dcg(gains_list: List[float]) -> float:
        return sum(
            gain / math.log2(rank + 1)
            for rank, gain in enumerate(gains_list, start=1)
        )

    def _doc_gain(doc: Dict) -> float:
        if graded_relevance:
            source = doc.get("source", "")
            for spec_token, grade in graded_relevance.items():
                pattern = r'(?<!\d)' + re.escape(spec_token) + r'(?!\d)'
                if re.search(pattern, source, re.IGNORECASE):
                    return float(grade)
            return 0.0
        return 1.0 if _is_relevant(doc, relevant_sources) else 0.0

    gains = [_doc_gain(doc) for doc in docs[:k]]
    dcg = _dcg(gains)

    # Build ideal gains (best possible ranking).
    #
    # In both modes, the ideal must reflect the actual gains achievable given
    # the full retrieved doc list — not just the number of distinct spec tokens.
    # Multiple chunks from the same spec can each receive a non-zero gain, so
    # counting only distinct relevant_sources would undercount the ideal DCG
    # and push nDCG > 1.0.
    if graded_relevance:
        # Compute the gain each retrieved doc *would* receive in graded mode,
        # then sort descending to produce the ideal ranking.
        all_achievable = [_doc_gain(doc) for doc in docs]
        ideal_gains_full = sorted(all_achievable, reverse=True)
        ideal_gains = (ideal_gains_full + [0.0] * k)[:k]
    else:
        # Binary mode: count total relevant docs in the full retrieved list.
        n_relevant_total = sum(1 for doc in docs if _is_relevant(doc, relevant_sources))
        n_relevant = min(n_relevant_total, k)
        ideal_gains = [1.0] * n_relevant + [0.0] * (k - n_relevant)

    idcg = _dcg(ideal_gains)
    if idcg == 0.0:
        return 0.0
    return round(dcg / idcg, 4)


# ---------------------------------------------------------------------------
# Heuristic metrics (RAGAS-inspired; carried forward from original harness)
# ---------------------------------------------------------------------------

def context_precision(docs: List[Dict], relevant_sources: List[str]) -> float:
    """HEURISTIC — Fraction of retrieved chunks from a relevant source.

    RAGAS-inspired.  Not IR-standard precision: it measures source-level
    relevance over ALL retrieved chunks, not over a fixed k window.

    Limitation: a perfect precision score (1.0) is trivially achievable
    if the corpus only contains on-topic specs.  In a domain-specific RAG
    over 3GPP specs, almost all sources are relevant by construction, making
    this metric less discriminating than the IR-standard ones above.
    """
    if not docs:
        return 0.0
    hits = sum(1 for d in docs if _is_relevant(d, relevant_sources))
    return round(hits / len(docs), 4)


def context_recall_keywords(docs: List[Dict], expected_keywords: List[str]) -> float:
    """HEURISTIC — Fraction of expected keywords found in top-3 retrieved chunks.

    RAGAS-inspired keyword coverage, NOT IR-standard recall.  Measures
    surface-level vocabulary coverage, not true relevance.

    Limitation: keyword matching is brittle for technical text where the
    same concept appears under multiple terms (e.g. "gNB" vs "base station").
    Use this as a quick sanity check, not a primary quality signal.
    """
    if not docs or not expected_keywords:
        return 0.0
    combined = " ".join(d["text"].lower() for d in docs[:3])
    hits = sum(1 for kw in expected_keywords if kw.lower() in combined)
    return round(hits / len(expected_keywords), 4)


# ---------------------------------------------------------------------------
# Similarity thresholds (per embedding model)
# ---------------------------------------------------------------------------

# Legacy-pass similarity threshold, calibrated per embedding model.
#
# Derivation (reproducible): scripts/eval/calibrate_threshold.py on the
# 2026-07-02 full-index golden-set artifact — sweeping candidate thresholds
# for maximum Youden's J separating correct in-corpus retrievals
# (hit_rate_at_k == 1, N=22, min avg-sim 0.426) from out-of-corpus probes
# (N=5, max avg-sim 0.560) gives optimum 0.421 (TPR 1.0, FPR 0.2, J 0.8).
# 0.42 sits inside the optimal interval (0.416, 0.426).
#
# Known limitation: neg-003 (nonexistent-spec trap "TS 99.999", avg sim
# 0.560) scores above ANY threshold that admits all correct retrievals — a
# similarity cutoff cannot catch a probe that is lexically native to the
# corpus. That case is caught at the answer layer (refusal), not here.
PASS_SIM_THRESHOLDS = {
    "bge-small": 0.42,
}
DEFAULT_PASS_SIM_THRESHOLD = 0.50

# The retrieval-refusal boundary for out-of-corpus probes stays at the
# historical 0.50. It is a different dial (is the evidence weak enough to
# refuse? vs is the evidence strong enough to pass?), and the higher value
# preserves margin on the OOC side (highest true-OOC sim below the trap:
# 0.416).
REFUSAL_SIM_THRESHOLD = 0.50


def pass_sim_threshold(embedding_model: str) -> float:
    """Calibrated legacy-pass similarity threshold for an embedding model.

    Falls back to the historical 0.50 for models without a calibration run.
    """
    return PASS_SIM_THRESHOLDS.get(embedding_model, DEFAULT_PASS_SIM_THRESHOLD)


# ---------------------------------------------------------------------------
# Refusal / out-of-corpus detection (negative cases)
# ---------------------------------------------------------------------------

_REFUSAL_SIGNALS = (
    "i don't have",
    "i do not have",
    "not covered",
    "not in",
    "outside",
    "no information",
    "cannot find",
    "cannot answer",
    "no relevant",
    "not available",
    "not indexed",
    "not part of",
    "unable to find",
    "unable to answer",
    "out of scope",
    # Patterns observed in real refusals from the 2026-07-02 full-index run
    # (see data/eval_results.json answer_preview fields for neg-001..005):
    # the original list detected 0/5 actual refusals.
    "cannot provide",
    "can't provide",
    "unable to provide",
    "unrelated to",
    "not related to",
    "does not contain",
    "do not contain",
    "not enough information",
    "i don't know",
)

# Refusals lead with the decline: all observed real refusals state it in the
# opening sentence.  Scanning only the answer's opening avoids false positives
# from signal words ("outside", "not in") appearing deep in long technical
# answers.
_REFUSAL_SCAN_CHARS = 400


def is_refusal(answer: str) -> bool:
    """Heuristic check: does the answer signal that the system declined to answer?

    Used for negative/out-of-corpus eval cases.  Looks for surface-level
    refusal phrases in the lowercased opening of the answer (first
    ``_REFUSAL_SCAN_CHARS`` characters — observed refusals lead with the
    decline, and scanning full long answers produced false positives).

    Limitations: this is a heuristic.  A system may generate a correct refusal
    without matching any of these phrases, or may match a phrase in context
    without actually refusing (false positive).  For production evaluation use
    an LLM judge to score refusal quality.

    Args:
        answer: The generated answer string.

    Returns:
        True if the answer appears to be a refusal, False otherwise.
    """
    low = answer[:_REFUSAL_SCAN_CHARS].lower()
    return any(signal in low for signal in _REFUSAL_SIGNALS)


def refusal_rate(answers: List[str]) -> float:
    """Fraction of answers that appear to be refusals.

    Intended for evaluating a batch of out-of-corpus (negative) test cases
    where every answer *should* be a refusal.  A refusal_rate of 1.0 on the
    negative set means the system correctly declined all out-of-corpus queries.

    Status: [RUN REQUIRED] for full-index numbers — this function runs against
    real generated answers from the live system, which requires the built index
    and a live LLM.  The function itself is unit-testable with mocked answers.

    Args:
        answers: List of generated answer strings (one per negative query).

    Returns:
        Float in [0, 1].  Returns 0.0 for an empty list.
    """
    if not answers:
        return 0.0
    return round(sum(1 for a in answers if is_refusal(a)) / len(answers), 4)


# ---------------------------------------------------------------------------
# Per-case metric bundle
# ---------------------------------------------------------------------------

def compute_retrieval_metrics(
    docs: List[Dict],
    relevant_sources: List[str],
    expected_keywords: List[str],
    k: int = 5,
    graded_relevance: Optional[Dict[str, int]] = None,
) -> Dict:
    """Compute the full set of retrieval metrics for one query result.

    Returns a dict with both standard IR metrics (Recall@k, MRR, nDCG@k,
    hit-rate) and the legacy heuristic metrics (context precision/recall).

    Args:
        docs:               Ranked list returned by DocumentRetriever.retrieve().
        relevant_sources:   Ground-truth source spec numbers for this query.
        expected_keywords:  Keywords expected in relevant chunks (heuristic only).
        k:                  Window depth for @k metrics.
        graded_relevance:   Optional {spec_token: grade} dict for graded nDCG.
                            When provided, ndcg_at_k uses graded gains;
                            binary metrics (hit-rate, recall, MRR) continue
                            to use the binary relevant_sources list.

    Returns:
        {
            # Standard IR
            "hit_rate_at_k":   float,
            "recall_at_k":     float,
            "mrr":             float,
            "ndcg_at_k":       float,   # binary or graded depending on input
            # Heuristic (RAGAS-inspired)
            "context_precision":       float,   # heuristic
            "context_recall_keywords": float,   # heuristic (keyword coverage)
            # Meta
            "k":               int,
            "num_docs":        int,
            "graded":          bool,    # True when graded_relevance was used
        }
    """
    return {
        # Standard IR
        "hit_rate_at_k":          hit_rate_at_k(docs, relevant_sources, k),
        "recall_at_k":            recall_at_k(docs, relevant_sources, k),
        "mrr":                    mrr(docs, relevant_sources),
        "ndcg_at_k":              ndcg_at_k(docs, relevant_sources, k,
                                             graded_relevance=graded_relevance),
        # Heuristic
        "context_precision":      context_precision(docs, relevant_sources),
        "context_recall_keywords": context_recall_keywords(docs, expected_keywords),
        # Meta
        "k":       k,
        "num_docs": len(docs),
        "graded":   graded_relevance is not None,
    }

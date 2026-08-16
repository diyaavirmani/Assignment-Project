"""
Comparison-query decomposition for 3GPP retrieval.

Why
---
The two retrieval misses that survived Track B's expansion fusion (SA-vs-NSA,
NR-vs-LTE physical layer) are comparison questions. A single query embedding
has to average both sides of a "difference between X and Y" question, so it
lands between the two document neighborhoods and retrieves neither well.
Decomposition issues one sub-query per side; the retriever merges the
sub-query rankings with the raw ranking via the same Reciprocal Rank Fusion
used for vocabulary expansion, so decomposition can only add candidate
evidence — the raw ranking is never discarded.

Heuristics and limitations
--------------------------
Side extraction is deliberately simple, pattern-based, and uniform:

* "… between A and B <tail>" — A is a short non-greedy capture, B is the
  single token after "and", and the remaining tail is appended to both
  sides ("between SA and NSA 5G deployment options" -> "SA 5G deployment
  options" / "NSA 5G deployment options").
* "A vs B" / "A versus B" — one- or two-word sides.

A multi-word right side ("between X and the secondary node") truncates to
its first token, and a tail that repeats part of a side reads awkwardly —
both are harmless for embedding purposes and accepted for simplicity.
Non-comparison queries return no sub-queries and cost nothing.
"""
import re
from typing import List

# "… between A and B <tail>"  (A: 1-40 chars non-greedy; B: single token)
_BETWEEN = re.compile(
    r"\bbetween\s+(.{1,40}?)\s+and\s+([A-Za-z0-9./-]+)\s*(.*)",
    re.IGNORECASE,
)

# "A vs B" / "A vs. B" / "A versus B"  (sides: one or two words)
_VS = re.compile(
    r"([A-Za-z0-9./-]+(?:\s+[A-Za-z0-9./-]+)?)\s+(?:vs\.?|versus)\s+"
    r"([A-Za-z0-9./-]+(?:\s+[A-Za-z0-9./-]+)?)",
    re.IGNORECASE,
)

# Cheap pre-filter: only queries with explicit comparison wording are
# candidates. Bare "between" is deliberately NOT a hint — interface
# questions ("the F1 interface between gNB-CU and gNB-DU", "the NG
# interface between NG-RAN and 5GC") use "between" without being
# comparisons, and decomposing them splits a single concept in half.
_COMPARE_HINTS = ("differ", "compar", " vs ", " vs. ", "versus")


def _clean(text: str) -> str:
    return text.strip(" \t?.!,")


def decompose_query(query: str) -> List[str]:
    """Split a comparison-style question into one sub-query per side.

    Returns an empty list for non-comparison queries (the common case),
    or a two-element list of sub-query strings.

    Examples:
        "What is the difference between SA and NSA 5G deployment options?"
            -> ["SA 5G deployment options", "NSA 5G deployment options"]
        "NR vs LTE handover procedures"
            -> ["NR", "LTE"]
    """
    low = query.lower()
    if not any(h in low for h in _COMPARE_HINTS):
        return []

    m = _BETWEEN.search(query)
    if m:
        side_a, side_b, tail = _clean(m.group(1)), _clean(m.group(2)), _clean(m.group(3))
        sub_a = f"{side_a} {tail}".strip()
        sub_b = f"{side_b} {tail}".strip()
        if sub_a and sub_b and sub_a != sub_b:
            return [sub_a, sub_b]
        return []

    m = _VS.search(query)
    if m:
        side_a, side_b = _clean(m.group(1)), _clean(m.group(2))
        if side_a and side_b and side_a.lower() != side_b.lower():
            return [side_a, side_b]

    return []

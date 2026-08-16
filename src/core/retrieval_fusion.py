"""
Retrieval rank-fusion and query-rewriting orchestration.

This module is the single source of truth for the Track B retrieval
improvements (query-expansion fusion, comparison decomposition, coverage
capping). It is deliberately **pure Python** — it imports only stdlib plus
the two pure-stdlib rewriter modules — so it can run on both retrieval
backends without dragging in a heavy dependency:

  * the local / API path (``DocumentRetriever`` over sentence-transformers
    bge-small), and
  * the Streamlit Cloud path (``streamlit_app.py`` over ChromaDB's built-in
    ONNX embeddings), which must stay torch-free.

The orchestration is embedding-agnostic: it operates on the query *text*
(rewriting) and on ranked *result lists* (fusion/allocation), never on the
vectors themselves. It works against any search backend via a ``search_fn``
callback:

    search_fn(text: str, n: int) -> List[Dict]

returning up to ``n`` result dicts, each carrying at least ``source`` and
``chunk_index`` (the fusion de-duplication key) and ``similarity``.

Note on calibration: the fusion *mechanics* transfer across embedding
models, but tuned *constants* do not. The 0.42 legacy-pass similarity
threshold (``scripts/eval/metrics.py``) was calibrated for bge-small and is
an eval-time display concern, not part of retrieval — it is intentionally
absent here. A different backend (e.g. the ONNX cloud path) would need its
own calibration before that threshold is applied to it.
"""

from typing import Callable, Dict, List

from src.core.query_decomposition import decompose_query
from src.core.query_expansion import expand_query

# A backend search primitive: given query text and a result cap, return up
# to that many ranked result dicts.
SearchFn = Callable[[str, int], List[Dict]]

# Maximum chunks from a single source document in a fused top-k. Multi-spec
# questions need evidence from more than one spec; without a cap a single
# strong document can fill every slot. Value chosen so a dominant source
# keeps a clear majority (3 of 5) while up to two slots stay open for other
# sources; effects are measured by the eval harness.
MAX_CHUNKS_PER_SOURCE = 3


def _chunk_key(doc: Dict):
    return (doc.get("source"), doc.get("chunk_index"))


def rrf_merge(ranked_lists: List[List[Dict]], k_const: int = 60) -> List[Dict]:
    """Merge ranked doc lists via Reciprocal Rank Fusion.

    RRF score(doc) = sum over lists of 1 / (k_const + rank), so a doc ranked
    well in several lists beats a doc ranked well in only one. Chunks are
    identified by (source, chunk_index); the first-seen doc dict (with its
    own similarity) is kept for each chunk.

    Why fusion: replacing the query with its vocabulary-expanded form fixed
    vocabulary-divergent misses but regressed previously-good queries
    (measured on bge-small: hit-rate@5 0.88 -> 0.80). Fusing the raw and
    expanded rankings keeps the raw ranking's signal while the expanded
    ranking rescues queries whose phrasing diverges from spec terminology.
    """
    scores: Dict = {}
    docs_by_key: Dict = {}
    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked, start=1):
            key = _chunk_key(doc)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k_const + rank)
            docs_by_key.setdefault(key, doc)
    ordered = sorted(scores, key=lambda key: scores[key], reverse=True)
    return [docs_by_key[key] for key in ordered]


def cap_per_source(
    docs: List[Dict], k: int, max_per_source: int = MAX_CHUNKS_PER_SOURCE
) -> List[Dict]:
    """Take the top-k of a ranked list, limiting chunks per source.

    Walks the ranking in order, deferring chunks beyond ``max_per_source``
    from the same source document; deferred chunks backfill only if fewer
    than k others exist.
    """
    selected: List[Dict] = []
    deferred: List[Dict] = []
    counts: Dict = {}
    for doc in docs:
        source = doc.get("source")
        if counts.get(source, 0) < max_per_source:
            selected.append(doc)
            counts[source] = counts.get(source, 0) + 1
        else:
            deferred.append(doc)
        if len(selected) >= k:
            break
    for doc in deferred:
        if len(selected) >= k:
            break
        selected.append(doc)
    return selected[:k]


def interleave_sides(side_lists: List[List[Dict]], backfill: List[Dict], k: int) -> List[Dict]:
    """Allocate top-k slots round-robin across a comparison's sides.

    Each side's list is walked in rank order; a chunk is taken only if its
    source document is not yet represented (a comparison needs breadth
    across specs, not depth in one). Remaining slots backfill from the fused
    base ranking.
    """
    selected: List[Dict] = []
    seen_chunks: set = set()
    seen_sources: set = set()

    cursors = [0] * len(side_lists)
    exhausted = [False] * len(side_lists)
    while len(selected) < k and not all(exhausted):
        for i, side in enumerate(side_lists):
            if len(selected) >= k:
                break
            while cursors[i] < len(side):
                doc = side[cursors[i]]
                cursors[i] += 1
                key = _chunk_key(doc)
                if doc.get("source") in seen_sources or key in seen_chunks:
                    continue
                selected.append(doc)
                seen_chunks.add(key)
                seen_sources.add(doc.get("source"))
                break
            else:
                exhausted[i] = True

    for doc in backfill:
        if len(selected) >= k:
            break
        key = _chunk_key(doc)
        if key in seen_chunks:
            continue
        selected.append(doc)
        seen_chunks.add(key)
    return selected[:k]


def fused_retrieve(
    query: str,
    search_fn: SearchFn,
    k: int,
    *,
    expand: bool = True,
    decompose: bool = True,
) -> List[Dict]:
    """Retrieve for a query with query-rewriting rank fusion.

    Two rewriters, two merge strategies:

    1. Vocabulary expansion (TR 21.905 full forms) — RRF-fused with the raw
       ranking. The raw ranking is never discarded: replacing it was
       measured to regress previously-good queries (see ``rrf_merge``).
    2. Comparison decomposition — a "difference between X and Y" question
       gets one sub-query per side. RRF is deliberately NOT used to merge
       the sides: it rewards consensus across lists, and a comparison's
       per-side evidence appears in exactly one list, so it always loses to
       consensus noise (measured: global RRF left both comparison misses at
       hit-rate 0). Sides get round-robin slot allocation instead, with the
       fused base ranking as backfill.

    Queries that trigger neither rewriter run a single search and return.

    Args:
        query: The user's natural-language query.
        search_fn: Backend primitive ``(text, n) -> List[Dict]`` returning
            up to n ranked result dicts (each with source/chunk_index).
        k: Number of results to return.
        expand: Enable vocabulary-expansion fusion.
        decompose: Enable comparison decomposition.

    Returns:
        Up to k result dicts.
    """
    base_raw = search_fn(query, k)

    expanded = expand_query(query) if expand else query
    has_expanded = expand and expanded != query
    subqueries = decompose_query(query) if decompose else []

    if not has_expanded and not subqueries:
        return base_raw[:k]

    base_lists = [base_raw]
    if has_expanded:
        base_lists.append(search_fn(expanded, k))
    base = rrf_merge(base_lists)

    if subqueries:
        # Each side is itself a raw+expanded RRF fusion — the same
        # never-discard-the-raw-text principle as the main path (searching
        # only the expanded side text was measured to push the relevant
        # spec out of the side's own top-10). Sides search deeper (2k) so
        # per-source dedup has room.
        side_lists: List[List[Dict]] = []
        for sub in subqueries:
            sub_lists = [search_fn(sub, k * 2)]
            if expand:
                sub_expanded = expand_query(sub)
                if sub_expanded != sub:
                    sub_lists.append(search_fn(sub_expanded, k * 2))
            side_lists.append(rrf_merge(sub_lists) if len(sub_lists) > 1 else sub_lists[0])
        return interleave_sides(side_lists, base, k)

    return cap_per_source(base, k)

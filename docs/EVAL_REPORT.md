# Evaluation Report — 3GPP RAG Assistant

**Last run (full + LLM-judge):** 2026-07-03 (`--full --judge --judge-provider groq`; see `data/eval_results.json`; retrieval uses query-expansion rank fusion + comparison decomposition)
**Retrieval baseline:** 2026-07-03 Track B baseline (committed in `data/eval/baseline.json`; hit-rate@5 0.96, Recall@5 0.737). History: 2026-06-25 pre-Track-B baseline (0.88 / 0.64, reproduced exactly by two runs), then the 2026-07-02 fusion increment (0.88 / 0.683).
**Last run (sample fixture):** 2026-06-23 (see `data/eval/sample_results.json`)
**Harness version:** v4 (v3 + per-model calibrated pass threshold, query-expansion rank fusion, fixed refusal detector with answer-refusal aggregation, answer-relevance golden-set fallback, non-destructive `--regression`)
**Index size:** 43,121 chunks (local/API index; the hosted Streamlit demo loads a 41,429-chunk subset sized for free-tier deployment)
**Dataset:** 30-query golden set — 25 in-corpus + 5 out-of-corpus refusal probes
**Reproduce (sample fixture, no index required):** `python scripts/eval/run_sample_eval.py`
**Reproduce (retrieval only):** `python scripts/eval_retrieval.py --output data/eval_results.json`
**Reproduce (full + judge):** `python scripts/eval_retrieval.py --full --judge --judge-provider groq --output data/eval_results.json`

---

## Overview

This document defines every metric in the eval harness, explains why it matters, states its limitations honestly, and cites the current real numbers from the full-index runs (43,121 chunks, 30-query golden set): the 2026-06-25 retrieval baseline and the 2026-07-02 full + LLM-judge run. Standard IR metrics are reported over the 25 in-corpus queries; the refusal axis over the 5 out-of-corpus probes. Anything not yet measured is marked **[RUN REQUIRED]** — those placeholders are the correct deliverable when the infrastructure is not available at report-generation time.

---

## 1. Retrieval metrics

### 1.1 Standard IR metrics (primary)

These are the signals a skeptical enterprise-RAG PM or technical reviewer will look for. They come from information-retrieval literature and have well-understood properties.

#### Hit-rate@k

**Definition:** 1.0 if any of the top-k retrieved chunks comes from a ground-truth relevant source specification; 0.0 otherwise. Averaged across all queries in the dataset.

**Why it matters:** The minimum bar for usefulness. If the retriever surfaces zero relevant chunks for a query, no amount of LLM quality saves the answer. For enterprise RAG over technical/regulated documents, a hit-rate below 0.80 is a serious problem.

**Limitation:** Binary and tolerant — a single lucky chunk in position k scores the same as k/k relevant chunks. Doesn't distinguish a retriever that gets the right chunk at rank 1 from one that barely squeaks a relevant chunk in at rank k. Use alongside MRR and nDCG for a complete picture.

**Current value (full index, in-corpus N=25):** **0.96** (0.88 before Track B's comparison decomposition recovered both remaining misses to at least one relevant source). It sits above the 84% legacy pass rate because hit-rate credits a relevant chunk in top-5 without also requiring the keyword and similarity thresholds the legacy criterion imposes.

---

#### Recall@k

**Definition:** Fraction of the distinct expected source specifications found at least once anywhere in the top-k results. For a query expecting chunks from TS 38.300 and TS 38.401, a top-5 result containing one 38.300 chunk and no 38.401 chunk scores 0.5.

**Why it matters:** Many complex 3GPP questions require evidence from multiple specifications (e.g., the F1 interface is described in both TS 38.401 architecture and TS 38.470/38.473 protocol specs). Recall@k measures whether the retriever casts a wide enough net across the relevant spec set, not just whether it finds one good chunk.

**Limitation:** Treats each expected source as a single binary signal — it doesn't account for how many chunks per spec were retrieved or whether the retrieved chunks contain the specific expected information. A query expecting two specs where the retriever finds one chunk from each scores 1.0, regardless of chunk quality.

**Current value (full index, in-corpus N=25):** **0.737** (0.64 → 0.683 → 0.737 across the two Track B increments). Still lower than hit-rate because several multi-spec queries surface some but not all expected specs. The measured residual against the 0.80 M9 target is an embedding-resolution limit, not query mechanics: direct probing shows "NR physical layer" cannot surface TS 38.211 in its own top-12 — bge-small ranks sibling PHY specs above the labeled definitional spec (see §7).

---

#### MRR (Mean Reciprocal Rank)

**Definition:** 1/rank of the first relevant document retrieved, averaged across all queries. A relevant chunk at rank 1 scores 1.0; at rank 2, 0.5; at rank 10, 0.1.

**Why it matters:** For conversational RAG, the LLM's answer quality correlates with how early a relevant chunk appears in the context window. An LLM given 5 chunks where the only relevant one is at position 5 will produce a worse answer than one where the relevant chunk is at position 1. MRR captures this rank sensitivity.

**Limitation:** Only the rank of the **first** relevant document matters — all subsequent relevant documents are ignored. This makes it less informative when queries require multiple relevant chunks (common in 3GPP reasoning). For multi-evidence queries, nDCG@k is more appropriate.

**Current value (full index, in-corpus N=25):** **0.720** (0.679 before Track B). When a relevant chunk is retrieved at all, it tends to land near the top — consistent with the high hit-rate.

---

#### nDCG@k (Normalised Discounted Cumulative Gain)

**Definition:** Measures both relevance and rank position, then normalises by the ideal (best possible) ranking. Supports two modes:

- **Binary mode (default):** Each retrieved chunk receives gain 1 if its source matches a relevant spec token, else 0. The ideal ranking places all relevant chunks first.
- **Graded mode:** Each chunk receives a gain from a `graded_relevance` dict (e.g. `{"38300": 2, "38401": 1}`), where 2 = highly relevant (primary spec), 1 = partially relevant (supporting spec), 0 = not relevant. The ideal ranking orders chunks by descending gain. Graded nDCG is more discriminating than binary nDCG because it distinguishes the primary definitional spec from secondary supporting specs.

The gain at each rank is discounted: gain(rank) = gain / log2(rank + 1). The score is normalised to [0, 1] by dividing by the DCG of the ideal ranking.

**Why it matters:** nDCG is the most complete single retrieval metric. It rewards the retriever for putting relevant chunks early (like MRR) while also crediting retrieval of multiple relevant chunks (unlike MRR). It's the metric used in TREC, MS-MARCO, and most enterprise RAG benchmark papers. Reviewers of enterprise RAG systems expect to see nDCG. The graded variant adds signal for queries where the retrieval system must distinguish primary from supporting evidence.

**Implementation note:** The ideal DCG is computed from the actual gains achievable in the full retrieved list (not just the number of distinct spec tokens). This correctly handles the case where multiple chunks from the same spec appear in the result set, preventing nDCG > 1.0.

**Limitation:** Binary relevance flattens the distinction between a highly relevant chunk and a marginally relevant one. Graded relevance is more informative but requires additional annotation effort. A subset of 7 golden-set queries now carry graded labels (see §3).

**Current value (full index, in-corpus N=25):** **0.755** (graded nDCG@5; 0.723 pre-Track-B, briefly 0.718 after the fusion increment before decomposition lifted it) — the harness reports this as a "GOOD" band. This is the single most complete retrieval signal in the report: it credits both early placement and multiple relevant chunks.

**Sample fixture result (2026-06-23):**
- nDCG@5 binary: **0.984** (avg over 6 fixture queries)
- nDCG@5 graded: **0.982** (avg over 6 fixture queries)
- Note: these are sample fixture numbers from `data/eval/sample_results.json`. They confirm the harness computes graded nDCG correctly; they do not reflect full-corpus quality (the full-index value above is the real one).

---

### 1.2 Heuristic metrics (RAGAS-inspired; supplementary)

These were the original harness metrics. They are keyword-based heuristics, not IR-standard precision/recall. They are retained for trend continuity but should be treated as supplementary sanity checks, not primary quality signals.

#### Context precision (heuristic)

**Definition:** Fraction of ALL retrieved chunks (not just top-k) whose source filename matches a ground-truth relevant specification.

**Why it matters as a sanity check:** A score near 1.0 indicates the retriever is not surfacing chunks from obviously irrelevant spec series. In a domain-specific corpus like 3GPP, high precision is expected by construction — the index only contains 3GPP specs, and most are relevant to most queries.

**Limitation:** In a domain-specific RAG corpus, this metric is easily near-1.0 without any real quality signal. It cannot distinguish between a retriever that returns the specific relevant spec section vs. any spec section. It is NOT IR-standard precision. The name is kept for API compatibility.

**Full-index result (2026-07-03, in-corpus N=25):** avg_context_precision = **0.456**. This is lower than the old 10-query figure (1.0) because the 30-query golden set is harder and broader, and because this strict source-filename definition penalizes retrieving a *related* spec section that isn't the exact expected file. It is a deliberately conservative heuristic, not the primary signal — read it alongside hit-rate@5 (0.96).

---

#### Context recall (heuristic keyword coverage)

**Definition:** Fraction of expected technical keywords found anywhere in the combined text of the top-3 retrieved chunks. For example, a query expecting ["gnb", "base station", "node", "ran"] scores 0.75 if 3 of 4 keywords appear in the top-3 chunks.

**Why it matters as a sanity check:** A quick check that the retrieved chunks contain the basic vocabulary of the right answer. Cheap to compute; no LLM required.

**Limitation:** Brittle for technical text. The same concept often appears under multiple terms ("gNB" vs "base station" vs "next-generation NodeB"). A chunk that perfectly explains the concept using different terminology will score 0 on a keyword it should score 1 on. This is part of why several in-corpus queries miss the legacy pass bar despite the retriever surfacing genuinely relevant chunks (see §6). This is NOT IR-standard recall.

**Full-index result (2026-07-03, in-corpus N=25):** avg_context_recall = **0.75** (0.71 pre-fusion; individual query scores range from 0.0 to 1.0)

---

### 1.3 Latency

**Definition:** Wall-clock time for a single retrieval call (embed query + ANN search in ChromaDB), measured in seconds. Reported as p50 and p95 across all queries in the run.

**Why it matters:** For interactive use, retrieval latency is the dominant component of perceived response time (LLM generation is slower but asynchronous in streaming UI). Sub-400ms retrieval keeps the overall time-to-first-token competitive for technical users who expect search-engine-like responsiveness.

**Limitation:** Latency is hardware- and load-dependent. These numbers were recorded on a specific machine (Apple M2, CPU only). The embedding model (bge-small ONNX) loads on first query, so a cold session's first query is slower; over a 30-query run the warm-up is amortized into p95.

**Full-index results (Apple M2 CPU, 30-query runs):**

| Metric | Pre-Track-B (2026-06-25) | Track B baseline (2026-07-03) |
|---|---|---|
| Retrieve p50 | 0.021s | **0.069s** |
| Retrieve p95 | 0.048s | **0.213s** |

Expansion fusion runs two embed+search passes for queries containing known abbreviations (the ~3× p50 increase), and comparison queries run up to six (the p95 tail) — accepted tradeoffs (ADR-009, ADR-010), still two orders of magnitude below generation latency. Queries triggering neither rewriter run a single pass. Generation latency (Groq / Ollama) is separate and tracked in METRICS.md.

---

## 2. Answer quality metrics

### 2.1 Heuristic answer metrics (no LLM required)

These run when `--full` is passed. They use keyword matching and text-overlap heuristics — fast but approximate.

#### Answer relevance (heuristic)
Fraction of expected answer keywords present in the generated response. Same brittleness caveats as context recall above.

#### Faithfulness (heuristic)
Proxy for grounding: scores (a) presence of grounding vocabulary ("according to", "3GPP", "specification") and (b) word-overlap between the answer and the top retrieved chunk. This is a surface heuristic — it can be fooled by an answer that copies terminology from the retrieved chunk without actually being grounded in it.

**2026-07-03 results** (responder: local Ollama `llama3.2`, 30 answers):

| Heuristic metric | Value | Note |
|---|---|---|
| Answer relevance | **0.77** | Above the 0.70 target. Computed for the golden set via the `expected_keywords` fallback (Track B fix — earlier runs scored a flat 0.0 because the metric read a field the golden set lacks) |
| Faithfulness (heuristic) | **0.716** | Below the 0.80 target on this run; earlier runs scored 0.716–0.834 across four runs. Answers regenerate at temperature 0.1 and this surface heuristic is noisy — treat as directional |
| Composite answer score | **0.794** | Above the 0.70 target (earlier runs' lower composites were deflated by the relevance defect) |
| Answer pass rate | **1.00** | Composite pass criterion |

---

### 2.2 LLM-judge metrics (faithfulness + answer-correctness)

**Status: measured (latest run 2026-07-03).** Responder: local Ollama `llama3.2` (the local deploy path). Judge: Groq `llama-3.3-70b-versatile` — a different, much larger model than the responder, which gives the independent-judge setup this report recommends. Note the scores below evaluate the **local** answer path; the cloud path (Groq 70B as responder) has not been judged and is a named follow-up.

**Activate with:**
```
python scripts/eval_retrieval.py --full --judge --judge-provider groq
# Requires: GROQ_API_KEY environment variable
```

#### Faithfulness (LLM-judge)
**Definition:** LLM judge scores 0–5: does the answer make claims supported by the retrieved context? 5 = every claim traces to a context passage; 0 = answer ignores or contradicts the context. Normalised to [0, 1].

**Why it matters:** Hallucination in technical/regulated domains is a safety and compliance risk. For 3GPP specs used in network engineering, a hallucinated interface definition or protocol requirement could propagate errors into product decisions. LLM-judge faithfulness is the closest proxy for hallucination rate short of human review.

**Limitation:** The judge model's own capabilities bound the score quality. Using the same model as the responder is a known bias risk (self-judging); the 2026-07-02 run avoided it by judging local llama3.2 answers with Groq llama-3.3-70b. Judge prompts must be versioned; changing the prompt changes scores.

**Current value (2026-07-03, N=30):** **0.64** average (four runs: 0.647, 0.68, 0.647, 0.64). Judge notes most often credit grounding but flag missing detail — consistent with a 2 GB responder model summarizing 1,000-char chunks.

---

#### Answer correctness (LLM-judge)
**Definition:** LLM judge scores 0–5: does the answer address the question accurately and completely, given what the retrieved context provides? This is not a reference-answer comparison (the golden set does not include full reference answers) — it is the judge's assessment of technical accuracy within the context window. Normalised to [0, 1].

**Why it matters:** Answer relevance (keyword overlap) misses cases where an answer uses the right vocabulary but gives the wrong technical explanation. LLM-judge correctness catches substantive errors that keyword heuristics cannot.

**Limitation:** Without ground-truth reference answers, "correctness" is the judge's interpretation, not a ground-truth oracle. Treat as a directional signal; calibrate by sampling and reviewing judge reasoning on a small set. Note the average below includes the 5 out-of-corpus probes, which the judge scores 0 by design (a correct refusal "does not address the question").

**Current value (2026-07-03, N=30):** **0.387** average (four runs: 0.387, 0.40, 0.38, 0.387). Over the 25 in-corpus cases alone the picture is a small responder model that grounds its answers but frequently lacks depth — the judge's most common note is "correct but lacks specific details." This is a responder-capability ceiling, not a retrieval failure, and it quantifies the quality gap between the 2 GB local model and the 70B cloud path.

---

## 3. Golden dataset

**File:** `data/eval/golden_set.jsonl`
**Size:** 30 queries (25 positive + 5 negative/refusal cases)
**Format:** JSONL, one case per line. Positive cases have fields: `id`, `query`, `relevant_sources` (list of TS spec numbers), `expected_keywords`, `expected_facts`, `spec_refs`, `notes`. Negative cases additionally have `is_out_of_corpus: true`, `expected_behavior: "refusal"`, and `refusal_hint`.

### Dataset construction

Queries were written to cover:
- Core architecture concepts (gNB, NG-RAN, CU/DU split) → TS 38.300, 38.401
- Interface specifications (F1, E1, Xn, NG, X2) → TS 38.470, 38.460, 38.420, 38.410, 36.423
- Protocol specifications (MAC, RLC, PDCP, RRC, SDAP) → TS 38.321, 38.322, 38.323, 38.331
- Physical layer (SSB, HARQ, RACH, NR vs LTE) → TS 38.211, 38.213, 36.211
- Advanced 5G features (dual connectivity, network slicing, beamforming) → TS 38.300, 38.331

### What is and isn't ground truth

`relevant_sources` and `expected_keywords` are genuine ground truth — they identify the 3GPP specifications that define the answer to each query, based on the actual 3GPP catalog. These are checkable against the corpus.

`expected_facts` are directionally correct technical claims about 3GPP specifications. They are NOT full reference answers and should not be used for automated answer-correctness scoring without human review. They serve as audit markers for a human reviewer confirming whether a generated answer addressed the core technical point.

### Graded relevance labels (v3 addition)

Seven queries carry a `graded_relevance` field: a dict mapping spec token to integer grade:
- **2** = primary/definitional spec — the system must retrieve from this spec to answer the query correctly.
- **1** = supporting spec — provides useful context but not the primary definition.
- **0** = not relevant (implicit; any spec not in the dict).

Queries with graded labels: gs-001, gs-002, gs-003, gs-004, gs-005, gs-008, gs-009. These cover gNB definition, F1/E1 interfaces, NG-RAN architecture, handover, MAC, and RRC.

When `graded_relevance` is present in a case, `ndcg_at_k` uses graded gains. Binary metrics (hit-rate, Recall@k, MRR) continue to use the binary `relevant_sources` list.

### Negative / out-of-corpus cases (v3 addition)

Five cases (`neg-001` through `neg-005`) are explicitly out-of-corpus:
- `neg-001`: Food/nutrition question (completely unrelated domain)
- `neg-002`: Kubernetes/Calico networking (cloud-native, not 3GPP)
- `neg-003`: Non-existent spec number TS 99.999 (hallucination trap)
- `neg-004`: Ericsson stock price (financial, not telecom standards)
- `neg-005`: Cisco IOS BGP configuration (vendor-specific, adjacent domain)

These test that the system declines rather than confabulating a 3GPP answer. **Measured 2026-07-03: answer-refusal rate 5/5 (1.0), machine-scored.** The original detector had scored 0/5 on the same behavior — its phrase list did not match the refusal wording the model actually produces ("I must point out…", "I cannot provide an answer…") — so Track B widened the phrase list using the observed real refusals as regression fixtures and scoped the scan to the answer's opening (see `scripts/eval/metrics.py`). The `answer_refusal_rate` is now aggregated into the eval summary whenever `--full` runs.

### What was NOT done (remaining gaps)

- No paraphrase variants per query
- No human annotation of individual retrieved chunks (only source-level relevance)
- No judge-based refusal check (the detector is a phrase heuristic; a judge would grade refusal *quality*)

These are the logical next steps for a v4 golden set.

---

## 4. Regression mode

**Activate with:**
```
# Save current run as baseline
python scripts/eval_retrieval.py --save-baseline

# On subsequent runs, check for regression (CI gate)
python scripts/eval_retrieval.py --regression
```

**Tracked metrics:**
- `retrieval.avg_hit_rate_at_k` (tolerance: ±0.05)
- `retrieval.avg_recall_at_k` (tolerance: ±0.05)
- `retrieval.avg_mrr` (tolerance: ±0.05)
- `retrieval.avg_ndcg_at_k` (tolerance: ±0.05)
- `retrieval.avg_context_precision` (tolerance: ±0.05)
- `retrieval.avg_context_recall` (tolerance: ±0.05)
- `latency.retrieve.p50` (tolerance: +0.10s)
- `latency.retrieve.p95` (tolerance: +0.20s)

**Regression definition:** A metric regresses when the current value falls below `baseline - tolerance` (quality metrics) or rises above `baseline + tolerance` (latency). The tolerances are set to absorb normal run-to-run noise while catching genuine quality drops.

**CI integration:** The eval script exits with code 1 on regression. Add to CI:
```yaml
- name: Regression check
  run: python scripts/eval_retrieval.py --regression
```
This gate is meaningful once a baseline is established from a full run on the complete index.

**Baseline file:** `data/eval/baseline.json` — created by `--save-baseline`, tracked in version control. Each baseline commit should note the index version and date it was produced.

**Current baseline:** `data/eval/baseline.json` is the full-index fusion baseline (2026-07-02, 43,121 chunks, 25 in-corpus queries, query-expansion rank fusion enabled), tracked in version control so `--regression` compares like-for-like. Regenerate after an index, model, or retrieval-behavior change with `python scripts/eval_retrieval.py --save-baseline`.

**Committed baseline values (full index, in-corpus N=25):**
- avg_hit_rate_at_k: 0.880
- avg_recall_at_k: 0.683
- avg_mrr: 0.697
- avg_ndcg_at_k: 0.718
- avg_context_precision: 0.440
- avg_context_recall: 0.752
- retrieve p50 / p95: ~0.079s / ~0.137s (two searches per expanded query — see §1.3)

**Non-destructive by design (Track B fix):** running `--regression` alone no longer writes `data/eval_results.json` — the gate is read-only unless an explicit `--output` is given. (An earlier run of the gate silently overwrote a judge-eval artifact; the CI step's `--no-save` is retained as belt-and-braces.)

---

## 5. How to reproduce

### Sample fixture run (no index, no LLM required — runs immediately)
```bash
python scripts/eval/run_sample_eval.py
```
Runs 6 fixture queries against a 10-chunk in-memory TF-IDF index. Confirms the harness computes all metrics end-to-end. Results are labeled "sample fixture — NOT full-corpus quality."

To write results and set as regression baseline:
```bash
python scripts/eval/run_sample_eval.py --output data/eval/sample_results.json --save-baseline
```

### Fast retrieval-only run (no LLM required; requires built index)
```bash
python scripts/eval_retrieval.py --output data/eval_results.json
```
Runs all 30 golden-set queries (25 positive + 5 negative), computes standard IR + heuristic retrieval metrics, writes results to `data/eval_results.json`. Typical runtime: 10–30 seconds depending on embedding warm-up.

### With heuristic answer quality (requires Ollama)
```bash
ollama serve
ollama pull llama3.2
python scripts/eval_retrieval.py --full --output data/eval_results.json
```

### With LLM-judge [RUN REQUIRED]
```bash
export GROQ_API_KEY=your_key_here
python scripts/eval_retrieval.py --full --judge --judge-provider groq --output data/eval_results.json
```

### Legacy 10-query run (original inline test set)
```bash
python scripts/eval_retrieval.py --no-golden --output data/eval_results.json
```

### Build the index first (if not already built)
```bash
python scripts/download_specs.py
python scripts/build_index.py
```
Index build requires ~2–4 GB disk for 3GPP spec downloads and ~30–60 minutes for chunking + embedding 43k+ passages.

---

## 6. Current results

### 6.1 Sample fixture run (2026-06-23) — harness smoke test

Produced by `python scripts/eval/run_sample_eval.py`. Results in `data/eval/sample_results.json`.

**Fixture: 10 hand-built chunks, 6 queries, k=5, TF-IDF similarity (no vector DB or LLM)**

These numbers confirm the harness computes every metric correctly end-to-end. They are NOT representative of full-corpus quality.

| Metric | Value | Note |
|---|---|---|
| Hit-rate@5 | **1.000** | All 6 fixture queries retrieved ≥1 relevant chunk in top 5 |
| Recall@5 | **1.000** | All expected source specs found in top 5 for every query |
| MRR | **1.000** | Relevant chunk ranked first for every query |
| nDCG@5 (binary) | **0.984** | Avg; fix-001 scores 0.906 (two 38300 chunks split the grade) |
| nDCG@5 (graded) | **0.982** | Avg with 0/1/2 grades; confirms graded path works |
| Context precision (heuristic) | **0.300** | Expected: fixture has non-relevant chunks that score 0 |
| Context recall / keyword (heuristic) | **0.958** | 5/6 queries achieve full keyword coverage |
| Retrieve latency p50 | **~0.000045s** | In-memory TF-IDF; orders of magnitude faster than ChromaDB |
| Retrieve latency p95 | **~0.000065s** | — |

fix-001 nDCG < 1.0 is expected and correct: the TF-IDF retriever surfaces two chunks from `38300-g10.docx` in the top 5 (chunks 001 and 006), placing the `38401` chunk at rank 3 instead of rank 2, which is a suboptimal ranking for this two-spec query. The metric correctly penalizes it.

Context precision of 0.3 is expected: the fixture index contains 10 chunks from 10 different specs; for a query expecting 1-2 specs, 3-4 of the 5 retrieved chunks will typically come from non-matching specs. This is a characteristic of a small diverse fixture, not a production quality signal.

---

### 6.2 Full index baseline (2026-06-25) — pre-Track-B baseline (superseded)

All numbers below were produced on the full 43,121-chunk local/API index using the 30-query golden set (25 in-corpus + 5 out-of-corpus refusal probes), before Track B changed retrieval. Kept for history; the committed regression baseline is now the 2026-07-02 fusion baseline (§6.4).

**Retrieval — in-corpus (N=25):**

| Metric | Value | Note |
|---|---|---|
| Hit-rate@5 | **0.88** | Relevant source in top-5 for 88% of queries |
| nDCG@5 (graded) | **0.723** | "GOOD" band; primary retrieval signal |
| MRR | **0.679** | First relevant hit lands high on average |
| Recall@5 | **0.64** | Fraction of all expected source specs surfaced |
| Avg context precision (heuristic) | **0.44** | Strict source-filename match; supplementary |
| Avg context recall kw (heuristic) | **0.71** | Avg keyword coverage across top-3 |
| Legacy pass rate | **18/25 (72%)** | Composite: keyword recall ≥0.5 AND avg sim ≥0.50 |
| Avg cosine similarity | **0.557** | Mean of top-3 chunks per query |
| Retrieve p50 / p95 latency | **0.021s / 0.048s** | Apple M2 CPU |

**Refusal axis — out-of-corpus probes (N=5):**

| Metric | Value | Note |
|---|---|---|
| Retrieval-refusal rate | **4/5 (80%)** | Top-3 avg sim correctly below the 0.50 threshold |
| Avg out-of-corpus similarity | **0.331** | vs 0.557 in-corpus — measurably less confident on junk |
| LLM-judged refusal rate | **[RUN REQUIRED]** | Whether the generated answer declines — needs `--full` + live LLM |
| Faithfulness (LLM-judge) | **[RUN REQUIRED]** | Requires live LLM |
| Answer correctness (LLM-judge) | **[RUN REQUIRED]** | Requires live LLM |

**Reading the 88% hit-rate vs 72% legacy pass.** Seven in-corpus queries miss the legacy bar. Four of them actually retrieve the correct source (nDCG@5 = 1.0) and miss only because bge-small's cosine similarity sits just below the 0.50 threshold: gNB (0.426), HARQ (0.496), LTE X2 (0.500), SDAP (0.462, with full keyword recall). The other three are genuine retrieval misses (nDCG@5 = 0.0): E1 interface (gs-003), SA-vs-NSA (gs-007), and NR-vs-LTE physical layer (gs-025) — all comparison/interface queries where the question wording diverges from spec terminology.

---

### 6.3 Full + LLM-judge run (2026-07-02)

Produced by `python scripts/eval_retrieval.py --full --judge --judge-provider groq`. Responder: local Ollama `llama3.2` (2 GB, CPU). Judge: Groq `llama-3.3-70b-versatile` (independent of the responder). **Every retrieval metric reproduced the 2026-06-25 baseline exactly** — hit-rate@5 0.88, nDCG@5 0.723, MRR 0.679, Recall@5 0.64 — confirming run-to-run determinism of the retrieval layer.

| Metric | Value | Note |
|---|---|---|
| LLM-judge faithfulness (N=30) | **0.68** | Groq 70B judging llama3.2 answers; grounding credited, depth flagged |
| LLM-judge answer correctness (N=30) | **0.40** | Includes the 5 refusal probes, judged 0 by design; most common in-corpus note: "correct but lacks specific details" |
| Heuristic faithfulness | **0.834** | Above the 0.80 target |
| Heuristic answer pass rate | **0.96** | — |
| Composite answer score | **0.534** | Below 0.70 target; deflated by the `answer_keywords` harness gap (§2.1) |
| Answer-layer refusal (manual, N=5) | **5/5** | All probe answers explicitly decline; automated `is_refusal()` detector scored 0/5 — detector defect, see §3 |
| Generate latency p50 / p95 (llama3.2, CPU) | **44.3s / 69.8s** | Long RAG prompts on a CPU-only 2 GB model; the <5s end-to-end target is not met on the local path |
| Retrieve latency p50 / p95 (this run) | **0.022s / 0.056s** | Consistent with the clean-run baseline (0.021s / 0.048s) despite concurrent CPU load from generation |

**What these judge numbers evaluate:** the local deploy path. The public cloud app answers with Groq 70B, which was the judge here, not the responder — judging the cloud path with a distinct judge model is a named follow-up (`PHASE2.md` Track B).

**Run-to-run variance:** answer generation runs at temperature 0.1, so regenerated answers differ slightly between runs; four full runs produced judge faithfulness 0.647 / 0.68 / 0.647 / 0.64 and correctness 0.387 / 0.40 / 0.38 / 0.387 (the judge itself runs at temperature 0.0). Treat judge scores as directional with roughly ±0.03 noise; pinning generation temperature/seed for eval runs remains an open harness item.

---

### 6.4 Track B increment 1 (2026-07-03, morning) — calibrated threshold + query-expansion fusion

Produced by `python scripts/eval_retrieval.py --full --judge --judge-provider groq` after the first Track B retrieval changes: per-model calibrated pass threshold (0.42 for bge-small; `scripts/eval/calibrate_threshold.py`) and query-expansion rank fusion (ADR-009). Superseded as the current artifact by the increment-2 run (§6.5).

**Retrieval — in-corpus (N=25), before/after Track B:**

| Metric | Pre-Track-B (§6.2) | Replace-mode expansion (rejected) | Fusion (shipped) |
|---|---|---|---|
| Hit-rate@5 | 0.88 | 0.80 | **0.88** |
| Recall@5 | 0.64 | 0.57 | **0.683** |
| MRR | 0.679 | 0.598 | **0.697** |
| nDCG@5 (graded) | 0.723 | 0.641 | **0.718** |
| Legacy pass rate | 18/25 (72%) | — | **21/25 (84%)** (calibrated threshold) |
| E1 miss (gs-003) | nDCG 0.0 | fixed | **fixed** (hit-rate 1.0) |

Replace-mode expansion — substituting the expanded query for the raw one — was measured, found to regress previously-healthy queries, and rejected before fusion was built (full comparison in ADR-009). The two remaining genuine misses (SA-vs-NSA, NR-vs-LTE PHY) are comparison queries; they are the target of the M9 remainder (query decomposition).

**Answer + refusal axes (this run):**

| Metric | Value | Note |
|---|---|---|
| Answer-refusal rate (N=5, machine-scored) | **1.0 (5/5)** | Fixed detector + summary aggregation (Track B); previously manual-only |
| Answer relevance (heuristic) | **0.76** | Above target; first run able to score the golden set |
| Composite answer score | **0.80** | Above the 0.70 target |
| Heuristic faithfulness | **0.741** | Below the 0.80 target this run (0.825–0.834 in earlier runs); noisy surface heuristic |
| LLM-judge faithfulness (N=30) | **0.647** | Within the established ±0.03 band |
| LLM-judge answer correctness (N=30) | **0.38** | Within the established band; local-path responder ceiling |
| Generate latency p50 / p95 (llama3.2, CPU) | **37.5s / 56.5s** | <5s end-to-end target still not met on the local path |

---

### 6.5 Track B increment 2 (2026-07-03) — comparison decomposition + coverage cap (current)

Produced by `python scripts/eval_retrieval.py --full --judge --judge-provider groq` after the second Track B retrieval change: comparison-query decomposition with side-aware slot allocation, plus a per-source cap on fused rankings (ADR-010). This run is the current `data/eval_results.json`; its retrieval metrics match the committed 2026-07-03 baseline.

**Retrieval — in-corpus (N=25), across Track B:**

| Metric | Pre-Track-B | Increment 1 (fusion) | Increment 2 (decomposition) |
|---|---|---|---|
| Hit-rate@5 | 0.88 | 0.88 | **0.96** |
| Recall@5 | 0.64 | 0.683 | **0.737** |
| MRR | 0.679 | 0.697 | **0.720** |
| nDCG@5 (graded) | 0.723 | 0.718 | **0.755** |
| SA-vs-NSA (gs-007) | miss | miss | **recovered** (recall 1.0) |
| NR-vs-LTE PHY (gs-025) | miss | miss | **hit-rate 1.0**, recall 1/3 |

Global RRF over the decomposed sub-queries was measured first and left both comparison misses at zero — RRF's consensus bias buries per-side evidence — which is why sides get round-robin slot allocation instead (full comparison in ADR-010).

**Answer + refusal axes (this run):**

| Metric | Value | Note |
|---|---|---|
| Answer-refusal rate (N=5, machine-scored) | **1.0 (5/5)** | — |
| Answer relevance (heuristic) | **0.77** | Above target |
| Composite answer score | **0.794** | Above the 0.70 target |
| Heuristic faithfulness | **0.716** | Below the 0.80 target; noisy surface heuristic (0.716–0.834 across four runs) |
| LLM-judge faithfulness (N=30) | **0.64** | Within the ±0.03 band (four runs) |
| LLM-judge answer correctness (N=30) | **0.387** | Within the band; local-path responder ceiling |
| Generate latency p50 / p95 (llama3.2, CPU) | **35.5s / 60.5s** | <5s target still not met on the local path |

---

## 7. Interpretation and next steps

After both Track B increments, the honest picture:

- **Strong on lookup and now on comparisons:** hit-rate@5 0.96 and MRR 0.720 — every in-corpus query except one surfaces at least one relevant source in the top-5.
- **Multi-evidence recall at 0.737 against the 0.80 M9 target.** The measured residual is an embedding-resolution limit, not query mechanics: "NR physical layer" cannot surface TS 38.211 in its own top-12 — bge-small ranks sibling PHY specs (38.212–38.215) above the labeled definitional spec. The named levers are the bge-base upgrade ADR-003 already designated (full index rebuild) and graded-relevance review of whether sibling specs deserve partial credit.
- **The threshold artifact is fixed:** the calibrated 0.42 pass threshold (bge-small) recovered the four correctly-retrieving queries the old 0.50 cutoff failed.
- **The vocabulary/comparison miss cluster is closed at the hit-rate level:** E1 via expansion fusion (ADR-009), SA-vs-NSA fully and NR-vs-LTE partially via decomposition (ADR-010).
- **Refusal behaviour is machine-verified at both layers:** 4/5 probes stay below the retrieval threshold; answer-refusal rate 1.0 (5/5), aggregated automatically.
- **The local answer path grounds but lacks depth:** judge faithfulness ~0.65 vs correctness ~0.39 across four runs quantifies the 2 GB responder's ceiling. The cloud path's 70B responder is not yet judged.

**Priority improvements (tracked in `PHASE2.md`):**
1. ~~Recalibrate the per-embedding-model similarity threshold~~ **Done** — calibrated 0.42 (bge-small), derivation in `scripts/eval/calibrate_threshold.py`.
2. ~~Query-side vocabulary expansion~~ **Done via rank fusion (ADR-009)** — E1 recovered; replace-mode measured and rejected.
3. ~~Query decomposition for comparison questions~~ **Done via side-aware allocation (ADR-010)** — SA-vs-NSA recovered; global-RRF merge measured and rejected.
4. Close the Recall@5 gap to ≥ 0.80: bge-base rebuild experiment (ADR-003's designated upgrade; requires full re-index) and graded-relevance label review for sibling-spec credit. (M9 remainder)
5. ~~Fix the refusal detector, wire aggregation, enable golden-set answer-relevance~~ **Done** — see §3 and §6.5.
6. Judge the cloud answer path (Groq 70B responder) with a distinct judge model. (Track B remainder)
7. Expand graded relevance labels beyond the current 7 in-corpus queries; pin generation temperature/seed for reproducible judge runs. (Track B remainder)

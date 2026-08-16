# Phase 2 Plan — Telecom Standards AI Assistant

**Status:** Approved plan — tracks not yet started except Track A
**Date:** 2026-07-02
**Owner:** Eric Costa
**Related:** [`ROADMAP.md`](ROADMAP.md) (milestones M9–M11) · [`EVAL_REPORT.md`](EVAL_REPORT.md) · [`DECISIONS.md`](DECISIONS.md)

---

## 1. Where Phase 1 landed

Phase 1 shipped a public, dual-deploy RAG system over 37 3GPP specifications (43,121 chunks) with a reproducible eval harness, a committed regression baseline, a STRIDE threat model, and a corpus-agnostic configuration seam. Headline retrieval numbers (2026-06-25 full-index run, 25 in-corpus queries): hit-rate@5 0.88, nDCG@5 0.723, MRR 0.679, Recall@5 0.64. Full context in [`METRICS.md`](METRICS.md).

## 2. What the Phase 1 eval says to build next

Phase 2 is scoped from measured findings, not feature ideas:

| # | Finding (source: `data/eval_results.json`, EVAL_REPORT §6–7) | Implication |
|---|---|---|
| F1 | Recall@5 is 0.64 — several two-spec queries surface only one of the two expected specs | Multi-evidence retrieval needs query decomposition or coverage-aware ranking. *Both shipped (2026-07-03, ADR-010): Recall@5 at 0.737; the residual to 0.80 is a measured embedding-resolution limit (see Track B status)* |
| F2 | The three genuine retrieval misses (E1 interface, SA-vs-NSA, NR-vs-LTE PHY) are all queries whose phrasing diverges from spec vocabulary | Query-side expansion from a domain glossary should recover them. *Closed at the hit-rate level (2026-07-03): E1 via expansion fusion (ADR-009); SA-vs-NSA fully and NR-vs-LTE to hit-rate 1.0 via decomposition (ADR-010)* |
| F3 | Four legacy-pass "failures" retrieved the correct source but scored under the fixed 0.50 cosine threshold | The pass threshold needs per-embedding-model calibration. *Closed (2026-07-03): calibrated 0.42 for bge-small via `scripts/eval/calibrate_threshold.py`; all four pass* |
| F4 | LLM-judge faithfulness / answer-correctness / refusal were `[RUN REQUIRED]` | Closed by Track A (run 2026-07-02: faithfulness 0.68, correctness 0.40, local path). The run surfaced two harness defects, deferred to Track B |

## 3. Goals and non-goals

**Goals**

1. Close every Phase 1 `[RUN REQUIRED]` that does not depend on new infrastructure (Track A).
2. Raise multi-evidence retrieval quality on the measured gaps, without regressing single-spec quality (Track B → M9).
3. Ship a release-delta capability: answer "what changed between releases?" with citable Change Requests (Track C → M10).
4. Validate the corpus-agnostic claim on a real second corpus (Track D → M11).

**Non-goals for Phase 2**

- User counts, adoption, or growth targets — not how this project measures success.
- Authentication / multi-user hosting (stays M7).
- Distributed rate limiting (revisited with M7).
- UI redesign beyond what the release filter requires.

## 4. Tracks

### Track A — Phase 1 close-out (completes M4)

Run the LLM-judge evaluation (answers via the local Ollama path, judged independently by Groq `llama-3.3-70b-versatile` — the independent-judge setup EVAL_REPORT recommends), publish the numbers, reconcile all documentation to one milestone numbering, and record the demo GIF per [`assets/DEMO_CAPTURE.md`](assets/DEMO_CAPTURE.md).

**Status (2026-07-02):** judge run complete and published (EVAL_REPORT §6.3); docs reconciled. Remaining: the demo GIF recording. Two harness defects the run surfaced (refusal detector, missing `answer_keywords` field) are explicitly deferred to Track B, as is judging the cloud answer path.

**Done when:** judge metrics appear in `data/eval_results.json` and the docs; no `[RUN REQUIRED]` remains except items explicitly deferred to a named milestone; README/ROADMAP milestone tables match.

### Track B — Retrieval quality v2 (M9)

**Problem.** F1–F3 above.

**Deliverables**

1. Per-embedding-model similarity-threshold calibration (replaces the fixed 0.50 cutoff; calibrated against the golden set, stored alongside the model choice in config).
2. Query-time vocabulary expansion built from 3GPP TR 21.905 (the official abbreviations/vocabulary spec) — acronym and synonym mapping applied before embedding.
3. Query decomposition for multi-spec questions: detect comparison/interface phrasing, issue sub-queries, merge with coverage-aware ranking.
4. Graded relevance labels extended beyond the current 7 queries.
5. Harness fixes surfaced by the 2026-07-02 judge run: add `answer_keywords` to the golden set (answer-relevance currently cannot score it), widen or replace the `is_refusal()` phrase detector (it missed 5/5 actual refusals), wire automated refusal-rate aggregation into the summary, and make `--regression` non-destructive by default (today it overwrites `data/eval_results.json` unless `--no-save` is passed — the CI gate now passes `--no-save` as a workaround).
6. Judge the cloud answer path (Groq 70B as responder) with a distinct judge model, so both deploy paths carry published answer-quality numbers.

**Success metrics (targets, not results):** Recall@5 ≥ 0.80 in-corpus; the E1 / SA-vs-NSA / NR-vs-LTE-PHY queries pass; hit-rate@5 ≥ 0.88 maintained; `--regression` gate green throughout.

**Status (2026-07-03, second increment):** items 1, 2, 3, and 5 shipped. Increment 1: calibrated threshold (0.42 for bge-small), query-expansion rank fusion (ADR-009; replace-mode was measured to regress and rejected), and all four harness fixes. Increment 2: comparison-query decomposition with side-aware slot allocation (ADR-010 — global RRF was measured to bury per-side evidence and replaced with round-robin allocation across per-source-deduped sides). Measured after both increments: **hit-rate@5 0.96 (from 0.88), Recall@5 0.737 (from 0.64), MRR 0.720, nDCG@5 0.755**; SA-vs-NSA fully recovered; NR-vs-LTE reaches hit-rate 1.0; answer-refusal 5/5 machine-scored; gate green.

**The honest residual on the 0.80 Recall@5 target:** direct probing shows "NR physical layer" cannot surface TS 38.211 in its own top-12 — the embedding model ranks sibling PHY specs above the labeled definitional spec. This is an embedding-resolution limit, not query mechanics; the named next levers are the bge-base upgrade already designated in ADR-003 (full index rebuild required) and graded-relevance label review (item 4, needs human review of whether sibling specs deserve partial credit). Item 6 (cloud-path judging) also remains.

### Track C — Release intelligence (M10)

**Hypothesis (to validate before building).** H1: engineers assessing a new 3GPP release need "what changed for X between Rel-N and Rel-N+1" answers with citable Change Requests, and today assemble this manually from change logs. Validation: a handful of structured conversations with practicing RAN/Core engineers against a clickable mock, before implementation starts.

**Deliverables**

1. Version-aware indexing: retain multiple releases per spec with `release` and `version` chunk metadata. This deliberately supersedes ADR-007 (dedup-to-latest) — a new ADR records the reversal and its cost.
2. Change Request ingestion: CR metadata (number, target release, subject, status) from the 3GPP portal exports, plus the `-cl` change-log companion files the Phase 1 indexer deliberately skipped.
3. Release filter in retrieval + a delta-answer prompt path that cites spec version and CR number.
4. A release-delta axis in the golden set with its own pass criteria.

**Success metrics:** delta queries on the new golden-set axis answered with the correct release attribution and a verifiable CR citation; no regression on the existing 25-query in-corpus axis.

**Status (2026-07-03):** ungated groundwork landed — `infer_release_from_filename()` in `spec_catalog.py` parses the release from archive filenames (h30 → Rel-17) and the index builder now stores `release` chunk metadata (effective at the next rebuild), and ADR-008 is drafted as **Proposed** in DECISIONS.md. The build itself remains gated on H1 validation conversations with practicing engineers.

### Track D — Second corpus validated (M11)

**Problem.** [`GENERALIZE.md`](GENERALIZE.md) honestly labels generalization "designed, not validated," and names the two remaining seam gaps (the processor's inline cleaner; the retriever's hardcoded filter params).

**Deliverables**

1. Close both named seam gaps so `CorpusConfig` fully parameterizes cleaning and filtering.
2. Index NIST SP 800-53 rev 5: catalog from NIST's machine-readable OSCAL JSON, prose from the official publication (public domain — zero licensing friction).
3. A small (≈10-query) golden set for the second corpus and a corpus-switching example in GENERALIZE.md.

**Success metrics:** end-to-end index + query on the NIST corpus through `CorpusConfig` with no pipeline forks; 3GPP suite and regression gate unaffected; GENERALIZE.md claim upgraded from "designed to generalize" to "validated on a second corpus."

## 5. Standardization data required

| Dataset | Source | Used by | Notes |
|---|---|---|---|
| Multi-version spec archives (Rel-15 → Rel-19) | 3GPP FTP `/Specs/archive/<series>/` — all historical versions retained | Track C | Filename version letters encode the release (f=Rel-15, g=16, h=17, i=18, j=19); existing filename parsing extends naturally |
| Change Request database | portal.3gpp.org exports; `-cl` change-log files inside spec zips | Track C | Structured metadata (CR number, release, subject, status) — indexed as metadata-rich chunks |
| TR 21.905 — Vocabulary for 3GPP Specifications | 3GPP, freely downloadable | Track B | The official acronym/term source for query expansion; fixes trace directly to eval finding F2 |
| 3GPP Work Plan / spec status | 3gpp.org (spreadsheet) | Track C | Release dates and status as chunk metadata; lets the UI label "Rel-18, frozen" |
| NIST SP 800-53 rev 5 | NIST — public domain; OSCAL JSON catalog + official PDF | Track D | OSCAL provides the document catalog machine-readably; no scraping, no licensing risk |

**Licensing and compliance.** 3GPP/ETSI specification text is copyrighted; the Phase 1 posture (source-excerpt truncation in the UI, legal disclaimer, no full-text redistribution) carries forward unchanged and applies to all newly indexed versions. CR metadata is factual data. NIST publications are US-government works in the public domain.

## 6. Sequencing and dependencies

```
Track A ──► Track B ──► Track C
                 (C inherits B's retrieval quality;
                  delta answers are only as good as retrieval)
Track D ─────────────────► (independent; can run parallel to B/C)
```

Every track lands behind the existing `--regression` gate; a track that degrades the committed baseline does not merge.

## 7. Risks

| Risk | Mitigation |
|---|---|
| Version-aware index grows the corpus several-fold (storage, latency) | Start with 2–3 releases for a subset of flagship specs (38.300, 38.331, 23.501); measure before expanding |
| CR export formats vary by spec family | Pilot with two spec families before generalizing the parser |
| Query expansion hurts precision on queries that were already passing | Expansion is applied behind the regression gate; tolerance thresholds already defined in `data/eval/baseline.json` |
| OSCAL catalog structure differs from prose-document corpora | Ingest the official PDF prose alongside the OSCAL catalog so the second-corpus test exercises the same PDF pipeline as 3GPP |
| H1 (release-delta need) fails validation | Track C is gated on the validation conversations; if the hypothesis fails, M10 is re-scoped before any indexing work starts |

## 8. Decision records this phase will add

- **ADR-008** — Version-aware indexing (supersedes ADR-007's dedup-to-latest; records the storage/consistency tradeoff).
- **ADR-009** — Query expansion source and placement (TR 21.905 glossary at query time vs. index-time enrichment).
- **ADR-010** — Second-corpus selection (NIST SP 800-53 via OSCAL; alternatives considered: O-RAN, ETSI NFV — both carry more licensing friction).

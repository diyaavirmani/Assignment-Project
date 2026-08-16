# Product Roadmap — 3GPP RAG Assistant

Milestones are defined by the outcome they unlock, the metric that confirms them, and the architectural decision each one either closes or defers. This is not a sprint calendar — phases are complete when their success criteria are met, not when a date arrives.

---

## Completed milestones

### M1: Working RAG over a single spec
**Outcome:** An engineer can ask a natural-language question about one 3GPP spec and get a cited, grounded answer.  
**Success criteria met:** End-to-end pipeline functional (retrieve → prompt → generate); answer includes source citation; multi-turn conversation works; CLI and Streamlit UI available.  
**Key decisions closed:** RAG over fine-tuning (ADR-001); bge-small embedding choice (ADR-003); chunk size 1000/200 (ADR-004).  
**What it proved:** The chunking + retrieval pipeline works for dense technical text. Semantic search surfaces relevant clauses that keyword search misses.

---

### M2: Production-ready multi-spec coverage
**Outcome:** The same engineer can query across 37 specs spanning 5G NR, LTE, and Core — and filter by domain or generation — without touching config files.  
**Success criteria met:** 43,121 chunks indexed from 37 specs; domain/generation filtering in UI and API; dedup-to-latest-version indexing prevents conflicting results across releases; `/catalog` endpoint shows indexed status per spec.  
**Key decisions closed:** Dedup to latest version (ADR-007); metadata-driven filtering via ChromaDB `where` clauses.  
**What it proved:** The indexing and retrieval architecture scales to a real corpus without degrading query quality.

---

### M3: Public deploy, cloud path, security hardening
**Outcome:** Anyone with a browser can use the assistant without installing anything, and the app is safe to expose publicly.  
**Success criteria met:** Live Streamlit Cloud app running Groq llama-3.3-70b; pre-built vectordb distributed via GitHub Release asset; 42 vulnerabilities from formal security audit resolved across two hardening rounds [source: owner's audit notes]; rate limiting, SSRF prevention, path-traversal protection, prompt injection defence, security headers, input bounds all in code.  
**Key decisions closed:** Dual-provider LLM architecture (ADR-002); ChromaDB ONNX on cloud (ADR-005); security-before-public-launch (ADR-006).  
**What it proved:** A side project can ship with production-grade security posture. Most personal AI tools skip this entirely; this one didn't.

---

## Completed — Phase 1 close-out

### M4: Eval rigor — retrieval + answer quality
**Outcome:** Any reviewer can reproduce the eval in one command and trust the numbers.  
**Success criteria:**
- Recall@k, MRR, nDCG@k, and hit-rate added to the retrieval harness alongside the existing precision/recall heuristics
- A versioned golden dataset (`data/eval/golden_set.jsonl`) with ground-truth sources and expected facts — real and checkable against the corpus
- LLM-judge faithfulness and answer-correctness path (configurable Groq or Ollama judge)
- `--regression` mode: non-zero exit if any metric falls below the stored baseline
- `EVAL_REPORT.md` documenting methodology, limitations, current numbers (2026-06-25 and 2026-07-02 full-index runs), and `[RUN REQUIRED]` placeholders for anything not yet measured

**Metric that confirms it:** Retrieval pass rate and Recall@5 reproducible with a single command; LLM-judge faithfulness run and published.  
**Status:** Complete. The LLM-judge run landed 2026-07-02 (faithfulness 0.68, answer correctness 0.40 on the local answer path; see `EVAL_REPORT.md` §6.3). The run surfaced two harness defects — the refusal detector and a missing golden-set field — deferred to Phase 2 Track B (`PHASE2.md`).

---

### M5: Corpus-agnostic architecture
**Outcome:** A developer can point this system at any technical or regulated document corpus by swapping a config — not by forking the codebase.  
**Success criteria:**
- A documented seam for corpus configuration (loader interface or `CorpusConfig` dataclass) so 3GPP-specific logic is isolated and swappable
- `GENERALIZE.md` with a worked example: how to replace the 3GPP catalog and cleaner with a different corpus
- The full mocked test suite stays green; 3GPP remains the default and only implemented corpus
- No fabricated "supports X corpora" claims — framed honestly as "designed to generalize; 3GPP is the implemented corpus"

**Metric that confirms it:** A reader can trace in code exactly where the 3GPP-specific parts plug in; no behavior change on the 3GPP path.  
**Status:** Complete. The seam is live in `src/core/corpus_config.py` and consumed by `scripts/build_index.py`; validation against a real second corpus is Phase 2 work (M11, `docs/PHASE2.md`).

---

## Planned

### M6: Multi-document reasoning across spec boundaries
**Outcome:** A question that spans two or more specs (e.g., "How does the F1 interface interact with the NGAP protocol?") gets a single coherent answer that correctly integrates content from TS 38.473 and TS 38.413, rather than two disconnected responses.  
**Why it's hard:** Current retrieval returns top-k chunks without regard for cross-spec dependencies. Multi-hop reasoning requires either a re-ranking step that promotes cross-spec coverage, or an agentic loop that issues sub-queries per spec and merges answers.  
**Decision to make:** Re-ranker vs agentic retrieval vs query decomposition. All three are architecturally viable; the choice depends on latency tolerance and whether the user can see intermediate reasoning steps (which changes the UX model).  
**Success criteria:** Pass rate on a cross-spec golden set ≥ single-spec pass rate; no regression on single-spec queries.

---

### M7: Hosted team version with auth and shared sessions
**Outcome:** A telecom team (e.g., an RAN engineering group) can use the assistant as a shared tool — with login, per-user session history, and an admin view — rather than as a public anonymous app.  
**Why it matters:** The current public app has no user context. A team version enables usage analytics (which specs are queried most?), per-engineer session continuity, and access control for sensitive configurations.  
**Decision to make:** Auth provider (OAuth via GitHub/Google vs enterprise SSO); session storage (in-process dict vs Redis); hosting (Streamlit Cloud doesn't support server-side auth well — likely moves to a container deploy).  
**Success criteria:** Authenticated users can persist session history across logins; admin can see aggregate query patterns; rate limiting migrates from in-process (current) to distributed.

---

### M8: Eval-gated CI and regression prevention
**Outcome:** A pull request that degrades retrieval quality fails CI before it merges.  
**Why it matters:** Currently the eval harness is a manual script. As the codebase evolves, a chunking change or a config update could silently degrade retrieval pass rate. The `--regression` mode (planned in M4) is the foundation; this milestone wires it into `.github/workflows/ci.yml`.  
**Decision to make:** Which metrics gate the build (pass rate only? recall@5? faithfulness?); what the regression threshold is; whether the eval runs on a small fast subset or the full 43K-chunk index (runtime cost in CI).  
**Success criteria:** CI fails on a deliberate regression (test); eval result is committed to the repo after each run so the baseline stays current.

---

### M9: Retrieval quality v2 (Phase 2, Track B)
**Outcome:** The measured retrieval gaps from the Phase 1 eval are closed: multi-evidence queries surface all expected specs, and vocabulary-divergent queries stop missing.  
**Success criteria:** Recall@5 from 0.64 to ≥ 0.80 in-corpus; the E1 / SA-vs-NSA / NR-vs-LTE-PHY miss cluster passes; hit-rate@5 ≥ 0.88 maintained; regression gate green.  
**Plan:** [`PHASE2.md`](PHASE2.md) §Track B — per-model threshold calibration, TR 21.905 vocabulary expansion, multi-spec query decomposition.  
**Status:** In progress (2026-07-03, two increments shipped). Increment 1: calibrated pass threshold (0.42 for bge-small) + query-expansion rank fusion (ADR-009 — replace-mode measured to regress and rejected). Increment 2: comparison decomposition with side-aware slot allocation + per-source coverage cap (ADR-010 — global-RRF merge measured to bury per-side evidence and rejected). Measured: hit-rate@5 0.96, Recall@5 0.737, MRR 0.720, nDCG@5 0.755; E1 and SA-vs-NSA recovered; gate green. Remaining: the recall ≥ 0.80 target — a measured embedding-resolution limit (bge-small cannot surface TS 38.211 for "NR physical layer"); levers are the bge-base rebuild (ADR-003's designated upgrade) and graded-relevance label review.

---

### M10: Release intelligence (Phase 2, Track C)
**Outcome:** An engineer can ask "what changed for X between Rel-17 and Rel-18?" and get an answer with release attribution and a citable Change Request.  
**Success criteria:** A release-delta golden-set axis passes with correct release attribution and verifiable CR citations; no regression on the existing in-corpus axis.  
**Plan:** [`PHASE2.md`](PHASE2.md) §Track C — version-aware indexing (supersedes ADR-007), CR ingestion, release-filtered retrieval. Gated on hypothesis validation with practicing engineers.  
**Status:** Groundwork landed 2026-07-03 (release parsed from filenames into chunk metadata at index time; ADR-008 drafted as Proposed). Build gated on H1 validation.

---

### M11: Second corpus validated (Phase 2, Track D)
**Outcome:** The corpus-agnostic claim is proven, not just designed: a real non-3GPP corpus runs end to end through `CorpusConfig`.  
**Success criteria:** NIST SP 800-53 rev 5 indexed and queryable with no pipeline forks; the two seam gaps named in GENERALIZE.md closed; 3GPP suite unaffected.  
**Plan:** [`PHASE2.md`](PHASE2.md) §Track D.

---

## What this roadmap does not include

- User counts, adoption numbers, or revenue. This is an open-source personal project. Usage metrics are not the measure of success for this tool — eval quality, security posture, and architectural clarity are.
- A "launch date." Milestones ship when their success criteria are met.
- Features without a defined success metric. If it can't be measured, it belongs in a speculative backlog, not a roadmap.

---

## README embed block

The following condensed block is embedded in README.md; keep the two in sync:

```markdown
## Roadmap

Milestones are defined by outcome and success criteria, not sprint dates. Full milestone details: [`docs/ROADMAP.md`](docs/ROADMAP.md) · Phase 2 plan: [`docs/PHASE2.md`](docs/PHASE2.md)

| Milestone | Status | Success criteria |
|---|---|---|
| M1: Working RAG over a single spec | Complete | End-to-end pipeline; cited answers; multi-turn UI |
| M2: Multi-spec coverage (37 specs, domain/generation filtering) | Complete | 43,121 chunks indexed; metadata filtering in UI + API |
| M3: Public deploy, dual LLM path, security hardening | Complete | Live Streamlit Cloud app; audited vulnerabilities resolved |
| M4: Eval rigor (Recall@k, MRR, nDCG, LLM-judge) | Complete | Reproducible one-command harness; golden dataset; regression gate; LLM-judge run published |
| M5: Corpus-agnostic architecture | Complete | Documented seam in code; GENERALIZE.md; full suite green |
| M6: Multi-document reasoning across spec boundaries | Planned | Cross-spec golden-set pass rate ≥ single-spec baseline |
| M7: Hosted team version with auth + shared sessions | Planned | Auth, per-user history, distributed rate limiting |
| M8: Eval-gated CI | Planned | A regression in retrieval quality fails CI automatically |
| M9: Retrieval quality v2 — Phase 2, Track B | In progress | Hit-rate@5 0.96, Recall@5 0.737 (from 0.88 / 0.64) via calibrated threshold, expansion fusion, comparison decomposition; remaining: bge-base rebuild + graded labels to reach recall ≥ 0.80 |
| M10: Release intelligence — Phase 2, Track C | Planned | Version-aware index; release-delta answers citing Change Requests |
| M11: Second corpus validated — Phase 2, Track D | Planned | NIST SP 800-53 indexed and queryable through CorpusConfig |
```

# Telecom Standards AI Assistant

A production RAG system — the **3GPP RAG Assistant** — over 37 3GPP technical specifications: 43,121 chunks, dual-deploy (local Ollama or cloud Groq), citation-grounded answers for engineers who spend hours lost in 1,000-page regulated specs.

**Live demo:** [3gpp-rag-assistant.streamlit.app](https://3gpp-rag-assistant.streamlit.app/) — animated walkthrough coming soon (see [docs/assets/DEMO_CAPTURE.md](docs/assets/DEMO_CAPTURE.md))
<!-- demo GIF — drop the recording at docs/assets/demo.gif and replace the line above with: ![Demo](docs/assets/demo.gif) -->

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Tests](https://img.shields.io/badge/tests-302%20passing-brightgreen.svg)](#testing)
[![Specs](https://img.shields.io/badge/3GPP%20specs-37%20indexed-blue.svg)](#architecture)
[![Chunks](https://img.shields.io/badge/chunks-43%2C121-blue.svg)](#architecture)

---

## Table of Contents

1. [What this is](#what-this-is)
2. [Evaluation](#evaluation)
3. [Security and regulated-domain readiness](#security-and-regulated-domain-readiness)
4. [Architecture](#architecture)
5. [Generalize to your corpus](#generalize-to-your-corpus)
6. [Quick start](#quick-start)
7. [Configuration](#configuration)
8. [Project structure](#project-structure)
9. [Testing](#testing)
10. [Roadmap](#roadmap)
11. [Deployment details](#deployment-details)
12. [Contributing](#contributing)
13. [License](#license)
14. [Contact](#contact)

---

## What this is

3GPP specifications are the authoritative source for all cellular standards (4G LTE, 5G NR). They are also notoriously hard to navigate: documents run 500–2,000 pages each, answers routinely span two or three specs, and there is no semantic search — only Ctrl+F inside a PDF.

This system closes that gap. An engineer types a question in plain English; the assistant retrieves the relevant spec excerpts, grounds the answer in the source text, and cites the document and section. Every answer is traceable.

**Who uses it:** telecom engineers, standards researchers, and engineers onboarding to 5G. A few colleagues at Rogers began using the public deploy voluntarily — it is a personal side project, not a Rogers program.

**Beyond the tool itself,** this repository documents production RAG practice on a dense, regulated corpus — dual-deploy architecture, an eval harness with reproducible metrics, security controls mapped to a threat model, and a corpus-agnostic design built to generalize beyond 3GPP.

---

## Evaluation

> Eval report with full methodology: [`docs/EVAL_REPORT.md`](docs/EVAL_REPORT.md)
> Reproduce: `python scripts/eval_retrieval.py --output data/eval_results.json`

Current results from `data/eval_results.json` (run 2026-07-03 with `--full --judge`, full index of 43,121 chunks, 30-query golden set, top-k=5; retrieval uses the Track B calibrated threshold, query-expansion rank fusion, and comparison decomposition, and matches the committed 2026-07-03 baseline). The golden set splits into 25 answerable (in-corpus) queries and 5 deliberately out-of-corpus probes; retrieval metrics are reported over the 25, the refusal axis over the 5.

**Retrieval quality — in-corpus (N=25), standard IR metrics:**

| Metric | Value | Notes |
|---|---|---|
| Hit-rate@5 | 0.96 | a relevant source appears in top-5 for 96% of queries (0.88 pre-Track-B) |
| nDCG@5 | 0.755 | rank-weighted; "GOOD" band (0.723 pre-Track-B) |
| MRR | 0.720 | first relevant hit lands high on average (0.679 pre-Track-B) |
| Recall@5 | 0.737 | fraction of all expected source specs surfaced (0.64 pre-Track-B; target ≥ 0.80) |
| Avg context precision | 0.456 | heuristic (keyword); supplementary, not IR-standard |
| Avg context recall | 0.75 | heuristic keyword coverage across top-3 |
| Legacy pass rate | 21/25 (84%) | keyword recall ≥0.5 AND avg sim ≥ calibrated threshold (0.42 for bge-small) |
| Retrieve latency p50 / p95 | 0.069s / 0.213s | Apple M2 CPU; expanded queries run 2 searches, comparisons up to 6 |

**Refusal axis — out-of-corpus probes (N=5):** 4/5 (80%) correctly stay below the relevance threshold at the retrieval layer; avg out-of-corpus similarity 0.331 vs 0.562 in-corpus. At the answer layer, **5/5 generated answers explicitly decline — now machine-scored** (`answer_refusal_rate` 1.0; the original detector had scored 0/5 on the same behavior, and Track B fixed it using the real refusals as regression fixtures).

**LLM-judge — answer quality (2026-07-03, N=30):** faithfulness **0.64**, answer correctness **0.387** (stable across four runs: 0.647/0.68/0.647/0.64 and 0.387/0.40/0.38/0.387) — local Ollama `llama3.2` answers judged independently by Groq `llama-3.3-70b-versatile`. The judge's most common note is "correct but lacks specific details": the small local responder grounds its answers but runs thin on depth. These scores evaluate the local path; judging the cloud path (70B responder) is a named follow-up.

**What Track B changed, by measurement.** The Phase 1 eval found two retrieval defect classes, and both fixes were chosen by measuring and rejecting the obvious approach first. Threshold artifacts were fixed by calibrating the pass threshold per embedding model (`scripts/eval/calibrate_threshold.py`). Vocabulary-divergence misses: replacing the query with its expanded form was **measured to regress** hit-rate 0.88 → 0.80 and rejected; rank-fusing raw and expanded rankings shipped instead (ADR-009), recovering the E1 miss. Comparison misses: merging decomposed sub-queries into the global RRF was **measured to do nothing** (consensus bias buries per-side evidence) and rejected; side-aware slot allocation shipped instead (ADR-010), recovering SA-vs-NSA fully and NR-vs-LTE to hit-rate 1.0. Net: hit-rate 0.88 → 0.96, Recall@5 0.64 → 0.737. The residual against the 0.80 recall target is a measured embedding-resolution limit (bge-small cannot surface TS 38.211 for "NR physical layer"); the named lever is the bge-base upgrade from ADR-003 (see [roadmap](#roadmap)).

**Heuristic vs IR-standard.** Context precision/recall are keyword-based heuristics (RAGAS-inspired), kept as supplementary signals. Hit-rate/Recall@k/MRR/nDCG are the standard IR metrics; LLM-judge faithfulness and answer-correctness are measured (see [`docs/EVAL_REPORT.md`](docs/EVAL_REPORT.md) §6.3–6.4). **Still open:** query decomposition for comparison questions, and judging the cloud answer path — tracked in [`docs/PHASE2.md`](docs/PHASE2.md).

---

## Security and regulated-domain readiness

> Disclosure policy and reporting: [`SECURITY.md`](SECURITY.md)
> Threat model with code-level control mapping: [`docs/THREAT_MODEL.md`](docs/THREAT_MODEL.md)

The live deploy is public-facing. The following controls are in the codebase — the threat model links each to its file and line:

| Threat | Control | Location |
|---|---|---|
| DoS / rate abuse | Per-IP rate limiting (LRU store, bounded size) | `src/api/main.py` |
| SSRF via Ollama URL | Allowlist validator (localhost / docker hosts only) | `src/config.py` |
| SSRF via vectordb download | Domain allowlist + redirect limit | `streamlit_app.py` |
| Path traversal (tar extract) | Safe-extract guard, rejects `../` members | `streamlit_app.py` |
| Prompt injection / output XSS | HTML-escaping of source text before render | `streamlit_app.py` |
| Oversized / malformed input | Pydantic bounds (min/max length, Literal enums) | `src/api/models.py` |
| Clickjacking / MIME sniffing | Security headers (X-Frame-Options, X-Content-Type-Options, CSP) | `src/api/main.py` |

A personal-project security audit surfaced 42 issues resolved across two hardening rounds. The individual findings are not reproduced here; the threat model maps the control categories that are verifiable in the repository.

---

## Architecture

The system has two deploy modes that share the same retrieval stack but differ in the LLM and embedding path:

```
                        User query
                            |
            ┌───────────────┴───────────────┐
            │           RAG Chain           │
            │       src/core/rag_chain.py   │
            │  multi-turn history, prompt,  │
            │  streaming                    │
            └──────────┬────────────────────┘
                       |
          ┌────────────┴────────────┐
          |                         |
   ┌──────▼───────┐         ┌───────▼────────────────────────┐
   │   Retriever  │         │           LLM                  │
   │  top-k chunk │         │                                │
   │  search with │         │  LOCAL: Ollama (llama3.2,      │
   │  domain /    │         │  mistral, phi3) — no API key   │
   │  generation  │         │                                │
   │  filters     │         │  CLOUD: Groq llama-3.3-70b-    │
   └──────┬───────┘         │  versatile — GROQ_API_KEY      │
          |                 │  required                      │
   ┌──────▼───────┐         └────────────────────────────────┘
   │  Vector Store│
   │  ChromaDB    │
   │              │
   │  LOCAL: full │  sentence-transformers bge-small-en-v1.5
   │  persist DB  │  (local, no API key)
   │              │
   │  CLOUD: ONNX │  ChromaDB built-in ONNX runtime
   │  embeddings, │  (no torch), DB pulled from
   │  DB from     │  GitHub Release asset at startup
   │  GH Release  │
   └──────┬───────┘
          |
   ┌──────▼───────────────────────────────────────────────┐
   │  Document Processor  +  Spec Catalog                  │
   │  37 specs × (5G/LTE) × (RAN/CORE)                    │
   │  PDF / DOCX / DOC → 1000-char chunks, 200 overlap     │
   │  3GPP-specific cleaning, metadata per chunk           │
   └──────────────────────────────────────────────────────┘
```

**Key decisions** (full ADR log: [`docs/DECISIONS.md`](docs/DECISIONS.md)):

- **RAG over fine-tuning.** 3GPP specs update every release cycle; RAG re-indexes without retraining. Citations are non-negotiable for standards work.
- **bge-small-en-v1.5.** Best accuracy-to-size ratio on technical/scientific text (MTEB). 130MB, 384-dim, runs on CPU.
- **Chunk size 1000 / overlap 200.** Large enough to preserve context around a protocol definition; overlap prevents boundary splits from losing cross-sentence references.
- **Ollama (local) + Groq (cloud) as a toggle.** `src/config.py` `llm_provider` field selects the path. The cloud deploy uses Groq because Streamlit Cloud has no GPU; Ollama is the local-dev default. Both paths are real.
- **ChromaDB ONNX on cloud.** Avoids a PyTorch dependency on the Streamlit Cloud runtime. The pre-built vectordb is attached to a GitHub Release and downloaded at app startup.
- **Query expansion via rank fusion.** Queries containing known 3GPP abbreviations are also searched with their TR 21.905 full forms and the two rankings are RRF-merged — replace-mode expansion was measured to regress and rejected (ADR-009). Both deploy paths share one embedding-agnostic fusion module (`src/core/retrieval_fusion.py`, ADR-011); the measured gains are bge-small numbers (the cloud ONNX space is unevaluated).

---

## Generalize to your corpus

> Full guide: [`docs/GENERALIZE.md`](docs/GENERALIZE.md)

3GPP is the reference corpus; the architecture is not 3GPP-specific. The retrieval stack (document processor, ChromaDB, filtered retrieval, RAG chain) works on any technical or regulated document set. 3GPP-specific logic is contained in `src/core/spec_catalog.py` (the spec registry) and the 3GPP-aware text cleaner in `document_processor.py`.

To point the system at a different corpus — medical device standards, financial regulations, internal engineering wikis — you replace the catalog and cleaning logic. GENERALIZE.md documents the seam: what to implement, what to leave unchanged, and the 3GPP catalog as a worked example.

---

## Quick start

### Option A — Local (Ollama, no API key needed)

```bash
# 1. Clone and install
git clone https://github.com/ericnc09/3gpp-rag-assistant.git
cd 3gpp-rag-assistant
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env

# 2. Install Ollama and pull a model
brew install ollama          # or: https://ollama.com for Windows/Linux
ollama pull llama3.2
ollama serve &

# 3. Download specs and build the vector index
python scripts/download_specs.py          # all 37 specs from 3GPP FTP
python scripts/build_index.py

# 4. Run the assistant
python scripts/query.py "What is the gNB-CU/DU split?"   # single query
python scripts/query.py                                    # interactive REPL
streamlit run src/frontend/app.py                          # web UI
```

### Option B — Cloud deploy (Groq API key required)

The live app at [3gpp-rag-assistant.streamlit.app](https://3gpp-rag-assistant.streamlit.app/) runs Groq `llama-3.3-70b-versatile` with a pre-built ChromaDB vectordb. To self-host the same configuration:

1. Get a free Groq API key at [console.groq.com/keys](https://console.groq.com/keys).
2. Upload a pre-built vectordb tarball to a GitHub Release; set `VECTORDB_URL` to the asset URL.
3. Set Streamlit secrets: `GROQ_API_KEY`, `VECTORDB_URL`.
4. Deploy `streamlit_app.py` via [share.streamlit.io](https://share.streamlit.io).

For local development with Groq instead of Ollama, set `LLM_PROVIDER=groq` and `GROQ_API_KEY=<key>` in `.env`.

```bash
# Install cloud dependencies only
pip install -r requirements-cloud.txt
```

---

## Configuration

Key `.env` settings:

```env
# LLM provider: "ollama" (local, no API key) or "groq" (cloud, key required)
LLM_PROVIDER=ollama

# Ollama (local path)
OLLAMA_BASE_URL=http://localhost:11434
LLM_MODEL=llama3.2            # or: mistral, phi3, deepseek-r1

# Groq (cloud path)
GROQ_API_KEY=                 # get at https://console.groq.com/keys
# groq_model is set in src/config.py: llama-3.3-70b-versatile

# Embeddings (local sentence-transformers, applies to local path only)
EMBEDDING_MODEL=bge-small     # or: mini, mpnet, bge-base

# Vector database
VECTOR_DB_PATH=./data/vectordb
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# API
API_PORT=8000
LOG_LEVEL=INFO

# RAG
MAX_HISTORY_LENGTH=5
TOP_K_RESULTS=5
QUERY_EXPANSION=true          # 3GPP vocabulary expansion + rank fusion at query time
```

### Embedding models (local path)

| Key | Model | Size | Dimensions | Notes |
|---|---|---|---|---|
| `bge-small` | BAAI/bge-small-en-v1.5 | 130MB | 384 | Default — best technical-doc accuracy at this size |
| `bge-base` | BAAI/bge-base-en-v1.5 | 440MB | 768 | Higher accuracy, more RAM |
| `mini` | all-MiniLM-L6-v2 | 80MB | 384 | Faster, slightly lower accuracy |
| `mpnet` | all-mpnet-base-v2 | 420MB | 768 | Good balance |

### LLM options (local path via Ollama)

| Model | Size | Notes |
|---|---|---|
| `llama3.2` | 2GB | Default |
| `mistral` | 4GB | Strong on technical text |
| `phi3` | 2.2GB | Fastest, good for constrained hardware |
| `deepseek-r1` | 4.7GB | Best for multi-step reasoning queries |

---

## Project structure

```
3gpp-rag-assistant/
├── src/
│   ├── api/
│   │   ├── main.py             # FastAPI: query, catalog, eval, health; rate limiting; security headers
│   │   └── models.py           # Pydantic v2 schemas with input bounds
│   ├── core/
│   │   ├── document_processor.py
│   │   ├── embeddings.py       # sentence-transformers, 4 model options
│   │   ├── vector_store.py     # ChromaDB wrapper with metadata where-filters
│   │   ├── retriever.py        # Semantic search, domain/generation filtering
│   │   ├── llm.py              # Ollama client (local path)
│   │   ├── groq_llm.py         # Groq client (cloud path)
│   │   ├── rag_chain.py        # Full pipeline + conversation memory
│   │   └── spec_catalog.py     # 37-spec registry (5G/LTE × RAN/CORE) + FTP URLs
│   ├── utils/
│   │   ├── logger.py
│   │   └── metrics.py          # Per-query timing + JSON persistence
│   └── frontend/
│       └── app.py              # Streamlit chat UI (connects to FastAPI)
├── streamlit_app.py            # Self-contained Streamlit Cloud entry point (Groq + ONNX)
├── scripts/
│   ├── download_specs.py       # Fetch latest specs from 3GPP FTP
│   ├── build_index.py          # Build ChromaDB index
│   ├── build_index_cpu.py      # CPU-only build (avoids MPS stalls on large files)
│   ├── query.py                # CLI query interface
│   └── eval_retrieval.py       # Retrieval + answer-quality eval harness
├── tests/
│   ├── conftest.py
│   ├── test_embeddings.py      # 14 tests
│   ├── test_vector_store.py    # 11 tests
│   ├── test_retriever.py       # 15 tests
│   ├── test_llm.py             # 9 tests
│   ├── test_rag_chain.py       # 14 tests
│   ├── test_metrics.py         # 21 tests
│   ├── test_api.py             # 30 tests
│   ├── test_eval.py            # 40 tests
│   └── test_integration.py     # Requires live index (excluded from unit run)
├── data/
│   ├── raw/                    # 3GPP spec files (PDF / DOCX / DOC)
│   └── eval_results.json       # Latest retrieval eval output
├── docs/
│   ├── GETTING_STARTED.md
│   ├── EVAL_REPORT.md          # Eval methodology + results
│   ├── THREAT_MODEL.md         # STRIDE threat model with file:line control map
│   ├── DECISIONS.md            # Architecture decision log (ADRs)
│   ├── METRICS.md              # Product + eval metrics
│   ├── ROADMAP.md              # Canonical product roadmap
│   ├── PHASE2.md               # Phase 2 plan — retrieval v2, release intelligence, second corpus
│   ├── GENERALIZE.md           # How to use this with a non-3GPP corpus
│   ├── assets/
│   │   └── DEMO_CAPTURE.md     # Demo GIF script (recording pending)
│   ├── demo_qa.md
│   └── demo_slide.md
├── requirements.txt            # Full local deps (Ollama path)
├── requirements-cloud.txt      # Cloud deps (Groq + ONNX, no torch)
├── Dockerfile
├── docker-compose.yml
└── .env.example
```

---

## Testing

```bash
# Run all unit tests (no live services needed)
pytest tests/ --ignore=tests/test_integration.py

# With coverage
pytest tests/ --ignore=tests/test_integration.py --cov=src --cov-report=term-missing

# Integration tests (requires populated vector store)
pytest -m integration tests/test_integration.py

# Run the eval harness (retrieval metrics only, no LLM needed)
python scripts/eval_retrieval.py --output data/eval_results.json

# Run full eval including answer quality (requires Ollama running with a model)
python scripts/eval_retrieval.py --full --output data/eval_results.json
```

**302 tests, all mocked — no live services required. 310 including integration tests (which need a populated vector store).** `pytest tests/ --ignore=tests/test_integration.py` collects 302 and passes all of them in a clean virtual environment with the mocked dependencies in `conftest.py`. (In a dev venv that has drifted to starlette ≥0.36, 35 FastAPI route tests error on a known `on_startup` incompatibility — `requirements.txt` pins `starlette>=0.35,<0.36` to keep the suite green.)

CI runs on Python 3.9, 3.10, 3.11 (GitHub Actions). Checks: flake8, black, pytest --cov, Codecov.

---

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

---

## Deployment details

### Docker Compose (local, all services)

```bash
cp .env.example .env
docker compose up -d
docker compose exec ollama ollama pull llama3.2
docker compose run --rm api python scripts/build_index.py
open http://localhost:8501
```

### Services

| Service | URL | Notes |
|---|---|---|
| Streamlit UI | http://localhost:8501 | Chat UI with domain/generation filter |
| FastAPI | http://localhost:8000 | REST API |
| API docs | http://localhost:8000/docs | OpenAPI |
| Health | http://localhost:8000/health | Component status |
| Catalog | http://localhost:8000/catalog | 37 specs with indexed status |
| Eval | http://localhost:8000/eval | Latest eval results |

---

## Contributing

1. Fork the repository.
2. Create a feature branch.
3. Add unit tests for new logic (mock external dependencies).
4. Run `black .` and `flake8` before opening a PR.
5. For changes that affect the architecture or eval methodology, update the relevant doc in `docs/`.

---

## License

MIT — see [LICENSE](LICENSE).

---

## Contact

**Eric Costa**
Email: ericcosta.public@gmail.com
LinkedIn: [linkedin.com/in/niloyericcosta](https://www.linkedin.com/in/niloyericcosta/)
GitHub: [github.com/ericnc09](https://github.com/ericnc09)

---

> This is a personal side project demonstrating AI product engineering applied to a regulated technical domain. It is not affiliated with or endorsed by 3GPP or Rogers Communications.


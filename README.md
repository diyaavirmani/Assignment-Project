# 3GPP Verified RAG

An evidence-gated Retrieval-Augmented Generation system for answering questions from 3GPP telecom standards with source citations, reranking, and post-generation verification.

| Resource | Link |
|---|---|
| Deployed Application | **[Live Demo](https://assignment-project-nokk2qwnkvtctabihvejnb.streamlit.app/)** |
| GitHub Repository | **[Assignment-Project](https://github.com/diyaavirmani/Assignment-Project)** |

---

## Description

3GPP standards contain thousands of pages of highly technical telecom documentation. Finding a precise answer often requires manually searching through multiple specifications.

**3GPP Verified RAG** converts this process into a conversational search experience.

Instead of allowing the language model to answer freely, the system follows a verification-first pipeline:

1. Search the indexed 3GPP specifications.
2. Retrieve the most relevant passages.
3. Rerank them using a Cross-Encoder.
4. Check whether enough strong evidence exists.
5. Generate an answer only when the evidence is sufficient.
6. Validate citations and verify the generated answer before returning it.

If reliable evidence cannot be found, the system **refuses to answer instead of guessing**.

> The goal is not to claim universal "zero hallucinations", but to reduce unsupported answers through retrieval, evidence gating, citations, and verification.

---

## Evaluation

The project includes a dedicated **60-query evaluation harness** containing:

- direct factual questions
- procedural questions
- specification lookups
- multi-source questions
- acronym definitions
- comparison questions
- out-of-domain questions
- unanswerable questions
- adversarial prompts
- misleading premises

### Final Evaluation Results

| Metric | Result |
|---|---:|
| Correct Answer Rate | **76.9%** |
| Correct Refusal Rate | **100%** |
| Unsafe Answer Rate | **0%** |
| Unsupported Answer Escape Rate | **0%** |
| Citation Validity | **100%** |
| Retrieval Hit@1 | **94.9%** |
| Retrieval Hit@3 | **100%** |
| Retrieval Hit@5 | **100%** |
| MRR | **97.0%** |

These results are measured on the project's curated 60-query evaluation suite for the currently indexed corpus.

The evaluation harness also records retrieval quality, evidence-gate decisions, verifier behavior, citations, refusals, and latency.

---

## Architecture

### System Architecture Diagram

![3GPP Verified RAG evidence-gated architecture](docs/assets/architecture.png)

### Application Flow

```text
Streamlit
-> FastAPI
-> OpenAI Embeddings
-> Pinecone
-> Cross-Encoder Reranker
-> Evidence Gate
-> LLM
-> Citation Validation
-> Answer Verifier
-> Response
```

The system uses **defence in depth** rather than trusting a single RAG retrieval score.

---

## Corpus

The current production corpus contains:

| Specification | Description |
|---|---|
| **TS 38.300** | NR and NG-RAN Overall Description |
| **TS 38.401** | NG-RAN Architecture Description |

The corpus contains approximately **1,561 indexed chunks**.

Documents are divided using:

- **Chunk size:** 1000
- **Chunk overlap:** 200

The overlap helps preserve context when important definitions or procedures cross chunk boundaries.

### Why Pinecone?

Pinecone is used as the production vector database because it provides:

- managed vector search
- scalable cloud deployment
- namespace isolation
- metadata filtering
- fast semantic retrieval
- easier production deployment than maintaining a local vector database

Metadata such as specification number, generation, and domain can be stored alongside each vector and used for filtered retrieval.

---

## RAG Strategy

The project uses a **retrieve -> rerank -> gate -> generate -> verify** strategy.

### Embeddings

**OpenAI `text-embedding-3-small`**

Each document chunk and user query is converted into a numerical vector representation.

### Retrieval

Pinecone retrieves the **top 10 candidate chunks** using semantic similarity.

### Reranking

A Cross-Encoder:

```text
cross-encoder/ms-marco-MiniLM-L6-v2
```

reranks the retrieved candidates based on how relevant each passage is to the actual question.

The best **5 passages** are passed to the next stage.

### Evidence Gate

Before calling the LLM, the Evidence Gate checks whether the retrieved evidence is strong enough.

Current calibrated configuration:

- minimum top reranker score: **0.80**
- minimum document score: **0.80**
- minimum strong documents: **2**

If the evidence is weak, the pipeline stops immediately and returns a refusal.

### Generation

Grounded answers are generated using:

**OpenAI `gpt-4o-mini`**

The model is instructed to answer using only the supplied evidence.

### Verification

Before the answer reaches the user, the system checks:

- whether citations exist
- whether citations refer to retrieved evidence
- whether claims are supported
- whether contradictions exist

A failed verification results in a safe refusal instead of releasing the generated answer.

---

## Trade-offs

The project intentionally prioritizes **reliability over minimum latency**.

Adding reranking and answer verification increases response time, but significantly improves control over unsupported answers.

Another important trade-off was avoiding unnecessary complexity.

Techniques such as additional hybrid retrieval layers or more agents were not added simply because they were popular. Changes were introduced only when the evaluation results showed that they solved a measurable problem.

---

## Interface

The application uses a **Streamlit frontend** connected to a **FastAPI backend**.

The interface exposes:

- natural-language questioning
- multi-turn conversations
- indexed corpus information
- source evidence
- reranker scores
- Evidence Gate decisions
- verification results
- latency information
- safe refusal states

---

## How to Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/diyaavirmani/Assignment-Project.git
cd Assignment-Project
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file using `.env.example`.

Add your:

```env
OPENAI_API_KEY=
PINECONE_API_KEY=
```

### 5. Start the FastAPI backend

```bash
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

### 6. Start the Streamlit frontend

Open another terminal:

```bash
streamlit run src/frontend/app.py
```

Then open:

```text
http://localhost:8501
```

---

## Configuration

Important production settings include:

```env
LLM_PROVIDER=openai
OPENAI_MODEL=gpt-4o-mini

EMBEDDING_PROVIDER=openai
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

VECTOR_STORE_PROVIDER=pinecone
PINECONE_INDEX_NAME=3gpp-rag
PINECONE_NAMESPACE=3gpp-specs

RERANKER_ENABLED=true
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L6-v2
RERANKER_CANDIDATE_K=10
RERANKER_TOP_K=5

EVIDENCE_GATE_ENABLED=true
EVIDENCE_SCORE_SOURCE=reranker
EVIDENCE_MIN_TOP_SCORE=0.80
EVIDENCE_MIN_DOC_SCORE=0.80
EVIDENCE_MIN_DOCS=2
```

Never commit `.env` or API keys to GitHub.

---

## Embedding and AI Models

| Component | Model |
|---|---|
| Embeddings | `text-embedding-3-small` |
| Vector Database | Pinecone |
| Reranker | `ms-marco-MiniLM-L6-v2` |
| Answer Generation | `gpt-4o-mini` |
| Answer Verification | `gpt-4o-mini` |

The key design decision is that the LLM is **not the only safety mechanism**. Retrieval quality, reranking, deterministic evidence gating, citation checks, and verification all participate in deciding whether an answer can be released.

---

## Project Structure

```text
Assignment-Project/
|
|-- src/
|   |-- api/                 # FastAPI endpoints
|   |-- core/                # Core RAG pipeline
|   |-- frontend/            # Streamlit interface
|   `-- utils/               # Logging and metrics
|
|-- scripts/                 # Indexing and evaluation utilities
|-- tests/                   # Automated tests
|
|-- data/
|   `-- eval/                # Evaluation dataset
|
|-- docs/                    # Architecture decisions and documentation
|
|-- requirements.txt
|-- .env.example
`-- README.md
```

---

## Testing

The system is tested at multiple levels:

**Component testing**

Embedding, retrieval, reranking, evidence gating, verification, API behavior, and frontend response handling.

**Workflow testing**

- valid telecom query
- weak-evidence query
- out-of-domain query
- misleading premise
- verification failure
- multi-turn conversation
- source citation handling

**Evaluation testing**

The 60-query evaluation harness measures both answering ability and, equally importantly, the system's ability to **refuse unsafe or unsupported questions**.

Run automated tests with:

```bash
pytest tests/ --ignore=tests/test_integration.py -q
```

---

## Future Scope

The same architecture can be extended beyond telecom standards into a complete **Company Knowledge AI system**.

A future version could ingest:

- company policies
- internal documentation
- SOPs
- contracts
- emails
- reports
- knowledge bases
- operational manuals

The next stage could combine this knowledge layer with a domain-specific **Small Language Model (SLM)** or fine-tuned model.

A company could then build:

**Private Company Assistant**

A secure chatbot trained or adapted to understand internal terminology and organizational knowledge.

**Enterprise Guardrails**

Role-based access, PII protection, policy enforcement, audit logs, citation requirements, and approval workflows.

**Office Automation**

The system could connect knowledge with actions such as:

- drafting reports
- preparing emails
- finding internal policies
- summarizing documents
- generating forms
- routing approvals

**Workflow Automation**

AI agents could use verified company knowledge to assist with processes such as:

- HR onboarding
- procurement
- compliance
- customer support
- incident management
- legal document review

The long-term idea is to move from:

```text
Documents -> Search
```

to:

```text
Documents -> Verified Intelligence -> Automated Workflows
```

---

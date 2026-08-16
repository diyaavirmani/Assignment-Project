# Verified RAG Evaluation

This suite measures the current guarded RAG stack. It does not tune evidence
thresholds, relax the verifier, add reranking, or change production behavior.

## Dataset

Default dataset:

```bash
data/eval/3gpp_verified_rag_eval.jsonl
```

Each JSONL row uses this schema:

```json
{
  "id": "df-001",
  "category": "direct_fact",
  "question": "What is a gNB in NG-RAN?",
  "expected_behavior": "answer",
  "gold_spec_numbers": ["38.300"],
  "gold_sources": ["38300"],
  "gold_answer": "Short curated reference answer or null",
  "required_terms": ["gNB"],
  "forbidden_terms": [],
  "notes": "Human-readable rationale"
}
```

`expected_behavior` is either `answer` or `refuse`. Gold references should only
name specifications present in the available corpus. The initial dataset is
therefore centered on locally present TS 38.300 and TS 38.401 evidence plus
out-of-domain, unanswerable, adversarial, and misleading-premise refusal cases.

## Metrics

The runner records per-query evidence, verification, citations, sources,
similarity scores, timings, and errors. It aggregates:

- Retrieval: Hit@1, Hit@3, Hit@5, MRR
- Behavior: answer rate, refusal rate, correct answer/attempt rate, correct
  refusal rate, false refusal rate, unsafe answer rate
- Evidence gate: pass/rejection rate and reason distributions
- Verifier: pass/rejection rate, rejection reasons, verification failure after
  successful evidence gate
- Citations: citation validity rate, citation coverage, source precision when
  gold labels exist
- Groundedness: verified grounded answer rate and unsupported answer escape rate
- Latency: mean/median retrieval, verification, and total latency
- Category breakdown and failure buckets for Prompt 8 analysis

Failure buckets include `retrieval_miss`, `false_refusal_evidence_gate`,
`verification_rejection`, `unsafe_answer`, `invalid_citation`,
`missing_citation`, and `api_error`.

## Running

The real evaluation runner uses the configured RAG stack and may call OpenAI
and Pinecone. It is never invoked by pytest.

Small smoke:

```bash
python scripts/run_evaluation.py --limit 5
```

Medium run:

```bash
python scripts/run_evaluation.py --limit 20
```

Full run:

```bash
python scripts/run_evaluation.py
```

Useful filters:

```bash
python scripts/run_evaluation.py --category out_of_domain
python scripts/run_evaluation.py --run-name guarded-current --output-dir eval_results
```

## Outputs

Each run writes timestamped files under `eval_results/`:

- `run_YYYYMMDD_HHMMSS.json`
- `run_YYYYMMDD_HHMMSS.csv`
- `run_YYYYMMDD_HHMMSS_summary.json`
- `run_YYYYMMDD_HHMMSS_report.md`

Runtime outputs are ignored by Git. They may contain generated answers and
retrieved snippets, but not API keys, hidden prompts, chain-of-thought, or
secret environment values.

## Unit Tests

Evaluation unit tests use fabricated mock RAG outputs. They validate dataset
parsing, schema checks, classification, metrics, aggregation, error handling,
and report serialization without real OpenAI, Pinecone, Ollama, or Groq calls.

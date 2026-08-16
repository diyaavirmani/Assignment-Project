"""
scripts.eval — retrieval and answer-quality metric functions for the 3GPP RAG eval harness.

Modules:
    metrics   — standard IR metrics: Recall@k, MRR, nDCG@k, hit-rate; heuristic
                context-precision/recall (RAGAS-inspired).
    judge     — LLM-judge path for faithfulness/groundedness + answer-correctness
                (Groq or Ollama; results marked [RUN REQUIRED] if not executed).
    regression — baseline comparison logic for --regression CI gate.
"""

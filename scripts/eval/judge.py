"""
LLM-judge path for faithfulness/groundedness and answer-correctness.

STATUS: Implementation is complete and wired up, but results are marked
[RUN REQUIRED] unless actually executed against the live index + LLM.
The judge model is configurable: pass --judge-provider groq|ollama and
--judge-model <name> to the eval harness.

Design
------
The LLM judge evaluates each (question, retrieved-context, answer) triple
on two axes using a structured prompt:

  1. Faithfulness / groundedness (0–5 integer score)
     Does the answer make claims supported by the retrieved context?
     A score of 5 means every claim traces to a context passage; 0 means
     the answer contradicts or fabricates beyond the context.
     This directly measures hallucination risk — the primary concern for
     enterprise RAG in regulated domains.

  2. Answer correctness (0–5 integer score)
     Does the answer address the question accurately and completely given
     what the context provides?  This is NOT "does it match a reference
     answer" (we don't have reference answers in the golden set); it is
     the judge's assessment of technical accuracy within the context window.

Both scores are normalised to [0, 1] before aggregation.

Why Groq-or-Ollama?
-------------------
The production app already dual-supports Groq (cloud) and Ollama (local).
Using the same providers for the judge avoids adding new dependencies and
keeps the eval self-contained.  Default judge model:
  - Groq:   llama-3.3-70b-versatile (same as production)
  - Ollama: llama3.2 (smaller, local; accuracy may be lower)

Limitations
-----------
- LLM-judge scores depend on the judge model's own capabilities and biases.
  A 70B judge is not a ground-truth oracle; treat scores as strong signals
  with ~±0.5 score uncertainty.
- Self-judging (using the same model as the responder) is a known bias risk.
  Use a different judge model where possible (e.g., judge with GPT-4 when
  the RAG uses Llama).
- Judge prompts must be versioned alongside golden-set updates; a prompt
  change can shift scores without any change to the underlying RAG quality.
"""

import json
import os
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Judge prompt
# ---------------------------------------------------------------------------

JUDGE_PROMPT_TEMPLATE = """\
You are evaluating a RAG system's answer against the retrieved context.
Respond ONLY with valid JSON — no explanation outside the JSON block.

Question: {question}

Retrieved context:
---
{context}
---

Answer to evaluate:
---
{answer}
---

Score the answer on TWO dimensions using integers 0-5:

1. faithfulness: Does the answer only make claims supported by the context?
   5 = every claim traces to the context; no fabrication.
   3 = mostly grounded, one unsupported claim.
   1 = several claims beyond or contradicting the context.
   0 = answer ignores the context entirely.

2. answer_correctness: Does the answer address the question accurately and
   completely, given what the context provides?
   5 = accurate and complete given the context.
   3 = partially correct or missing key details from the context.
   1 = mostly incorrect or misleading.
   0 = does not address the question.

Reply with exactly this JSON and nothing else:
{{"faithfulness": <int 0-5>, "answer_correctness": <int 0-5>, "notes": "<one sentence>"}}
"""


# ---------------------------------------------------------------------------
# Provider wrappers
# ---------------------------------------------------------------------------

def _call_groq(prompt: str, model: str, api_key: str) -> str:
    """Call Groq inference API and return the raw text response."""
    from groq import Groq  # type: ignore
    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=200,
    )
    return response.choices[0].message.content.strip()


def _call_ollama(prompt: str, model: str, base_url: str) -> str:
    """Call Ollama inference API and return the raw text response."""
    import urllib.request
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
    }).encode()
    req = urllib.request.Request(
        f"{base_url}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read())
    return data.get("response", "").strip()


# ---------------------------------------------------------------------------
# Score parsing
# ---------------------------------------------------------------------------

def _parse_judge_response(raw: str) -> Dict:
    """Parse the JSON block from the judge response.

    Returns a dict with faithfulness, answer_correctness, and notes.
    On parse failure returns a sentinel dict so the caller can log and skip.
    """
    try:
        # Strip markdown code fences if present
        text = raw.strip()
        if text.startswith("```"):
            text = "\n".join(text.split("\n")[1:])
            text = text.rstrip("`").strip()
        data = json.loads(text)
        return {
            "faithfulness":       max(0, min(5, int(data.get("faithfulness", 0)))),
            "answer_correctness": max(0, min(5, int(data.get("answer_correctness", 0)))),
            "notes":              str(data.get("notes", "")),
            "parse_ok":           True,
        }
    except Exception as exc:
        return {
            "faithfulness":       None,
            "answer_correctness": None,
            "notes":              f"parse_error: {exc}; raw={raw[:200]}",
            "parse_ok":           False,
        }


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

class LLMJudge:
    """Configurable LLM judge for faithfulness + answer-correctness scoring.

    [RUN REQUIRED] — Results are only produced when this class is instantiated
    and judge() is called with real answers and context.

    Args:
        provider:   "groq" | "ollama"
        model:      Model name. Defaults per provider:
                    groq  → "llama-3.3-70b-versatile"
                    ollama→ "llama3.2"
        api_key:    Groq API key. Falls back to GROQ_API_KEY env var.
        ollama_url: Ollama base URL (default "http://localhost:11434").
    """

    DEFAULTS = {
        "groq":   "llama-3.3-70b-versatile",
        "ollama": "llama3.2",
    }

    def __init__(
        self,
        provider: str = "groq",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        ollama_url: str = "http://localhost:11434",
    ) -> None:
        if provider not in ("groq", "ollama"):
            raise ValueError(f"provider must be 'groq' or 'ollama', got: {provider!r}")
        self.provider = provider
        self.model = model or self.DEFAULTS[provider]
        self.api_key = api_key or os.environ.get("GROQ_API_KEY", "")
        self.ollama_url = ollama_url

    def is_available(self) -> bool:
        """Return True if the configured provider appears reachable."""
        if self.provider == "groq":
            return bool(self.api_key)
        # Ollama: attempt a quick connection
        import urllib.request
        try:
            urllib.request.urlopen(f"{self.ollama_url}/api/tags", timeout=3)
            return True
        except Exception:
            return False

    def judge(
        self,
        question: str,
        context: str,
        answer: str,
    ) -> Dict:
        """Score one (question, context, answer) triple.

        Returns:
            {
                "faithfulness":          float,  # 0–1  (raw_score / 5)
                "answer_correctness":    float,  # 0–1
                "faithfulness_raw":      int,    # 0–5
                "answer_correctness_raw":int,    # 0–5
                "notes":                 str,
                "judge_model":           str,
                "parse_ok":              bool,
            }
        """
        prompt = JUDGE_PROMPT_TEMPLATE.format(
            question=question,
            context=context[:3000],   # cap to avoid token overrun
            answer=answer,
        )
        if self.provider == "groq":
            raw = _call_groq(prompt, self.model, self.api_key)
        else:
            raw = _call_ollama(prompt, self.model, self.ollama_url)

        parsed = _parse_judge_response(raw)
        if not parsed["parse_ok"]:
            return {**parsed, "judge_model": self.model,
                    "faithfulness": None, "answer_correctness": None,
                    "faithfulness_raw": None, "answer_correctness_raw": None}

        return {
            "faithfulness":            round(parsed["faithfulness"] / 5, 3),
            "answer_correctness":      round(parsed["answer_correctness"] / 5, 3),
            "faithfulness_raw":        parsed["faithfulness"],
            "answer_correctness_raw":  parsed["answer_correctness"],
            "notes":                   parsed["notes"],
            "judge_model":             self.model,
            "parse_ok":                True,
        }

    def judge_batch(
        self,
        cases: List[Dict],
        verbose: bool = False,
    ) -> List[Dict]:
        """Score a list of eval case dicts.

        Each case dict must contain:
            query, answer (str), context (str — the formatted RAG context)

        Returns the input list enriched with judge scores under the
        "llm_judge" key.  Cases where the judge fails to produce a valid
        score have llm_judge.parse_ok = False.
        """
        results = []
        for i, case in enumerate(cases, 1):
            if verbose:
                print(f"  Judging {i}/{len(cases)}: {case.get('query', '')[:60]}")
            scores = self.judge(
                question=case.get("query", ""),
                context=case.get("context", ""),
                answer=case.get("answer", ""),
            )
            results.append({**case, "llm_judge": scores})
        return results

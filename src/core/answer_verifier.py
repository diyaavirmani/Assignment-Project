"""Post-generation grounding and citation verification for RAG answers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import logging
import re
from typing import Any, Dict, List, Optional, Sequence, Set

from src.config import settings

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - exercised only when dependency is absent
    OpenAI = None

logger = logging.getLogger(__name__)


REASON_VERIFIED = "verified"
REASON_UNSUPPORTED = "unsupported_claim"
REASON_CONTRADICTION = "contradiction"
REASON_MISSING_CITATIONS = "missing_citations"
REASON_INVALID_CITATION = "invalid_citation"
REASON_VERIFICATION_ERROR = "verification_error"
REASON_EMPTY_ANSWER = "empty_answer"
REASON_DISABLED = "verification_disabled"

SUPPORTED = "SUPPORTED"
UNSUPPORTED = "UNSUPPORTED"
CONTRADICTED = "CONTRADICTED"

SOURCE_CITATION_RE = re.compile(r"\[S(\d+)\]")


@dataclass
class VerificationResult:
    """Structured verifier result safe to expose as response metadata."""

    passed: bool
    reason: str
    total_claims: int = 0
    supported_claims: int = 0
    unsupported_claims: List[str] = field(default_factory=list)
    contradicted_claims: List[str] = field(default_factory=list)
    citation_valid: bool = False
    cited_sources: List[str] = field(default_factory=list)
    invalid_citations: List[str] = field(default_factory=list)
    detail: str = ""

    def to_dict(self) -> Dict:
        """Return a JSON-serialisable representation."""
        return asdict(self)


class AnswerVerifier:
    """Verify generated answers against retrieved evidence before release."""

    def __init__(
        self,
        enabled: Optional[bool] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        client: Optional[Any] = None,
        max_tokens: int = 900,
    ) -> None:
        self.enabled = settings.answer_verification_enabled if enabled is None else enabled
        self.model = model or settings.openai_verifier_model or settings.openai_model
        self.max_tokens = max_tokens

        if not self.enabled:
            self._client = client
            return
        if not self.model:
            raise ValueError(
                "OPENAI_VERIFIER_MODEL or OPENAI_MODEL is required for answer verification"
            )
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be greater than zero")

        if client is not None:
            self._client = client
            return

        resolved_api_key = api_key if api_key is not None else settings.openai_api_key
        if not resolved_api_key:
            raise ValueError("OPENAI_API_KEY is required for answer verification")
        if OpenAI is None:
            raise ImportError("openai is required. Install with: pip install openai")

        self._client = OpenAI(api_key=resolved_api_key)
        logger.info("Initialized AnswerVerifier: model=%s", self.model)

    def verify(
        self,
        question: str,
        documents: Sequence[Dict],
        answer: str,
    ) -> VerificationResult:
        """Verify that an answer is cited and grounded in retrieved documents."""
        if not self.enabled:
            return VerificationResult(
                passed=True,
                reason=REASON_DISABLED,
                detail="Answer verification disabled by configuration.",
            )

        citation_result = self.validate_citations(answer, documents)
        if not citation_result.passed:
            return citation_result

        try:
            payload = self._call_grounding_verifier(question, documents, answer)
            return self._result_from_payload(payload, citation_result)
        except Exception as exc:
            logger.warning("Answer verification failed closed: %s", exc)
            return VerificationResult(
                passed=False,
                reason=REASON_VERIFICATION_ERROR,
                citation_valid=citation_result.citation_valid,
                cited_sources=citation_result.cited_sources,
                invalid_citations=citation_result.invalid_citations,
                detail="Verifier could not produce a valid grounded result.",
            )

    def validate_citations(
        self,
        answer: str,
        documents: Sequence[Dict],
    ) -> VerificationResult:
        """Deterministically validate [S<number>] citations in an answer."""
        if not answer or not answer.strip():
            return VerificationResult(
                passed=False,
                reason=REASON_EMPTY_ANSWER,
                detail="Generated answer was empty.",
            )

        cited_sources = self._extract_citations(answer)
        if not cited_sources:
            return VerificationResult(
                passed=False,
                reason=REASON_MISSING_CITATIONS,
                detail="Generated factual answer did not cite retrieved sources.",
            )

        valid_sources = self.available_source_ids(documents)
        invalid_citations = sorted(
            source_id for source_id in cited_sources if source_id not in valid_sources
        )
        if invalid_citations:
            return VerificationResult(
                passed=False,
                reason=REASON_INVALID_CITATION,
                citation_valid=False,
                cited_sources=sorted(cited_sources, key=self._source_sort_key),
                invalid_citations=invalid_citations,
                detail="Generated answer cited source IDs that were not retrieved.",
            )

        return VerificationResult(
            passed=True,
            reason=REASON_VERIFIED,
            citation_valid=True,
            cited_sources=sorted(cited_sources, key=self._source_sort_key),
        )

    def _call_grounding_verifier(
        self,
        question: str,
        documents: Sequence[Dict],
        answer: str,
    ) -> Dict:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a strict grounding verifier for a 3GPP RAG system. "
                    "Use only the supplied retrieved evidence. Do not use external "
                    "knowledge. Return only valid JSON."
                ),
            },
            {
                "role": "user",
                "content": self._build_verifier_prompt(question, documents, answer),
            },
        ]
        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0,
            max_tokens=self.max_tokens,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        if not content:
            raise ValueError("Verifier returned empty content")
        parsed = json.loads(content)
        if not isinstance(parsed, dict):
            raise ValueError("Verifier JSON must be an object")
        return parsed

    def _build_verifier_prompt(
        self,
        question: str,
        documents: Sequence[Dict],
        answer: str,
    ) -> str:
        evidence = self.format_evidence(documents)
        return f"""Verify the generated answer against the retrieved 3GPP evidence.

Return JSON with this exact shape:
{{
  "claims": [
    {{
      "claim": "material factual claim text",
      "status": "SUPPORTED|UNSUPPORTED|CONTRADICTED",
      "source_ids": ["S1"]
    }}
  ]
}}

Rules:
- Identify only material telecom/3GPP factual claims.
- Ignore harmless connective wording such as "in summary".
- Mark SUPPORTED only when the retrieved evidence supports the claim.
- Mark UNSUPPORTED when evidence does not establish the claim.
- Mark CONTRADICTED when evidence conflicts with the claim.
- Do not use external knowledge.

Question:
{question}

Retrieved evidence:
{evidence}

Generated answer:
{answer}
"""

    def _result_from_payload(
        self,
        payload: Dict,
        citation_result: VerificationResult,
    ) -> VerificationResult:
        claims = payload.get("claims")
        if not isinstance(claims, list):
            raise ValueError("Verifier response missing claims list")

        unsupported_claims: List[str] = []
        contradicted_claims: List[str] = []
        supported_claims = 0

        for item in claims:
            if not isinstance(item, dict):
                raise ValueError("Verifier claim entries must be objects")
            claim = item.get("claim")
            status = item.get("status")
            if not isinstance(claim, str) or not claim.strip():
                raise ValueError("Verifier claim missing text")
            if status not in {SUPPORTED, UNSUPPORTED, CONTRADICTED}:
                raise ValueError("Verifier claim has invalid status")

            if status == SUPPORTED:
                supported_claims += 1
            elif status == UNSUPPORTED:
                unsupported_claims.append(claim)
            elif status == CONTRADICTED:
                contradicted_claims.append(claim)

        total_claims = len(claims)
        if contradicted_claims:
            reason = REASON_CONTRADICTION
        elif unsupported_claims:
            reason = REASON_UNSUPPORTED
        else:
            reason = REASON_VERIFIED

        passed = reason == REASON_VERIFIED
        return VerificationResult(
            passed=passed,
            reason=reason,
            total_claims=total_claims,
            supported_claims=supported_claims,
            unsupported_claims=unsupported_claims,
            contradicted_claims=contradicted_claims,
            citation_valid=citation_result.citation_valid,
            cited_sources=citation_result.cited_sources,
            invalid_citations=citation_result.invalid_citations,
        )

    @staticmethod
    def available_source_ids(documents: Sequence[Dict]) -> Set[str]:
        """Return source IDs available for one retrieved response."""
        return {f"S{i}" for i, _ in enumerate(documents, 1)}

    @staticmethod
    def format_evidence(documents: Sequence[Dict]) -> str:
        """Format retrieved documents with deterministic source IDs."""
        parts = []
        for i, doc in enumerate(documents, 1):
            spec_number = doc.get("spec_number") or "unknown"
            spec_title = doc.get("spec_title") or "unknown"
            parts.append(
                "\n".join(
                    [
                        f"[S{i}]",
                        f"Spec: TS {spec_number}",
                        f"Title: {spec_title}",
                        f"Source: {doc.get('source', 'unknown')}",
                        f"Similarity: {float(doc.get('similarity', 0.0)):.3f}",
                        "Text:",
                        str(doc.get("text", "")),
                    ]
                )
            )
        return "\n\n".join(parts)

    @staticmethod
    def _extract_citations(answer: str) -> Set[str]:
        return {f"S{match}" for match in SOURCE_CITATION_RE.findall(answer or "")}

    @staticmethod
    def _source_sort_key(source_id: str) -> int:
        match = re.fullmatch(r"S(\d+)", source_id)
        return int(match.group(1)) if match else 0

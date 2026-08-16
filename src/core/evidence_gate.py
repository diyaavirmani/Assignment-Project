"""Deterministic evidence gate for RAG retrieval results."""

from dataclasses import asdict, dataclass
import logging
from typing import Dict, List, Optional

from src.config import settings

logger = logging.getLogger(__name__)


REASON_SUFFICIENT = "sufficient_evidence"
REASON_DISABLED = "gate_disabled"
REASON_NO_DOCUMENTS = "no_documents"
REASON_TOP_SCORE_LOW = "top_score_below_threshold"
REASON_NOT_ENOUGH_DOCS = "not_enough_qualifying_documents"
REASON_MEAN_SCORE_LOW = "mean_score_below_threshold"
REASON_MISSING_SCORES = "missing_scores"


@dataclass
class EvidenceDecision:
    """Structured evidence-gate result for debugging and later evaluation."""

    sufficient: bool
    reason: str
    top_score: float
    mean_score: float
    qualifying_docs: int
    total_docs: int
    score_source: str = "vector"
    explanation: str = ""

    def to_dict(self) -> Dict:
        """Return a JSON-serialisable representation."""
        return asdict(self)


class EvidenceGate:
    """Evaluate whether retrieved chunks provide enough support to generate."""

    def __init__(
        self,
        enabled: Optional[bool] = None,
        min_top_score: Optional[float] = None,
        min_doc_score: Optional[float] = None,
        min_docs: Optional[int] = None,
        mean_top_n: Optional[int] = None,
        min_mean_score: Optional[float] = None,
        score_source: Optional[str] = None,
    ) -> None:
        self.enabled = settings.evidence_gate_enabled if enabled is None else enabled
        self.min_top_score = (
            settings.evidence_min_top_score if min_top_score is None else min_top_score
        )
        self.min_doc_score = (
            settings.evidence_min_doc_score if min_doc_score is None else min_doc_score
        )
        self.min_docs = settings.evidence_min_docs if min_docs is None else min_docs
        self.mean_top_n = settings.evidence_mean_top_n if mean_top_n is None else mean_top_n
        self.min_mean_score = (
            settings.evidence_min_mean_score if min_mean_score is None else min_mean_score
        )
        self.score_source = (
            (settings.evidence_score_source if score_source is None else score_source)
            .strip()
            .lower()
        )
        self._validate()

    def evaluate(self, documents: List[Dict]) -> EvidenceDecision:
        """Evaluate retrieved documents before generation."""
        scores = []
        for doc in documents:
            score = self._score(doc)
            if score is not None:
                scores.append(score)
        scores = sorted(scores, reverse=True)
        total_docs = len(documents)

        if not documents:
            decision = EvidenceDecision(
                sufficient=False,
                reason=REASON_NO_DOCUMENTS,
                top_score=0.0,
                mean_score=0.0,
                qualifying_docs=0,
                total_docs=0,
                score_source=self.score_source,
                explanation="No documents were retrieved.",
            )
            self._log(decision)
            return decision

        if not self.enabled:
            decision = EvidenceDecision(
                sufficient=True,
                reason=REASON_DISABLED,
                top_score=scores[0] if scores else 0.0,
                mean_score=self._mean(scores[: self.mean_top_n]),
                qualifying_docs=sum(score >= self.min_doc_score for score in scores),
                total_docs=total_docs,
                score_source=self.score_source,
                explanation="Evidence gate disabled by configuration.",
            )
            self._log(decision)
            return decision

        if not scores:
            decision = EvidenceDecision(
                sufficient=False,
                reason=REASON_MISSING_SCORES,
                top_score=0.0,
                mean_score=0.0,
                qualifying_docs=0,
                total_docs=total_docs,
                score_source=self.score_source,
                explanation=(
                    f"No {self.score_source} scores were available for retrieved documents."
                ),
            )
            self._log(decision)
            return decision

        top_score = scores[0] if scores else 0.0
        top_scores = scores[: self.mean_top_n]
        mean_score = self._mean(top_scores)
        qualifying_docs = sum(score >= self.min_doc_score for score in scores)

        if top_score < self.min_top_score:
            decision = EvidenceDecision(
                sufficient=False,
                reason=REASON_TOP_SCORE_LOW,
                top_score=top_score,
                mean_score=mean_score,
                qualifying_docs=qualifying_docs,
                total_docs=total_docs,
                score_source=self.score_source,
                explanation="Best retrieved document score is below threshold.",
            )
            self._log(decision)
            return decision

        if qualifying_docs < self.min_docs:
            decision = EvidenceDecision(
                sufficient=False,
                reason=REASON_NOT_ENOUGH_DOCS,
                top_score=top_score,
                mean_score=mean_score,
                qualifying_docs=qualifying_docs,
                total_docs=total_docs,
                score_source=self.score_source,
                explanation="Not enough retrieved documents meet the evidence score.",
            )
            self._log(decision)
            return decision

        if mean_score < self.min_mean_score:
            decision = EvidenceDecision(
                sufficient=False,
                reason=REASON_MEAN_SCORE_LOW,
                top_score=top_score,
                mean_score=mean_score,
                qualifying_docs=qualifying_docs,
                total_docs=total_docs,
                score_source=self.score_source,
                explanation="Mean score of top retrieved documents is below threshold.",
            )
            self._log(decision)
            return decision

        decision = EvidenceDecision(
            sufficient=True,
            reason=REASON_SUFFICIENT,
            top_score=top_score,
            mean_score=mean_score,
            qualifying_docs=qualifying_docs,
            total_docs=total_docs,
            score_source=self.score_source,
            explanation="Retrieved documents meet the configured evidence thresholds.",
        )
        self._log(decision)
        return decision

    def _validate(self) -> None:
        for name, value in (
            ("min_top_score", self.min_top_score),
            ("min_doc_score", self.min_doc_score),
            ("min_mean_score", self.min_mean_score),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0.0 and 1.0")
        if self.min_docs < 0:
            raise ValueError("min_docs must be greater than or equal to zero")
        if self.mean_top_n <= 0:
            raise ValueError("mean_top_n must be greater than zero")
        if self.score_source not in {"vector", "reranker"}:
            raise ValueError("score_source must be 'vector' or 'reranker'")

    def _score(self, document: Dict) -> Optional[float]:
        if self.score_source == "reranker":
            value = document.get("reranker_score")
        else:
            value = document.get("vector_similarity", document.get("similarity"))
        if value is None:
            return None
        return float(value)

    @staticmethod
    def _mean(scores: List[float]) -> float:
        if not scores:
            return 0.0
        return sum(scores) / len(scores)

    @staticmethod
    def _log(decision: EvidenceDecision) -> None:
        logger.info(
            "Evidence gate: decision=%s reason=%s score_source=%s top_score=%.4f "
            "mean_score=%.4f qualifying_docs=%s total_docs=%s",
            "SUFFICIENT" if decision.sufficient else "INSUFFICIENT",
            decision.reason,
            decision.score_source,
            decision.top_score,
            decision.mean_score,
            decision.qualifying_docs,
            decision.total_docs,
        )

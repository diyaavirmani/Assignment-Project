"""Cross-encoder reranking for retrieved 3GPP evidence."""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Sequence

from src.config import settings

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Rerank vector-search candidates with a sentence-transformers cross-encoder.

    The cross-encoder returns relevance logits for ``(query, document)`` pairs.
    We keep the raw value in ``reranker_raw_score`` and attach a deterministic
    sigmoid-normalized ``reranker_score`` in the 0..1 range for analysis. The
    normalized score is a relevance score, not a calibrated probability.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        model: Optional[Any] = None,
        device: Optional[str] = None,
    ) -> None:
        self.model_name = model_name or settings.reranker_model
        self._model = model
        self.device = device

    def rerank(self, query: str, candidates: Sequence[Dict], top_n: int = 5) -> List[Dict]:
        """Return candidates sorted by cross-encoder relevance score."""
        if top_n <= 0:
            raise ValueError("top_n must be greater than zero")
        if not candidates:
            return []

        docs = [dict(candidate) for candidate in candidates]
        for rank, doc in enumerate(docs, start=1):
            doc.setdefault("rank_before_reranking", rank)
            if doc.get("vector_similarity") is None and doc.get("similarity") is not None:
                doc["vector_similarity"] = doc["similarity"]

        pairs = [(query, str(doc.get("text", ""))) for doc in docs]
        raw_scores = self._predict(pairs)

        scored_docs: List[Dict] = []
        for doc, raw_score in zip(docs, raw_scores):
            raw_float = float(raw_score)
            doc["reranker_raw_score"] = raw_float
            doc["reranker_score"] = self.normalize_score(raw_float)
            scored_docs.append(doc)

        scored_docs.sort(key=lambda doc: doc["reranker_score"], reverse=True)
        for rank, doc in enumerate(scored_docs, start=1):
            doc["rank_after_reranking"] = rank
        return scored_docs[:top_n]

    @staticmethod
    def normalize_score(raw_score: float) -> float:
        """Map an arbitrary cross-encoder logit to a 0..1 relevance score."""
        if raw_score >= 0:
            z = math.exp(-raw_score)
            return 1.0 / (1.0 + z)
        z = math.exp(raw_score)
        return z / (1.0 + z)

    def _predict(self, pairs: Sequence[tuple[str, str]]) -> List[float]:
        model = self._load_model()
        scores = model.predict(list(pairs))
        if hasattr(scores, "tolist"):
            scores = scores.tolist()
        if isinstance(scores, (int, float)):
            scores = [scores]
        return [float(score) for score in scores]

    def _load_model(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:  # pragma: no cover
                raise ImportError(
                    "sentence-transformers is required for CrossEncoderReranker"
                ) from exc

            kwargs: Dict[str, Any] = {}
            if self.device:
                kwargs["device"] = self.device
            logger.info("Loading cross-encoder reranker model: %s", self.model_name)
            self._model = CrossEncoder(self.model_name, **kwargs)
        return self._model

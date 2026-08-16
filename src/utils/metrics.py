"""
Performance metrics tracker for the RAG pipeline

Tracks per-query timing and aggregated statistics without any external
dependencies. All data is kept in memory (and optionally persisted to JSON).

Usage:
    tracker = MetricsTracker()
    tracker.record(query="...", retrieve_time=0.3, generate_time=1.2,
                   num_sources=5, answer_length=420)
    print(tracker.summary())
"""
import hashlib
import json
import time
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from statistics import mean, median
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class QueryMetric:
    """Metrics captured for a single query."""
    query: str
    retrieve_time: float        # seconds
    generate_time: float        # seconds
    total_time: float           # seconds
    num_sources: int            # chunks retrieved
    answer_length: int          # characters in the answer
    timestamp: float = field(default_factory=time.time)

    @property
    def short_query(self) -> str:
        return self.query[:60] + "..." if len(self.query) > 60 else self.query


class MetricsTracker:
    """Collects and summarises RAG pipeline performance metrics."""

    def __init__(self, persist_path: Optional[str] = None):
        """
        Args:
            persist_path: If set, metrics are saved to this JSON file after
                          each record() call (e.g. "data/metrics.json")
        """
        self._records: List[QueryMetric] = []
        self.persist_path = Path(persist_path) if persist_path else None

        if self.persist_path and self.persist_path.exists():
            self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record(
        self,
        query: str,
        retrieve_time: float,
        generate_time: float,
        num_sources: int,
        answer_length: int,
    ) -> QueryMetric:
        """Record metrics for one completed query.

        The query text is hashed (SHA-256 prefix) before storage to avoid
        persisting user questions in plaintext.
        """
        # Store hashed query to protect user privacy
        query_hash = hashlib.sha256(query.encode()).hexdigest()[:16]
        metric = QueryMetric(
            query=query_hash,
            retrieve_time=retrieve_time,
            generate_time=generate_time,
            total_time=retrieve_time + generate_time,
            num_sources=num_sources,
            answer_length=answer_length,
        )
        self._records.append(metric)
        logger.debug(
            f"Recorded metric: total={metric.total_time:.2f}s "
            f"retrieve={metric.retrieve_time:.2f}s "
            f"generate={metric.generate_time:.2f}s"
        )

        if self.persist_path:
            self._save()

        return metric

    def summary(self) -> dict:
        """Return aggregate statistics across all recorded queries."""
        if not self._records:
            return {"total_queries": 0, "message": "No queries recorded yet"}

        total_times = [r.total_time for r in self._records]
        retrieve_times = [r.retrieve_time for r in self._records]
        generate_times = [r.generate_time for r in self._records]

        return {
            "total_queries": len(self._records),
            "total_time": {
                "mean":   round(mean(total_times), 3),
                "median": round(median(total_times), 3),
                "min":    round(min(total_times), 3),
                "max":    round(max(total_times), 3),
            },
            "retrieve_time": {
                "mean":   round(mean(retrieve_times), 3),
                "median": round(median(retrieve_times), 3),
            },
            "generate_time": {
                "mean":   round(mean(generate_times), 3),
                "median": round(median(generate_times), 3),
            },
            "avg_sources_per_query": round(
                mean(r.num_sources for r in self._records), 1
            ),
            "avg_answer_length": round(
                mean(r.answer_length for r in self._records), 0
            ),
        }

    def print_summary(self) -> None:
        """Print a formatted summary table to stdout."""
        s = self.summary()
        if s["total_queries"] == 0:
            print("No queries recorded yet.")
            return

        print("\n" + "=" * 50)
        print("RAG Pipeline Performance Summary")
        print("=" * 50)
        print(f"Total queries:        {s['total_queries']}")
        print(f"Avg total time:       {s['total_time']['mean']:.3f}s")
        print(f"Avg retrieve time:    {s['retrieve_time']['mean']:.3f}s")
        print(f"Avg generate time:    {s['generate_time']['mean']:.3f}s")
        print(f"Avg sources / query:  {s['avg_sources_per_query']}")
        print(f"Avg answer length:    {s['avg_answer_length']:.0f} chars")
        print("=" * 50 + "\n")

    def recent(self, n: int = 5) -> List[QueryMetric]:
        """Return the N most recent query metrics."""
        return self._records[-n:]

    def reset(self) -> None:
        """Clear all recorded metrics."""
        self._records = []
        if self.persist_path and self.persist_path.exists():
            self.persist_path.unlink()

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _save(self) -> None:
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        data = [asdict(r) for r in self._records]
        with open(self.persist_path, "w") as f:
            json.dump(data, f, indent=2)

    def _load(self) -> None:
        try:
            with open(self.persist_path) as f:
                data = json.load(f)
            self._records = [QueryMetric(**r) for r in data]
            logger.info(
                f"Loaded {len(self._records)} metrics from {self.persist_path}"
            )
        except Exception as e:
            logger.warning(f"Could not load metrics from {self.persist_path}: {e}")

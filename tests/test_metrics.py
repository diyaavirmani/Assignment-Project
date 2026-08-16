"""
Tests for src/utils/metrics.py (MetricsTracker)
"""
import json
import pytest
from pathlib import Path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tracker():
    from src.utils.metrics import MetricsTracker
    return MetricsTracker()  # in-memory only


@pytest.fixture
def tracker_with_records(tracker):
    tracker.record("What is gNB?",        retrieve_time=0.3, generate_time=1.2, num_sources=5, answer_length=400)
    tracker.record("Explain NG-RAN",      retrieve_time=0.2, generate_time=0.9, num_sources=4, answer_length=350)
    tracker.record("5G protocol stack",   retrieve_time=0.4, generate_time=1.5, num_sources=5, answer_length=500)
    return tracker


# ---------------------------------------------------------------------------
# record
# ---------------------------------------------------------------------------

class TestRecord:
    def test_record_increments_count(self, tracker):
        assert len(tracker._records) == 0
        tracker.record("q", retrieve_time=0.1, generate_time=1.0, num_sources=3, answer_length=200)
        assert len(tracker._records) == 1

    def test_record_stores_correct_values(self, tracker):
        m = tracker.record("test query", retrieve_time=0.25, generate_time=1.1,
                           num_sources=4, answer_length=300)
        # query is now stored as a SHA-256 hash prefix (privacy)
        assert len(m.query) == 16  # 16-char hex hash
        assert m.query != "test query"
        assert m.retrieve_time == 0.25
        assert m.generate_time == 1.1
        assert m.num_sources == 4
        assert m.answer_length == 300

    def test_total_time_is_sum(self, tracker):
        m = tracker.record("q", retrieve_time=0.3, generate_time=0.7,
                           num_sources=5, answer_length=100)
        assert abs(m.total_time - 1.0) < 1e-9

    def test_timestamp_is_positive(self, tracker):
        m = tracker.record("q", retrieve_time=0.1, generate_time=0.5,
                           num_sources=2, answer_length=50)
        assert m.timestamp > 0

    def test_multiple_records(self, tracker):
        for i in range(5):
            tracker.record(f"query {i}", retrieve_time=0.1, generate_time=0.5,
                           num_sources=3, answer_length=100)
        assert len(tracker._records) == 5


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------

class TestSummary:
    def test_empty_summary(self, tracker):
        s = tracker.summary()
        assert s["total_queries"] == 0

    def test_summary_total_queries(self, tracker_with_records):
        s = tracker_with_records.summary()
        assert s["total_queries"] == 3

    def test_summary_has_timing_keys(self, tracker_with_records):
        s = tracker_with_records.summary()
        assert "total_time" in s
        assert "retrieve_time" in s
        assert "generate_time" in s

    def test_summary_mean_is_reasonable(self, tracker_with_records):
        s = tracker_with_records.summary()
        assert s["total_time"]["mean"] > 0
        assert s["retrieve_time"]["mean"] > 0
        assert s["generate_time"]["mean"] > 0

    def test_summary_avg_sources(self, tracker_with_records):
        s = tracker_with_records.summary()
        # records have 5, 4, 5 sources → avg 4.67
        assert abs(s["avg_sources_per_query"] - (14 / 3)) < 0.1

    def test_summary_min_max(self, tracker_with_records):
        s = tracker_with_records.summary()
        assert s["total_time"]["min"] <= s["total_time"]["mean"] <= s["total_time"]["max"]


# ---------------------------------------------------------------------------
# recent
# ---------------------------------------------------------------------------

class TestRecent:
    def test_recent_returns_last_n(self, tracker_with_records):
        recent = tracker_with_records.recent(2)
        assert len(recent) == 2
        # query is hashed — just check it's a 16-char hex string
        assert len(recent[-1].query) == 16

    def test_recent_default_is_5(self, tracker):
        for i in range(7):
            tracker.record(f"q{i}", retrieve_time=0.1, generate_time=0.5,
                           num_sources=3, answer_length=100)
        recent = tracker.recent()
        assert len(recent) == 5


# ---------------------------------------------------------------------------
# reset
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_clears_records(self, tracker_with_records):
        tracker_with_records.reset()
        assert len(tracker_with_records._records) == 0

    def test_summary_empty_after_reset(self, tracker_with_records):
        tracker_with_records.reset()
        assert tracker_with_records.summary()["total_queries"] == 0


# ---------------------------------------------------------------------------
# persistence
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_saves_to_json(self, tmp_path):
        from src.utils.metrics import MetricsTracker
        path = tmp_path / "metrics.json"
        t = MetricsTracker(persist_path=str(path))
        t.record("q", retrieve_time=0.1, generate_time=0.5, num_sources=3, answer_length=100)
        assert path.exists()
        data = json.loads(path.read_text())
        assert len(data) == 1
        assert len(data[0]["query"]) == 16  # hashed

    def test_loads_from_existing_json(self, tmp_path):
        from src.utils.metrics import MetricsTracker
        path = tmp_path / "metrics.json"

        # Write seed data
        t1 = MetricsTracker(persist_path=str(path))
        t1.record("first", retrieve_time=0.2, generate_time=0.8, num_sources=4, answer_length=200)

        # Load in new instance
        t2 = MetricsTracker(persist_path=str(path))
        assert len(t2._records) == 1
        assert len(t2._records[0].query) == 16  # hashed

    def test_reset_deletes_file(self, tmp_path):
        from src.utils.metrics import MetricsTracker
        path = tmp_path / "metrics.json"
        t = MetricsTracker(persist_path=str(path))
        t.record("q", retrieve_time=0.1, generate_time=0.5, num_sources=2, answer_length=50)
        assert path.exists()
        t.reset()
        assert not path.exists()

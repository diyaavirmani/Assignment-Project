"""
Tests for src/api/main.py (FastAPI backend)

Uses FastAPI's TestClient so no live server is needed.
All heavy components (VectorStore, OllamaLLM) are mocked via app_state patching.
"""
import pytest
from collections import OrderedDict
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def mock_app_state():
    """
    Patch app_state before each test so no real VectorStore or Ollama
    connection is attempted.
    """
    from src.api import main as api_main

    mock_vs = MagicMock()
    mock_vs.get_stats.return_value = {
        "collection_name": "3gpp_specs",
        "total_chunks": 250,
        "persist_directory": "data/vectordb",
    }

    mock_metrics = MagicMock()
    mock_metrics.summary.return_value = {
        "total_queries": 5,
        "total_time": {"mean": 1.8, "median": 1.7, "min": 1.2, "max": 2.5},
        "retrieve_time": {"mean": 0.3, "median": 0.3},
        "generate_time": {"mean": 1.5, "median": 1.4},
        "avg_sources_per_query": 5.0,
        "avg_answer_length": 420.0,
    }

    api_main.app_state.vector_store = mock_vs
    api_main.app_state.metrics = mock_metrics
    api_main.app_state.ready = True
    api_main.app_state.sessions = OrderedDict()

    yield

    # Reset sessions between tests
    api_main.app_state.sessions = OrderedDict()


@pytest.fixture
def mock_rag_chain():
    """A RAGChain mock that returns a canned query result."""
    chain = MagicMock()
    chain.query.return_value = {
        "answer": "The gNB-CU handles RRC and PDCP protocols.",
        "sources": [
            {"source": "38300-g30.docx", "similarity": 0.92,
             "text": "The gNB-CU is a logical node..."},
        ],
        "context": "context text",
        "query_time": 1.8,
        "retrieve_time": 0.3,
        "generate_time": 1.5,
    }
    chain.get_history.return_value = []
    chain.stream_query.return_value = iter([
        {"type": "sources", "sources": []},
        {"type": "token", "token": "Hello"},
        {"type": "token", "token": " world"},
        {"type": "done", "query_time": 1.5},
    ])
    return chain


@pytest.fixture
def client(mock_rag_chain):
    """TestClient with RAGChain patched."""
    with patch("src.api.main.RAGChain", return_value=mock_rag_chain), \
         patch("src.api.main.DocumentRetriever"), \
         patch("src.api.main.OllamaLLM"):
        from src.api.main import app
        with TestClient(app, raise_server_exceptions=True) as c:
            yield c


# ---------------------------------------------------------------------------
# Root
# ---------------------------------------------------------------------------

class TestRoot:
    def test_root_returns_200(self, client):
        r = client.get("/")
        assert r.status_code == 200

    def test_root_has_docs_key(self, client):
        r = client.get("/")
        assert "docs" in r.json()


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

class TestHealth:
    def test_health_returns_200(self, client):
        with patch("src.api.main.OllamaLLM") as MockLLM:
            MockLLM.return_value.is_available.return_value = True
            r = client.get("/health")
        assert r.status_code == 200

    def test_health_has_status_field(self, client):
        with patch("src.api.main.OllamaLLM") as MockLLM:
            MockLLM.return_value.is_available.return_value = True
            r = client.get("/health")
        assert "status" in r.json()

    def test_health_has_components(self, client):
        with patch("src.api.main.OllamaLLM") as MockLLM:
            MockLLM.return_value.is_available.return_value = True
            data = client.get("/health").json()
        assert "components" in data
        assert "vector_store" in data["components"]
        assert "llm" in data["components"]

    def test_health_degraded_when_llm_unavailable(self, client):
        with patch("src.api.main._create_llm") as mock_create:
            mock_llm = MagicMock()
            mock_llm.is_available.return_value = False
            mock_create.return_value = mock_llm
            data = client.get("/health").json()
        assert data["status"] == "degraded"


# ---------------------------------------------------------------------------
# Stats & Metrics
# ---------------------------------------------------------------------------

class TestStats:
    def test_stats_returns_200(self, client):
        r = client.get("/stats")
        assert r.status_code == 200

    def test_stats_has_vector_store(self, client):
        data = client.get("/stats").json()
        assert "vector_store" in data
        assert "total_chunks" in data["vector_store"]

    def test_stats_has_active_sessions(self, client):
        data = client.get("/stats").json()
        assert "active_sessions" in data

    def test_metrics_returns_200(self, client):
        r = client.get("/metrics")
        assert r.status_code == 200

    def test_metrics_has_total_queries(self, client):
        data = client.get("/metrics").json()
        assert "total_queries" in data


# ---------------------------------------------------------------------------
# POST /query
# ---------------------------------------------------------------------------

class TestQuery:
    def test_query_returns_200(self, client):
        r = client.post("/query", json={"question": "What is gNB-CU?"})
        assert r.status_code == 200

    def test_query_returns_answer(self, client):
        data = client.post("/query", json={"question": "What is gNB-CU?"}).json()
        assert "answer" in data
        assert len(data["answer"]) > 0

    def test_query_returns_session_id(self, client):
        data = client.post("/query", json={"question": "What is gNB?"}).json()
        assert "session_id" in data
        assert len(data["session_id"]) > 0

    def test_query_same_session_id_reuses_session(self, client):
        r1 = client.post("/query", json={"question": "First question"}).json()
        sid = r1["session_id"]
        r2 = client.post("/query",
                         json={"question": "Follow-up", "session_id": sid}).json()
        assert r2["session_id"] == sid

    def test_query_returns_sources(self, client):
        data = client.post("/query", json={"question": "gNB architecture"}).json()
        assert isinstance(data["sources"], list)
        if data["sources"]:
            assert "source" in data["sources"][0]
            assert "similarity" in data["sources"][0]

    def test_query_returns_timing(self, client):
        data = client.post("/query", json={"question": "5G protocol"}).json()
        assert data["query_time"] >= 0
        assert data["retrieve_time"] >= 0
        assert data["generate_time"] >= 0

    def test_query_rejects_short_question(self, client):
        r = client.post("/query", json={"question": "hi"})
        assert r.status_code == 422

    def test_query_with_source_filter(self, client, mock_rag_chain):
        client.post("/query",
                    json={"question": "gNB", "source_filter": "38300"})
        mock_rag_chain.query.assert_called_once_with(
            question="gNB", source_filter="38300", top_k=None,
            domain=None, generation=None,
        )

    def test_query_with_top_k_override(self, client, mock_rag_chain):
        client.post("/query",
                    json={"question": "gNB architecture", "top_k": 3})
        mock_rag_chain.query.assert_called_once_with(
            question="gNB architecture", source_filter=None, top_k=3,
            domain=None, generation=None,
        )

    def test_query_503_when_not_ready(self, client):
        from src.api import main as api_main
        api_main.app_state.ready = False
        r = client.post("/query", json={"question": "What is gNB?"})
        assert r.status_code == 503
        api_main.app_state.ready = True


# ---------------------------------------------------------------------------
# POST /query/stream
# ---------------------------------------------------------------------------

class TestQueryStream:
    def test_stream_returns_200(self, client):
        r = client.post("/query/stream",
                        json={"question": "What is gNB-CU?"},
                        headers={"Accept": "text/event-stream"})
        assert r.status_code == 200

    def test_stream_content_type(self, client):
        r = client.post("/query/stream",
                        json={"question": "What is gNB?"})
        assert "text/event-stream" in r.headers.get("content-type", "")

    def test_stream_contains_data_lines(self, client):
        r = client.post("/query/stream",
                        json={"question": "What is gNB?"})
        lines = [l for l in r.text.splitlines() if l.startswith("data:")]
        assert len(lines) > 0


# ---------------------------------------------------------------------------
# Session history endpoints
# ---------------------------------------------------------------------------

class TestHistory:
    def _create_session(self, client) -> str:
        """Helper: create a session via /query and return the session_id."""
        data = client.post("/query",
                           json={"question": "What is gNB?"}).json()
        return data["session_id"]

    def test_get_history_returns_200(self, client):
        sid = self._create_session(client)
        r = client.get(f"/history/{sid}")
        assert r.status_code == 200

    def test_get_history_has_messages_key(self, client):
        sid = self._create_session(client)
        data = client.get(f"/history/{sid}").json()
        assert "messages" in data
        assert "message_count" in data

    def test_get_history_404_for_unknown_session(self, client):
        r = client.get("/history/nonexistent-session-id")
        assert r.status_code == 404

    def test_delete_history_returns_200(self, client):
        sid = self._create_session(client)
        r = client.delete(f"/history/{sid}")
        assert r.status_code == 200

    def test_delete_history_clears_messages(self, client, mock_rag_chain):
        sid = self._create_session(client)
        client.delete(f"/history/{sid}")
        mock_rag_chain.clear_history.assert_called_once()

    def test_delete_history_404_for_unknown_session(self, client):
        r = client.delete("/history/nonexistent-id")
        assert r.status_code == 404

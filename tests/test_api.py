"""
Tests for src/api/main.py (FastAPI backend)

Uses FastAPI's TestClient so no live server is needed.
All heavy components (vector store, LLM provider) are mocked.
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
    Patch app_state before each test so no real vector-store connection is
    attempted.
    """
    from src.api import main as api_main

    mock_vs = MagicMock()
    mock_vs.get_stats.return_value = {
        "index_name": "3gpp-rag",
        "namespace": "3gpp-specs",
        "total_chunks": 250,
        "total_vector_count": 250,
    }
    mock_vs.get_indexed_spec_numbers.return_value = {"38.300", "38.401"}

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
    api_main.app_state.retriever = MagicMock(vector_store=mock_vs)
    api_main.app_state.metrics = mock_metrics
    api_main.app_state.ready = True
    api_main.app_state.sessions = OrderedDict()

    yield

    # Reset sessions between tests
    api_main.app_state.sessions = OrderedDict()
    api_main.app_state.retriever = None


@pytest.fixture
def mock_rag_chain():
    """A RAGChain mock that returns a canned query result."""
    chain = MagicMock()
    chain.query.return_value = {
        "answer": "The gNB-CU handles RRC and PDCP protocols.",
        "sources": [
            {
                "source": "38300-g30.docx",
                "similarity": 0.92,
                "text": "The gNB-CU is a logical node...",
            },
        ],
        "context": "context text",
        "query_time": 1.8,
        "retrieve_time": 0.3,
        "generate_time": 1.5,
    }
    chain.get_history.return_value = []
    chain.stream_query.return_value = iter(
        [
            {"type": "sources", "sources": []},
            {"type": "token", "token": "Hello"},
            {"type": "token", "token": " world"},
            {"type": "done", "query_time": 1.5},
        ]
    )
    return chain


@pytest.fixture
def client(mock_rag_chain):
    """TestClient with RAGChain patched."""
    from src.api import main as api_main

    mock_llm = MagicMock()
    mock_llm.is_available.return_value = True
    with (
        patch("src.api.main.RAGChain", return_value=mock_rag_chain),
        patch("src.api.main.DocumentRetriever"),
        patch("src.api.main.create_vector_store", return_value=api_main.app_state.vector_store),
        patch("src.api.main.MetricsTracker", return_value=api_main.app_state.metrics),
        patch("src.api.main._create_llm", return_value=mock_llm),
    ):
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

    def test_root_reports_providers(self, client):
        r = client.get("/")
        data = r.json()
        assert "providers" in data
        assert data["providers"]["llm"]["provider"] == "openai"


class TestCorsConfig:
    def test_allowed_origins_include_local_and_env_origins(self, monkeypatch):
        from src.api.main import _allowed_origins_from_env

        monkeypatch.setenv(
            "ALLOWED_ORIGINS",
            "https://example.streamlit.app, https://second.streamlit.app/",
        )

        origins = _allowed_origins_from_env()

        assert "http://localhost:8501" in origins
        assert "http://127.0.0.1:8501" in origins
        assert "https://example.streamlit.app" in origins
        assert "https://second.streamlit.app" in origins

    def test_allowed_origins_ignores_wildcard(self, monkeypatch):
        from src.api.main import _allowed_origins_from_env

        monkeypatch.setenv("ALLOWED_ORIGINS", "*, https://example.streamlit.app")

        origins = _allowed_origins_from_env()

        assert "*" not in origins
        assert "https://example.streamlit.app" in origins


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


class TestHealth:
    def test_health_returns_200(self, client):
        r = client.get("/health")
        assert r.status_code == 200

    def test_health_has_status_field(self, client):
        r = client.get("/health")
        assert "status" in r.json()

    def test_health_has_components(self, client):
        data = client.get("/health").json()
        assert "components" in data
        assert "vector_store" in data["components"]
        assert "llm" in data["components"]
        assert "reranker" in data["components"]
        assert "evidence_gate" in data["components"]
        assert "verification" in data["components"]

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

    def test_stats_reports_provider_config_without_secrets(self, client):
        data = client.get("/stats").json()
        providers = data["providers"]
        assert providers["llm"]["provider"] == "openai"
        assert providers["vector_store"]["provider"] == "pinecone"
        assert providers["reranker"]["enabled"] is True
        assert providers["evidence_gate"]["score_source"] == "reranker"
        assert providers["verification"]["enabled"] is True
        provider_text = str(providers).lower()
        assert "api_key" not in provider_text
        assert "secret" not in provider_text

    def test_metrics_returns_200(self, client):
        r = client.get("/metrics")
        assert r.status_code == 200

    def test_metrics_has_total_queries(self, client):
        data = client.get("/metrics").json()
        assert "total_queries" in data


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------


class TestCatalog:
    @staticmethod
    def _spec_by_number(data, spec_number):
        return next(spec for spec in data["specs"] if spec["spec_number"] == spec_number)

    def test_catalog_marks_pinecone_indexed_specs(self, client):
        data = client.get("/catalog").json()

        assert data["total"] == 37
        assert self._spec_by_number(data, "38.300")["indexed"] is True
        assert self._spec_by_number(data, "38.401")["indexed"] is True
        assert self._spec_by_number(data, "38.211")["indexed"] is False

    def test_catalog_uses_vector_store_indexed_spec_capability(self, client):
        from src.api import main as api_main

        client.get("/catalog")

        api_main.app_state.vector_store.get_indexed_spec_numbers.assert_called_once()
        candidate_specs = list(
            api_main.app_state.vector_store.get_indexed_spec_numbers.call_args.args[0]
        )
        assert "38.300" in candidate_specs
        assert "38.401" in candidate_specs

    def test_catalog_preserves_domain_generation_filters(self, client):
        data = client.get("/catalog", params={"domain": "RAN", "generation": "5G"}).json()

        assert data["total"] < 37
        assert all(spec["domain"] == "RAN" for spec in data["specs"])
        assert all(spec["generation"] == "5G" for spec in data["specs"])
        assert self._spec_by_number(data, "38.300")["indexed"] is True
        assert self._spec_by_number(data, "38.401")["indexed"] is True

    def test_catalog_supports_legacy_chroma_capability(self, client):
        from src.api import main as api_main

        legacy_store = MagicMock()
        legacy_store.get_indexed_spec_numbers.return_value = {"36.300"}
        api_main.app_state.vector_store = legacy_store

        data = client.get("/catalog").json()

        assert self._spec_by_number(data, "36.300")["indexed"] is True
        assert self._spec_by_number(data, "38.300")["indexed"] is False

    def test_catalog_detection_failure_fails_safely(self, client):
        from src.api import main as api_main

        api_main.app_state.vector_store.get_indexed_spec_numbers.side_effect = RuntimeError(
            "stats unavailable"
        )

        data = client.get("/catalog").json()

        assert data["total"] == 37
        assert all(spec["indexed"] is False for spec in data["specs"])


# ---------------------------------------------------------------------------
# Session dependency sharing
# ---------------------------------------------------------------------------


class FakeSessionChain:
    def __init__(self, retriever, llm, max_history_turns):
        self.retriever = retriever
        self.llm = llm
        self.max_history_turns = max_history_turns
        self._history = []

    def get_history(self):
        return list(self._history)


class TestSessionDependencySharing:
    @staticmethod
    def _fake_llm():
        llm = MagicMock()
        llm.model = "test-model"
        return llm

    def test_new_sessions_get_distinct_chains_with_shared_retriever(self):
        from src.api import main as api_main

        shared_retriever = MagicMock()
        shared_retriever.vector_store = api_main.app_state.vector_store
        api_main.app_state.retriever = shared_retriever

        with (
            patch("src.api.main.RAGChain", side_effect=FakeSessionChain),
            patch("src.api.main._create_llm", return_value=self._fake_llm()),
        ):
            sid1, chain1 = api_main._get_or_create_session(None)
            sid2, chain2 = api_main._get_or_create_session(None)

        assert sid1 != sid2
        assert chain1 is not chain2
        assert chain1.retriever is shared_retriever
        assert chain2.retriever is shared_retriever
        assert chain1.retriever.vector_store is api_main.app_state.vector_store

    def test_multiple_sessions_do_not_create_per_session_retrievers_or_rerankers(self):
        from src.api import main as api_main

        shared_retriever = MagicMock()
        api_main.app_state.retriever = shared_retriever

        with (
            patch("src.api.main.RAGChain", side_effect=FakeSessionChain),
            patch("src.api.main._create_llm", return_value=self._fake_llm()),
            patch("src.api.main.DocumentRetriever") as document_retriever_cls,
            patch("src.core.retriever.CrossEncoderReranker") as reranker_cls,
        ):
            chains = [api_main._get_or_create_session(None)[1] for _ in range(5)]

        assert len({id(chain) for chain in chains}) == 5
        assert all(chain.retriever is shared_retriever for chain in chains)
        document_retriever_cls.assert_not_called()
        reranker_cls.assert_not_called()

    def test_conversation_history_stays_isolated_between_shared_retriever_sessions(self):
        from src.api import main as api_main

        api_main.app_state.retriever = MagicMock()

        with (
            patch("src.api.main.RAGChain", side_effect=FakeSessionChain),
            patch("src.api.main._create_llm", return_value=self._fake_llm()),
        ):
            _, chain1 = api_main._get_or_create_session(None)
            _, chain2 = api_main._get_or_create_session(None)

        chain1._history.append({"role": "user", "content": "first session only"})

        assert chain1.get_history() == [{"role": "user", "content": "first session only"}]
        assert chain2.get_history() == []

    def test_session_eviction_still_removes_expired_sessions(self):
        from src.api import main as api_main

        old_chain = MagicMock()
        fresh_chain = MagicMock()
        now = api_main.time.time()
        api_main.app_state.sessions = OrderedDict(
            [
                ("old", (old_chain, now - api_main.SESSION_TTL_SECONDS - 1)),
                ("fresh", (fresh_chain, now)),
            ]
        )

        api_main.app_state.evict_expired_sessions()

        assert "old" not in api_main.app_state.sessions
        assert api_main.app_state.sessions["fresh"][0] is fresh_chain


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
        r2 = client.post("/query", json={"question": "Follow-up", "session_id": sid}).json()
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
        client.post("/query", json={"question": "gNB", "source_filter": "38300"})
        mock_rag_chain.query.assert_called_once_with(
            question="gNB",
            source_filter="38300",
            top_k=None,
            domain=None,
            generation=None,
        )

    def test_query_with_top_k_override(self, client, mock_rag_chain):
        client.post("/query", json={"question": "gNB architecture", "top_k": 3})
        mock_rag_chain.query.assert_called_once_with(
            question="gNB architecture",
            source_filter=None,
            top_k=3,
            domain=None,
            generation=None,
        )

    def test_query_503_when_not_ready(self, client):
        from src.api import main as api_main

        api_main.app_state.ready = False
        r = client.post("/query", json={"question": "What is gNB?"})
        assert r.status_code == 503
        api_main.app_state.ready = True

    def test_query_503_when_shared_retriever_missing(self, client):
        from src.api import main as api_main

        api_main.app_state.retriever = None
        r = client.post("/query", json={"question": "What is gNB?"})
        assert r.status_code == 503


# ---------------------------------------------------------------------------
# POST /query/stream
# ---------------------------------------------------------------------------


class TestQueryStream:
    def test_stream_returns_200(self, client):
        r = client.post(
            "/query/stream",
            json={"question": "What is gNB-CU?"},
            headers={"Accept": "text/event-stream"},
        )
        assert r.status_code == 200

    def test_stream_content_type(self, client):
        r = client.post("/query/stream", json={"question": "What is gNB?"})
        assert "text/event-stream" in r.headers.get("content-type", "")

    def test_stream_contains_data_lines(self, client):
        r = client.post("/query/stream", json={"question": "What is gNB?"})
        lines = [l for l in r.text.splitlines() if l.startswith("data:")]
        assert len(lines) > 0


# ---------------------------------------------------------------------------
# Session history endpoints
# ---------------------------------------------------------------------------


class TestHistory:
    def _create_session(self, client) -> str:
        """Helper: create a session via /query and return the session_id."""
        data = client.post("/query", json={"question": "What is gNB?"}).json()
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

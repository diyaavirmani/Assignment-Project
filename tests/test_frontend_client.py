"""Tests for pure Streamlit frontend helpers."""

from src.frontend.client import (
    api_online_from_health,
    assistant_message_from_result,
    ensure_state_defaults,
    indexed_specs_from_catalog,
    resolve_api_base,
    status_render_kind,
)


def test_health_status_ok_is_online():
    assert api_online_from_health({"_http_status": 200, "status": "ok"}) is True


def test_health_request_failure_is_offline():
    assert api_online_from_health({}) is False
    assert api_online_from_health({"_http_status": 503, "status": "ok"}) is False
    assert api_online_from_health({"_http_status": 200, "status": "degraded"}) is False


def test_api_base_resolves_env_then_secret_then_local_fallback():
    assert resolve_api_base("https://api.example.com/") == "https://api.example.com"
    assert resolve_api_base("", "https://secret-api.example.com/") == "https://secret-api.example.com"
    assert resolve_api_base("", "") == "http://127.0.0.1:8000"


def test_catalog_indexed_entries_render_source_list():
    catalog = [
        {"spec_number": "38.300", "indexed": True},
        {"spec_number": "38.401", "indexed": True},
        {"spec_number": "23.501", "indexed": False},
    ]

    indexed = indexed_specs_from_catalog(catalog)

    assert [spec["spec_number"] for spec in indexed] == ["38.300", "38.401"]
    assert len(indexed) == 2


def test_successful_query_response_is_appended_shape():
    result = {
        "answer": "F1 connects gNB-CU and gNB-DU.",
        "sources": [{"source_id": "S1"}],
        "session_id": "session-1",
        "query_time": 1.2,
        "retrieve_time": 0.5,
        "generate_time": 0.4,
        "verify_time": 0.3,
        "answer_status": "answered",
        "evidence": {"sufficient": True},
        "verification": {"passed": True},
    }

    message = assistant_message_from_result(result)

    assert message["role"] == "assistant"
    assert message["content"] == result["answer"]
    assert message["result"]["answer_status"] == "answered"
    assert message["result"]["sources"] == [{"source_id": "S1"}]
    assert message["result"]["evidence"] == {"sufficient": True}
    assert message["result"]["verification"] == {"passed": True}
    assert message["answer_status"] == "answered"
    assert message["sources"] == [{"source_id": "S1"}]
    assert message["query_time"] == 1.2


def test_answered_response_appended_and_persisted():
    messages = [{"role": "user", "content": "What is a gNB in NG-RAN?"}]
    result = {
        "answer": "A gNB is the NG-RAN node that provides NR user-plane and control-plane protocol terminations.",
        "answer_status": "answered",
        "sources": [{"source_id": "S1"}],
        "evidence": {"sufficient": True},
        "verification": {"passed": True},
        "query_time": 1.0,
    }

    messages.append(assistant_message_from_result(result))
    ensure_state_defaults({"messages": messages}, {"messages": []})

    assert messages[-1]["role"] == "assistant"
    assert messages[-1]["content"] == result["answer"]
    assert messages[-1]["answer_status"] == "answered"
    assert messages[-1]["sources"] == [{"source_id": "S1"}]


def test_refused_evidence_response_appended_and_persisted():
    messages = [{"role": "user", "content": "Who won the 2022 FIFA World Cup?"}]
    result = {
        "answer": "I cannot answer because the indexed 3GPP corpus does not contain sufficient evidence.",
        "answer_status": "refused_evidence",
        "sources": [],
        "evidence": {"sufficient": False, "reason": "out_of_scope"},
        "verification": None,
    }

    messages.append(assistant_message_from_result(result))
    ensure_state_defaults({"messages": messages}, {"messages": []})

    assert messages[-1]["role"] == "assistant"
    assert messages[-1]["content"] == result["answer"]
    assert messages[-1]["answer_status"] == "refused_evidence"
    assert messages[-1]["evidence"] == result["evidence"]


def test_refused_verification_response_appended_and_persisted():
    messages = [{"role": "user", "content": "Explain why the E1 interface connects gNB-DU to the UPF."}]
    result = {
        "answer": "I cannot release this answer because verification failed.",
        "answer_status": "refused_verification",
        "sources": [{"source_id": "S1"}],
        "evidence": {"sufficient": True},
        "verification": {"passed": False, "reason": "unsupported_claims"},
    }

    messages.append(assistant_message_from_result(result))
    ensure_state_defaults({"messages": messages}, {"messages": []})

    assert messages[-1]["role"] == "assistant"
    assert messages[-1]["content"] == result["answer"]
    assert messages[-1]["answer_status"] == "refused_verification"
    assert messages[-1]["verification"] == result["verification"]


def test_missing_answer_response_raises_render_error():
    try:
        assistant_message_from_result({"answer_status": "answered"})
    except ValueError as exc:
        assert "missing answer" in str(exc)
    else:
        raise AssertionError("Expected missing answer to raise ValueError")


def test_answered_response_render_kind():
    assert status_render_kind("answered") == "answered"


def test_refused_evidence_render_kind():
    assert status_render_kind("refused_evidence") == "refused_evidence"


def test_refused_verification_render_kind():
    assert status_render_kind("refused_verification") == "refused_verification"
    assert status_render_kind("verifier_error") == "refused_verification"


def test_streamlit_rerun_defaults_do_not_clear_messages():
    state = {"messages": [{"role": "assistant", "content": "kept"}], "session_id": "abc"}
    ensure_state_defaults(state, {"messages": [], "session_id": None, "api_online": False})

    assert state["messages"] == [{"role": "assistant", "content": "kept"}]
    assert state["session_id"] == "abc"
    assert state["api_online"] is False

"""Pure frontend helpers for Streamlit API state and response handling."""

from __future__ import annotations

from typing import Any, Dict, List


def ensure_state_defaults(state: Dict[str, Any], defaults: Dict[str, Any]) -> None:
    """Initialize missing state keys without overwriting existing values."""
    for key, value in defaults.items():
        if key not in state:
            state[key] = value


def resolve_api_base(env_value: Any = None, secret_value: Any = None) -> str:
    """Resolve the backend API URL for Streamlit Cloud and local development."""
    configured = env_value or secret_value or "http://127.0.0.1:8000"
    return str(configured).rstrip("/")


def api_online_from_health(health: Dict[str, Any]) -> bool:
    """Return True only when /health returned HTTP 200 and JSON status ok."""
    try:
        http_status = int(health.get("_http_status", 0))
    except (TypeError, ValueError):
        http_status = 0
    return http_status == 200 and health.get("status") == "ok"


def indexed_specs_from_catalog(catalog: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return catalog entries that are actually indexed."""
    return [spec for spec in catalog if spec.get("indexed") is True]


def assistant_message_from_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a QueryResponse payload into the persisted chat message shape."""
    if not isinstance(result, dict):
        raise ValueError("Query response is not a JSON object.")
    if "answer" not in result:
        raise ValueError("Query response is missing answer.")

    normalized = {
        "answer": result.get("answer", ""),
        "sources": result.get("sources") or [],
        "session_id": result.get("session_id"),
        "query_time": result.get("query_time"),
        "retrieve_time": result.get("retrieve_time"),
        "generate_time": result.get("generate_time"),
        "verify_time": result.get("verify_time"),
        "answer_status": result.get("answer_status") or "answered",
        "evidence": result.get("evidence"),
        "verification": result.get("verification"),
    }
    message = {
        "role": "assistant",
        "content": normalized["answer"],
        "result": normalized,
    }
    message.update(
        {
            "answer_status": normalized["answer_status"],
            "sources": normalized["sources"],
            "evidence": normalized["evidence"],
            "verification": normalized["verification"],
            "query_time": normalized["query_time"],
            "retrieve_time": normalized["retrieve_time"],
            "generate_time": normalized["generate_time"],
            "verify_time": normalized["verify_time"],
        }
    )
    return message


def status_render_kind(answer_status: str) -> str:
    """Map backend answer_status to the frontend rendering family."""
    if answer_status == "answered":
        return "answered"
    if answer_status == "refused_evidence":
        return "refused_evidence"
    if answer_status in {"refused_verification", "verifier_error"}:
        return "refused_verification"
    return "pending"

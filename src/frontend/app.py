"""
Streamlit frontend for the 3GPP Verified RAG Assistant.

Run:
    streamlit run src/frontend/app.py

Requires the FastAPI backend:
    uvicorn src.api.main:app --reload --port 8000
"""

from __future__ import annotations

import html
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import requests
import streamlit as st

FRONTEND_DIR = Path(__file__).resolve().parent
if str(FRONTEND_DIR) not in sys.path:
    sys.path.insert(0, str(FRONTEND_DIR))

from styles import inject_styles
from client import (
    api_online_from_health,
    assistant_message_from_result,
    ensure_state_defaults,
    indexed_specs_from_catalog,
    resolve_api_base,
    status_render_kind,
)

logger = logging.getLogger(__name__)


def _streamlit_secret(name: str) -> Optional[str]:
    try:
        return st.secrets.get(name)
    except Exception:
        return None


API_BASE = resolve_api_base(os.getenv("API_BASE_URL"), _streamlit_secret("API_BASE_URL"))
INDEXED_SPEC_DEFAULT = "All indexed"
EXAMPLE_QUESTIONS = [
    "What is a gNB in NG-RAN?",
    "What is the F1 interface used for?",
    "How are gNB-CU and gNB-DU connected?",
    "Compare NG and Xn interfaces.",
]


st.set_page_config(
    page_title="3GPP Verified RAG",
    page_icon=":material/verified:",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_styles()


def init_state() -> None:
    defaults = {
        "session_id": None,
        "messages": [],
        "api_online": False,
        "filter_generation": "All",
        "filter_domain": "All",
        "filter_spec": INDEXED_SPEC_DEFAULT,
        "latest_result": None,
        "last_response": None,
        "catalog": [],
        "health": {},
    }
    ensure_state_defaults(st.session_state, defaults)


init_state()


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------


def _get_json(path: str, timeout: int = 5) -> Dict[str, Any]:
    response = requests.get(f"{API_BASE}{path}", timeout=timeout)
    response.raise_for_status()
    data = response.json()
    if isinstance(data, dict):
        data["_http_status"] = response.status_code
    return data


def check_api() -> Dict[str, Any]:
    try:
        health = _get_json("/health", timeout=3)
        logger.debug(
            "Health check: http_status=%s json_status=%s",
            health.get("_http_status"),
            health.get("status"),
        )
        return health
    except Exception as exc:
        logger.info("Health check failed: %s", exc)
        return {}


def get_catalog() -> List[Dict[str, Any]]:
    try:
        specs = _get_json("/catalog").get("specs", [])
        logger.debug("Catalog loaded: count=%s", len(specs))
        return specs
    except Exception as exc:
        logger.info("Catalog unavailable: %s", exc)
        return []


def get_stats() -> Dict[str, Any]:
    try:
        return _get_json("/stats")
    except Exception as exc:
        logger.info("Stats unavailable: %s", exc)
        return {}


def get_eval_results() -> Dict[str, Any]:
    try:
        return _get_json("/eval")
    except Exception as exc:
        logger.info("Evaluation results unavailable: %s", exc)
        return {"available": False}


def _spec_to_source_filter(spec_choice: str) -> Optional[str]:
    if not spec_choice or spec_choice == INDEXED_SPEC_DEFAULT:
        return None
    return spec_choice.replace("TS ", "").replace(".", "")


def _active_filters() -> tuple[Optional[str], Optional[str]]:
    domain = st.session_state.filter_domain
    generation = st.session_state.filter_generation
    return (
        None if domain == "All" else domain,
        None if generation == "All" else generation,
    )


def post_query(question: str, source_filter: Optional[str], top_k: int) -> Dict[str, Any]:
    domain, generation = _active_filters()
    payload = {
        "question": question,
        "session_id": st.session_state.session_id,
        "source_filter": source_filter,
        "top_k": top_k,
        "domain": domain,
        "generation": generation,
    }
    response = requests.post(f"{API_BASE}/query", json=payload, timeout=90)
    response.raise_for_status()
    data = response.json()
    logger.info(
        "query response: http=%s status=%s has_answer=%s",
        response.status_code,
        data.get("answer_status") if isinstance(data, dict) else None,
        bool(data.get("answer")) if isinstance(data, dict) else False,
    )
    if not isinstance(data, dict):
        raise ValueError("Query response is not a JSON object.")
    return data


def stream_query(question: str, source_filter: Optional[str], top_k: int):
    domain, generation = _active_filters()
    payload = {
        "question": question,
        "session_id": st.session_state.session_id,
        "source_filter": source_filter,
        "top_k": top_k,
        "domain": domain,
        "generation": generation,
    }
    with requests.post(
        f"{API_BASE}/query/stream",
        json=payload,
        stream=True,
        timeout=140,
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if line and line.startswith(b"data: "):
                yield json.loads(line[6:])


def clear_backend_history() -> None:
    if st.session_state.session_id:
        try:
            requests.delete(f"{API_BASE}/history/{st.session_state.session_id}", timeout=5)
        except Exception as exc:
            logger.info("History clear failed: %s", exc)


def reset_session(delete_backend_history: bool = False) -> None:
    if delete_backend_history:
        clear_backend_history()
    st.session_state.session_id = None
    st.session_state.messages = []
    st.session_state.latest_result = None
    st.session_state.last_response = None


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def escape(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def fmt_score(value: Any, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "-"


def fmt_percent(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "-"
    if 0 <= number <= 1:
        number *= 100
    return f"{number:.1f}%"


def fmt_seconds(value: Any) -> str:
    try:
        return f"{float(value):.2f}s"
    except (TypeError, ValueError):
        return "-"


def short_excerpt(text: str, limit: int = 420) -> str:
    compact = " ".join((text or "").split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rstrip() + "..."


def spec_label(source: Dict[str, Any]) -> str:
    spec_number = source.get("spec_number")
    if spec_number and spec_number != "unknown":
        return f"TS {spec_number}"
    return "3GPP source"


def source_title(source: Dict[str, Any]) -> str:
    title = source.get("spec_title")
    if title and title != "unknown":
        return str(title)
    return str(source.get("source", "Source"))


def answer_status(result: Optional[Dict[str, Any]]) -> str:
    if not result:
        return "pending"
    return result.get("answer_status") or "answered"


def score_source_label(value: Optional[str]) -> str:
    if value == "reranker":
        return "Reranker"
    if value == "vector":
        return "Vector"
    return "-"


def recursive_find(data: Any, keys: Iterable[str]) -> Optional[Any]:
    wanted = {key.lower() for key in keys}
    if isinstance(data, dict):
        for key, value in data.items():
            normalized = key.lower().replace(" ", "_").replace("-", "_")
            if normalized in wanted:
                return value
        for value in data.values():
            found = recursive_find(value, wanted)
            if found is not None:
                return found
    elif isinstance(data, list):
        for item in data:
            found = recursive_find(item, wanted)
            if found is not None:
                return found
    return None


def nested_value(data: Dict[str, Any], path: Iterable[str]) -> Optional[Any]:
    current: Any = data
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def first_metric(
    summary: Dict[str, Any], keys: Iterable[str], *paths: Iterable[str]
) -> Optional[Any]:
    found = recursive_find(summary, keys)
    if found is not None:
        return found
    for path in paths:
        found = nested_value(summary, path)
        if found is not None:
            return found
    return None


def evaluation_metrics(eval_data: Dict[str, Any]) -> List[tuple[str, str]]:
    metric_labels = [
        "Correct Answer Rate",
        "Correct Refusal Rate",
        "Unsafe Answer Rate",
        "Citation Validity",
        "Hit@5",
    ]
    if not eval_data.get("available"):
        return [(label, "-") for label in metric_labels]

    summary = eval_data.get("summary", {})
    return [
        (
            "Correct Answer Rate",
            fmt_percent(
                first_metric(
                    summary,
                    ["correct_answer_rate", "answer_correct_rate"],
                    ("answer", "pass_rate"),
                )
            ),
        ),
        (
            "Correct Refusal Rate",
            fmt_percent(
                first_metric(
                    summary,
                    ["correct_refusal_rate"],
                    ("refusal", "answer_refusal_rate"),
                    ("refusal", "retrieval_refusal_rate"),
                )
            ),
        ),
        (
            "Unsafe Answer Rate",
            fmt_percent(
                first_metric(
                    summary,
                    ["unsafe_answer_rate", "unsafe_rate", "unsupported_answer_escape_rate"],
                )
            ),
        ),
        (
            "Citation Validity",
            fmt_percent(
                first_metric(
                    summary,
                    ["citation_validity", "citation_validity_rate"],
                )
            ),
        ),
        (
            "Hit@5",
            fmt_percent(
                first_metric(
                    summary,
                    ["hit_at_5", "hit@5", "avg_hit_rate_at_k"],
                    ("retrieval", "avg_hit_rate_at_k"),
                )
            ),
        ),
    ]


def evaluation_suite_caption(eval_data: Dict[str, Any]) -> str:
    cases = eval_data.get("cases") or []
    if cases:
        return f"Results from the current curated {len(cases)}-case evaluation suite."

    config = eval_data.get("config") or {}
    case_count = recursive_find(config, ["total_cases", "case_count", "num_cases"])
    if case_count:
        return f"Results from the current curated {case_count}-case evaluation suite."

    return "Results from the current curated evaluation suite exposed by /eval."


def catalog_matches_active_filters(spec: Dict[str, Any]) -> bool:
    domain, generation = _active_filters()
    if domain and spec.get("domain") != domain:
        return False
    if generation and spec.get("generation") != generation:
        return False
    selected = st.session_state.filter_spec
    if selected != INDEXED_SPEC_DEFAULT and f"TS {spec.get('spec_number')}" != selected:
        return False
    return True


def export_chat_markdown() -> str:
    lines = ["# 3GPP Verified RAG Chat", ""]
    if st.session_state.session_id:
        lines.extend([f"Session: `{st.session_state.session_id}`", ""])
    for message in st.session_state.messages:
        role = "User" if message["role"] == "user" else "Assistant"
        lines.extend([f"## {role}", message["content"], ""])
        result = message.get("result") or {}
        if message["role"] == "assistant" and result.get("sources"):
            lines.append("Sources:")
            for source in result["sources"]:
                source_id = source.get("source_id") or "S?"
                lines.append(f"- [{source_id}] {spec_label(source)} - {source_title(source)}")
            lines.append("")
    return "\n".join(lines)


def search_current_session(term: str) -> List[Dict[str, str]]:
    query = term.strip().lower()
    if not query:
        return []
    matches = []
    for message in st.session_state.messages:
        content = message.get("content", "")
        if query in content.lower():
            matches.append({"role": message.get("role", ""), "content": content})
    return matches[:4]


# ---------------------------------------------------------------------------
# Render helpers
# ---------------------------------------------------------------------------


def render_top_toolbar() -> None:
    _, export_col = st.columns([0.80, 0.20])
    with export_col:
        if st.session_state.messages:
            st.download_button(
                "Export chat",
                data=export_chat_markdown(),
                file_name="3gpp-verified-rag-chat.md",
                mime="text/markdown",
                use_container_width=True,
            )
        else:
            st.button("Export chat", disabled=True, use_container_width=True)


def render_sidebar(
    health: Dict[str, Any],
    stats: Dict[str, Any],
    catalog: List[Dict[str, Any]],
) -> tuple[int, bool, Optional[str]]:
    indexed_specs = indexed_specs_from_catalog(catalog)
    indexed_options = [f"TS {spec['spec_number']}" for spec in indexed_specs]
    spec_options = [INDEXED_SPEC_DEFAULT] + indexed_options

    if st.session_state.filter_spec not in spec_options:
        st.session_state.filter_spec = INDEXED_SPEC_DEFAULT

    with st.sidebar:
        st.markdown(
            """
<div class="sidebar-brand">
  <div class="brand-mark"></div>
  <div>
    <div class="brand-title">3GPP Verified RAG</div>
    <div class="brand-subtitle">Telecom Standards Assistant</div>
  </div>
</div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("+ New session", type="primary", use_container_width=True):
            reset_session(delete_backend_history=False)
            st.rerun()

        search_term = st.text_input(
            "Search current chat",
            placeholder="Search current chat...",
            label_visibility="collapsed",
            key="session_search",
        )
        search_matches = search_current_session(search_term)
        if search_term.strip():
            if search_matches:
                st.caption(
                    f"{len(search_matches)} match{'es' if len(search_matches) != 1 else ''} in current session"
                )
                for match in search_matches:
                    st.caption(f"{match['role'].title()}: {short_excerpt(match['content'], 86)}")
            else:
                st.caption("No matches in the current session.")

        online = api_online_from_health(health)
        st.session_state.api_online = online

        st.divider()
        st.markdown("### Corpus")
        st.markdown(
            f'<div class="corpus-count"><strong>{len(indexed_specs)} active</strong> indexed specifications</div>',
            unsafe_allow_html=True,
        )
        if indexed_specs:
            for spec in indexed_specs:
                st.markdown(
                    f"""<div class="indexed-spec-card">
  <div class="indexed-spec-icon">□</div>
  <div class="indexed-spec-body">
    <div class="indexed-spec-head"><strong>TS {escape(spec["spec_number"])}</strong><span><i></i>Indexed</span></div>
    <div class="indexed-spec-title">{escape(spec.get("title", ""))}</div>
  </div>
</div>""",
                    unsafe_allow_html=True,
                )
        else:
            st.caption("No indexed standards reported by /catalog.")

        with st.expander("Browse all corpus", expanded=False):
            if catalog:
                for spec in catalog:
                    active = "active" if spec.get("indexed") else ""
                    state = (
                        "Indexed" if spec.get("indexed") else "Available in catalog / not indexed"
                    )
                    st.markdown(
                        f"""<div class="catalog-row {active}">
  <div class="catalog-main">TS {escape(spec["spec_number"])}<span>{escape(state)}</span></div>
  <div class="catalog-title">{escape(spec.get("title", ""))}</div>
  <small>{escape(spec.get("generation", ""))} {escape(spec.get("domain", ""))}</small>
</div>""",
                        unsafe_allow_html=True,
                    )
            else:
                st.caption("Catalog unavailable.")

        st.divider()
        st.markdown("### Filters")
        st.radio(
            "Generation",
            ["All", "5G", "LTE"],
            horizontal=True,
            key="filter_generation",
        )
        st.radio(
            "Domain",
            ["All", "RAN", "CORE"],
            horizontal=True,
            key="filter_domain",
        )
        st.selectbox(
            "Specification",
            spec_options,
            key="filter_spec",
            help="Specification uses the existing source_filter request field.",
        )
        filtered_indexed = [spec for spec in indexed_specs if catalog_matches_active_filters(spec)]
        if online and not filtered_indexed:
            st.warning("No indexed corpus matches the selected filters.")

        st.divider()
        st.markdown("### Retrieval")
        top_k = st.slider("Evidence depth", min_value=1, max_value=15, value=5)
        use_streaming = st.toggle(
            "Streaming response",
            value=False,
            help="Uses the existing /query/stream endpoint.",
        )
        source_filter = _spec_to_source_filter(st.session_state.filter_spec)

        st.divider()
        st.markdown("### Current session")
        if st.session_state.session_id:
            st.caption(f"`{st.session_state.session_id}`")
        else:
            st.caption("Not started")
        st.caption(f"{len(st.session_state.messages)} messages")
        if st.button("Clear conversation", use_container_width=True):
            reset_session(delete_backend_history=True)
            st.rerun()

    return top_k, use_streaming, source_filter


def render_hero() -> None:
    st.markdown(
        """
<section class="hero">
  <div class="hero-orb-wrap">
    <div class="hero-orb"></div>
    <span class="sparkle s1"></span>
    <span class="sparkle s2"></span>
    <span class="sparkle s3"></span>
  </div>
  <h1>Hello</h1>
  <h2>Ask the standards with verified evidence.</h2>
  <p>Get accurate answers from 3GPP specifications with source citations and verification.</p>
</section>
        """,
        unsafe_allow_html=True,
    )


def render_query_composer() -> Optional[str]:
    with st.form("welcome_query_form", clear_on_submit=True):
        question = st.text_area(
            "Ask a question...",
            placeholder="Ask a question...",
            label_visibility="collapsed",
            height=122,
        )
        controls = st.columns([0.58, 0.16, 0.26])
        with controls[0]:
            st.empty()
        with controls[1]:
            st.empty()
        with controls[2]:
            submitted = st.form_submit_button(
                "Send",
                icon=":material/send:",
                type="primary",
                use_container_width=True,
            )
    if submitted and question.strip():
        return question.strip()
    return None


def render_capability_cards() -> None:
    cards = [
        (
            "◇",
            "Evidence Gating",
            "Answers are generated only when sufficient supporting 3GPP evidence is retrieved.",
        ),
        (
            "□",
            "Source Citations",
            "Answers include traceable citations to retrieved specification evidence.",
        ),
        (
            "◎",
            "Verification",
            "Generated answers are checked for citation validity and grounding before release.",
        ),
    ]
    cols = st.columns(3)
    for col, (icon, title, copy) in zip(cols, cards):
        with col:
            st.markdown(
                f"""
<div class="capability-card">
  <div class="capability-icon">{escape(icon)}</div>
  <div>
    <div class="capability-title">{escape(title)}</div>
    <div class="capability-copy">{escape(copy)}</div>
  </div>
</div>
                """,
                unsafe_allow_html=True,
            )


def render_example_buttons() -> None:
    cols = st.columns(4)
    for col, question in zip(cols, EXAMPLE_QUESTIONS):
        if col.button(question, use_container_width=True):
            st.session_state["pending_question"] = question
            st.rerun()


def component_ok(health: Dict[str, Any], name: str) -> bool:
    return (health.get("components", {}).get(name) or {}).get("status") == "ok"


def render_system_status_strip(health: Dict[str, Any], stats: Dict[str, Any]) -> None:
    online = health.get("status") in {"ok", "degraded"}
    providers = stats.get("providers", {})
    items = [
        ("OpenAI", "Healthy", component_ok(health, "llm")),
        ("Pinecone", "Healthy", component_ok(health, "vector_store")),
        (
            "Reranker",
            "Healthy",
            component_ok(health, "reranker") and providers.get("reranker", {}).get("enabled", True),
        ),
        (
            "Verifier",
            "Healthy",
            component_ok(health, "verification")
            and providers.get("verification", {}).get("enabled", True),
        ),
    ]
    status_items = []
    for label, healthy_label, ok in items:
        state = "ok" if online and ok else "danger"
        if online:
            text = f"{label} {healthy_label}" if ok else f"{label} Unavailable"
        else:
            text = "API Offline"
        status_items.append(
            f'<div class="status-item"><span class="status-glyph"></span>{escape(text)}<span class="dot {state}"></span></div>'
        )
    st.markdown(
        f"""
<div class="status-panel">
  <div class="section-label">SYSTEM STATUS</div>
  <div class="status-items">{''.join(status_items)}</div>
</div>
        """,
        unsafe_allow_html=True,
    )
    if not online:
        st.info(
            f"API Offline. Streamlit is connected to `{API_BASE}`; start the backend and refresh."
        )


def render_answer_state(result: Dict[str, Any]) -> None:
    status = answer_status(result)
    if status == "answered":
        st.markdown(
            '<div class="answer-state ok"><strong>VERIFIED</strong><span>Grounded in indexed 3GPP evidence.</span></div>',
            unsafe_allow_html=True,
        )
    elif status == "refused_evidence":
        st.markdown(
            '<div class="answer-state blocked"><strong>EVIDENCE BLOCKED</strong><span>Insufficient supporting evidence was found in the indexed 3GPP corpus.</span></div>',
            unsafe_allow_html=True,
        )
    elif status in {"refused_verification", "verifier_error"}:
        st.markdown(
            '<div class="answer-state failed"><strong>VERIFICATION BLOCKED</strong><span>Relevant evidence was found, but the generated response could not be fully verified.</span></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="answer-state blocked"><strong>PENDING</strong><span>Verification metadata is incomplete.</span></div>',
            unsafe_allow_html=True,
        )


def render_answer_metadata(result: Dict[str, Any]) -> None:
    evidence = result.get("evidence") or {}
    verification = result.get("verification") or {}
    sources = result.get("sources") or []
    citations = len({source.get("source_id") for source in sources if source.get("source_id")})
    claims_text = "-"
    if verification:
        claims_text = (
            f"{verification.get('supported_claims', 0)}/{verification.get('total_claims', 0)}"
        )
    st.markdown(
        f"""
<div class="message-meta">
  <span class="badge">Evidence {escape(evidence.get("qualifying_docs", "-"))}/{escape(evidence.get("total_docs", "-"))}</span>
  <span class="badge">Top {escape(fmt_score(evidence.get("top_score")))}</span>
  <span class="badge">{citations} citations</span>
  <span class="badge">Claims {escape(claims_text)}</span>
  <span class="badge">{escape(fmt_seconds(result.get("query_time")))}</span>
</div>
        """,
        unsafe_allow_html=True,
    )


def render_sources(sources: List[Dict[str, Any]]) -> None:
    if not sources:
        return
    st.markdown("##### Sources")
    for source in sources:
        source_id = source.get("source_id") or "S?"
        rank_before = source.get("rank_before_reranking")
        rank_after = source.get("rank_after_reranking")
        rank_line = "-"
        if rank_before is not None or rank_after is not None:
            rank_line = f"#{rank_before or '-'} -> #{rank_after or '-'}"
        st.markdown(
            f"""
<div class="source-card">
  <div><span class="source-id">[{escape(source_id)}]</span> <strong>{escape(spec_label(source))}</strong></div>
  <div class="source-title">{escape(source_title(source))}</div>
  <div class="source-metrics">
    <span><b>Relevance</b>{escape(fmt_score(source.get("reranker_score")))}</span>
    <span><b>Vector</b>{escape(fmt_score(source.get("vector_similarity", source.get("similarity"))))}</span>
    <span><b>Rank</b>{escape(rank_line)}</span>
  </div>
</div>
            """,
            unsafe_allow_html=True,
        )
        with st.expander(f"Source excerpt [{source_id}]", expanded=False):
            st.write(short_excerpt(source.get("text", "")))
            st.caption(f"Source file: {source.get('source', '-')}")
            st.caption(
                f"Domain: {source.get('domain', '-')} - Generation: {source.get('generation', '-')}"
            )


def render_evidence_gate(evidence: Optional[Dict[str, Any]]) -> None:
    with st.expander("Evidence Gate", expanded=False):
        if not evidence:
            st.caption("Evidence metadata unavailable.")
            return
        decision = "Passed" if evidence.get("sufficient") else "Blocked"
        fields = [
            ("Score source", score_source_label(evidence.get("score_source"))),
            ("Top relevance", fmt_score(evidence.get("top_score"), 4)),
            ("Mean score", fmt_score(evidence.get("mean_score"), 4)),
            (
                "Strong chunks",
                f"{evidence.get('qualifying_docs', '-')} / {evidence.get('total_docs', '-')}",
            ),
            ("Decision", decision),
            ("Reason", evidence.get("reason", "-")),
        ]
        for label, value in fields:
            st.caption(f"{label}: {value}")


def render_verification(verification: Optional[Dict[str, Any]], status: str) -> None:
    with st.expander("Verification", expanded=False):
        if not verification:
            if status == "refused_evidence":
                st.caption("Not released: evidence gate blocked generation before verification.")
            else:
                st.caption("Verification metadata unavailable.")
            return

        unsupported = verification.get("unsupported_claims") or []
        contradicted = verification.get("contradicted_claims") or []
        release_state = "Verified" if verification.get("passed") else "Not released"
        citation_state = "Valid" if verification.get("citation_valid") else "Invalid"

        st.caption(f"Status: {release_state}")
        st.caption(
            f"Claims supported: {verification.get('supported_claims', 0)} / {verification.get('total_claims', 0)}"
        )
        st.caption(f"Citation validity: {citation_state}")
        st.caption(f"Unsupported claims: {len(unsupported)}")
        st.caption(f"Contradictions: {len(contradicted)}")
        st.caption(f"Reason: {verification.get('reason', '-')}")
        if verification.get("detail"):
            st.caption(f"Detail: {verification.get('detail')}")
        if unsupported:
            st.write("Unsupported claims")
            for claim in unsupported:
                st.caption(claim)
        if contradicted:
            st.write("Contradictions")
            for claim in contradicted:
                st.caption(claim)


def render_response_details(result: Dict[str, Any]) -> None:
    with st.expander("Response details", expanded=False):
        st.caption(f"Retrieval: {fmt_seconds(result.get('retrieve_time'))}")
        st.caption(f"Generation: {fmt_seconds(result.get('generate_time'))}")
        st.caption(f"Verification: {fmt_seconds(result.get('verify_time'))}")
        st.caption(f"Total: {fmt_seconds(result.get('query_time'))}")


def render_assistant_result(content: str, result: Dict[str, Any]) -> None:
    status = answer_status(result)
    kind = status_render_kind(status)
    if kind == "answered":
        st.markdown(content)
        render_answer_state(result)
    elif kind == "refused_evidence":
        render_answer_state(result)
        st.warning(content or result.get("answer", "The evidence gate blocked this answer."))
    elif kind == "refused_verification":
        render_answer_state(result)
        st.error(content or result.get("answer", "The answer verifier blocked this response."))
    else:
        st.markdown(content)
        render_answer_state(result)

    render_answer_metadata(result)
    if kind == "answered":
        render_sources(result.get("sources") or [])
    render_evidence_gate(result.get("evidence"))
    render_verification(result.get("verification"), status)
    render_response_details(result)


def render_message(role: str, content: str, result: Optional[Dict[str, Any]] = None) -> None:
    with st.chat_message(role):
        if role == "assistant" and result:
            render_assistant_result(content, result)
        else:
            st.markdown(content)


def render_evaluation_panel(eval_data: Dict[str, Any]) -> None:
    with st.expander("Evaluation", expanded=False):
        cols = st.columns(4)
        for index, (label, value) in enumerate(evaluation_metrics(eval_data)):
            with cols[index % 4]:
                st.markdown(
                    f"""
<div class="metric-card">
  <div class="metric-label">{escape(label)}</div>
  <div class="metric-value">{escape(value)}</div>
</div>
                    """,
                    unsafe_allow_html=True,
                )
        if eval_data.get("available"):
            st.caption(evaluation_suite_caption(eval_data))
        else:
            st.caption("Evaluation artifact unavailable from /eval. No paid evaluation was run.")


def query_and_render(
    question: str, use_streaming: bool, source_filter: Optional[str], top_k: int
) -> None:
    st.session_state.messages.append({"role": "user", "content": question})
    result: Dict[str, Any]

    try:
        if use_streaming:
            full_answer: List[str] = []
            result = {
                "sources": [],
                "evidence": None,
                "verification": None,
                "query_time": None,
                "retrieve_time": None,
                "generate_time": None,
                "verify_time": None,
                "answer_status": "pending",
            }
            for chunk in stream_query(question, source_filter, top_k):
                if chunk.get("type") == "sources":
                    result["sources"] = chunk.get("sources", [])
                    result["evidence"] = chunk.get("evidence")
                elif chunk.get("type") == "token":
                    full_answer.append(chunk.get("token", ""))
                elif chunk.get("type") == "done":
                    result.update(
                        {
                            "query_time": chunk.get("query_time"),
                            "answer_status": chunk.get("answer_status"),
                            "evidence": chunk.get("evidence", result.get("evidence")),
                            "verification": chunk.get("verification"),
                            "session_id": chunk.get("session_id"),
                        }
                    )
                elif chunk.get("type") == "error":
                    raise RuntimeError(chunk.get("detail", "Streaming failed."))
            result["answer"] = "".join(full_answer) or result.get("answer", "")
        else:
            with st.spinner("Retrieving and verifying 3GPP evidence..."):
                result = post_query(question, source_filter, top_k)

        assistant_message = assistant_message_from_result(result)
    except requests.HTTPError as exc:
        logger.error("API request failed: %s", exc)
        st.error("API request failed. Please check backend health.")
        return
    except Exception as exc:
        logger.exception("Frontend could not render successful API response: %s", exc)
        st.error("The API responded, but the frontend could not render the result.")
        return

    response_result = assistant_message["result"]
    if response_result.get("session_id"):
        st.session_state.session_id = response_result["session_id"]
    st.session_state.messages.append(assistant_message)
    st.session_state.latest_result = response_result
    st.session_state.last_response = response_result
    logger.debug(
        "Assistant message appended: answer_status=%s total_messages=%s",
        response_result.get("answer_status"),
        len(st.session_state.messages),
    )


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------


health = check_api()
api_online = api_online_from_health(health)
catalog = get_catalog() if api_online else []
stats = get_stats() if api_online else {}
st.session_state.health = health
st.session_state.catalog = catalog

top_k, use_streaming, source_filter = render_sidebar(health, stats, catalog)
render_top_toolbar()

pending = st.session_state.pop("pending_question", None)
question: Optional[str] = None

if not st.session_state.messages:
    render_hero()
    question = render_query_composer()
    render_example_buttons()
else:
    st.markdown('<div class="chat-shell">', unsafe_allow_html=True)
    for message in st.session_state.messages:
        render_message(message["role"], message["content"], message.get("result"))
    st.markdown("</div>", unsafe_allow_html=True)
    question = st.chat_input(
        "Ask a question...",
        key="conversation_query",
    )

if pending:
    question = pending

if question:
    query_and_render(question, use_streaming, source_filter, top_k)
    st.rerun()

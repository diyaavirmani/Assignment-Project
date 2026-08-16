"""
Streamlit UI for the 3GPP RAG Assistant

Calls the FastAPI backend at http://localhost:8000

Run:
    streamlit run src/frontend/app.py

Requires the FastAPI backend to be running:
    uvicorn src.api.main:app --reload --port 8000
"""
import html
import json
import logging
import os
import time
import requests
import streamlit as st

logger = logging.getLogger(__name__)

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="3GPP RAG Assistant",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }
    .sub-header {
        color: #888;
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
    }
    .source-card {
        background: #1e1e2e;
        border-left: 3px solid #4f8ef7;
        border-radius: 4px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.5rem;
        font-size: 0.85rem;
    }
    .similarity-badge {
        background: #2a4a7f;
        color: #9ec5fe;
        padding: 2px 8px;
        border-radius: 10px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .timing-bar {
        color: #888;
        font-size: 0.8rem;
        margin-top: 0.4rem;
    }
    .status-ok    { color: #4ade80; }
    .status-warn  { color: #facc15; }
    .status-error { color: #f87171; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

def init_state():
    defaults = {
        "session_id": None,
        "messages": [],          # [{role, content, sources, timing}]
        "api_online": False,
        "filter_generation": "All",
        "filter_domain": "All",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def check_api() -> dict:
    """Call /health and return the response dict, or {} on failure."""
    try:
        r = requests.get(f"{API_BASE}/health", timeout=3)
        return r.json()
    except Exception:
        return {}


def _active_filters():
    """Return (domain, generation) as strings or None when set to 'All'."""
    domain = st.session_state.filter_domain
    generation = st.session_state.filter_generation
    return (
        None if domain == "All" else domain,
        None if generation == "All" else generation,
    )


def post_query(question: str, source_filter: str, top_k: int) -> dict:
    """POST /query and return parsed JSON."""
    domain, generation = _active_filters()
    payload = {
        "question": question,
        "session_id": st.session_state.session_id,
        "source_filter": source_filter or None,
        "top_k": top_k,
        "domain": domain,
        "generation": generation,
    }
    r = requests.post(f"{API_BASE}/query", json=payload, timeout=60)
    r.raise_for_status()
    return r.json()


def stream_query(question: str, source_filter: str, top_k: int):
    """
    POST /query/stream and yield parsed SSE chunks.
    Yields dicts: {type: sources|token|done|error, ...}
    """
    domain, generation = _active_filters()
    payload = {
        "question": question,
        "session_id": st.session_state.session_id,
        "source_filter": source_filter or None,
        "top_k": top_k,
        "domain": domain,
        "generation": generation,
    }
    with requests.post(
        f"{API_BASE}/query/stream",
        json=payload,
        stream=True,
        timeout=120,
    ) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if line and line.startswith(b"data: "):
                yield json.loads(line[6:])


def get_catalog() -> list:
    """GET /catalog and return the list of specs."""
    try:
        r = requests.get(f"{API_BASE}/catalog", timeout=5)
        return r.json().get("specs", [])
    except Exception:
        return []


def get_stats() -> dict:
    try:
        r = requests.get(f"{API_BASE}/stats", timeout=5)
        return r.json()
    except Exception:
        return {}


def clear_session_history():
    if st.session_state.session_id:
        try:
            requests.delete(
                f"{API_BASE}/history/{st.session_state.session_id}",
                timeout=5,
            )
        except Exception:
            pass
    st.session_state.messages = []
    st.session_state.session_id = None


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## ⚙️ Settings")

    # API status
    health = check_api()
    if health.get("status") == "ok":
        st.session_state.api_online = True
        st.markdown('<span class="status-ok">● API Online</span>', unsafe_allow_html=True)
    elif health.get("status") == "degraded":
        st.session_state.api_online = True
        st.markdown('<span class="status-warn">● API Degraded</span>', unsafe_allow_html=True)
    else:
        st.session_state.api_online = False
        st.markdown('<span class="status-error">● API Offline</span>', unsafe_allow_html=True)
        st.warning("Start the API:\n```\nuvicorn src.api.main:app --reload\n```")

    st.divider()

    # -----------------------------------------------------------------------
    # Spec filters
    # -----------------------------------------------------------------------
    st.markdown("### 📡 Spec Filters")
    st.caption("Restrict retrieval to a subset of indexed specs.")

    generation_choice = st.radio(
        "Generation",
        options=["All", "5G", "LTE"],
        index=["All", "5G", "LTE"].index(st.session_state.filter_generation),
        horizontal=True,
        key="filter_generation",
    )

    domain_choice = st.radio(
        "Domain",
        options=["All", "RAN", "CORE"],
        index=["All", "RAN", "CORE"].index(st.session_state.filter_domain),
        horizontal=True,
        key="filter_domain",
    )

    # Show which specs are active
    if st.session_state.api_online:
        catalog = get_catalog()
        if catalog:
            active_gen = None if generation_choice == "All" else generation_choice
            active_dom = None if domain_choice == "All" else domain_choice
            visible = [
                s for s in catalog
                if (not active_gen or s["generation"] == active_gen)
                and (not active_dom or s["domain"] == active_dom)
            ]
            indexed = [s for s in visible if s.get("indexed")]
            not_indexed = [s for s in visible if not s.get("indexed")]

            with st.expander(f"Active specs ({len(indexed)} indexed)", expanded=False):
                if indexed:
                    for s in indexed:
                        st.markdown(f"✅ **TS {s['spec_number']}** — {s['generation']} {s['domain']}")
                if not_indexed:
                    st.markdown("---")
                    st.caption("Not yet indexed:")
                    for s in not_indexed:
                        st.markdown(f"⬜ TS {s['spec_number']} — {s['generation']} {s['domain']}")

    st.divider()

    # -----------------------------------------------------------------------
    # Query settings
    # -----------------------------------------------------------------------
    st.markdown("### Query Settings")
    top_k = st.slider("Chunks to retrieve", min_value=1, max_value=15, value=5)
    source_filter = st.text_input(
        "Filter by filename",
        placeholder="e.g. 38300",
        help="Only retrieve chunks from documents whose filename contains this string",
    )
    use_streaming = st.toggle("Streaming mode", value=True,
                              help="Stream tokens as they are generated")

    st.divider()

    # Component health detail
    if health.get("components"):
        st.markdown("### Component Status")
        icons = {"ok": "✅", "degraded": "⚠️", "unavailable": "❌"}
        for name, info in health["components"].items():
            icon = icons.get(info.get("status", "unavailable"), "❓")
            st.markdown(f"{icon} **{name}**: {info.get('detail', '')}")

    st.divider()

    # Session controls
    st.markdown("### Session")
    if st.session_state.session_id:
        st.caption(f"ID: `{st.session_state.session_id[:16]}...`")
    if st.button("🗑️ Clear conversation", use_container_width=True):
        clear_session_history()
        st.rerun()

    # Stats panel
    if st.session_state.api_online:
        with st.expander("📊 Performance Stats"):
            s = get_stats()
            if s:
                m = s.get("metrics", {})
                if m.get("total_queries", 0) > 0:
                    st.metric("Total queries", m["total_queries"])
                    st.metric("Avg total time",
                              f"{m['total_time']['mean']:.2f}s")
                    st.metric("Avg retrieve time",
                              f"{m['retrieve_time']['mean']:.2f}s")
                    st.metric("Avg generate time",
                              f"{m['generate_time']['mean']:.2f}s")
                    vs = s.get("vector_store", {})
                    st.metric("Chunks indexed",
                              vs.get("total_chunks", "—"))
                else:
                    st.caption("No queries recorded yet.")

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------

st.markdown('<div class="main-header">📡 3GPP RAG Assistant</div>', unsafe_allow_html=True)

# Build active filter indicator
_gen = st.session_state.filter_generation
_dom = st.session_state.filter_domain
_filter_parts = [p for p in [_gen if _gen != "All" else "", _dom if _dom != "All" else ""] if p]
_filter_label = " · ".join(_filter_parts) if _filter_parts else "All specs"
st.markdown(
    f'<div class="sub-header">Ask questions about 3GPP specifications — '
    f'AI-powered, cited answers &nbsp;|&nbsp; '
    f'<b>Scope: {_filter_label}</b></div>',
    unsafe_allow_html=True,
)

# Example query chips
example_queries = [
    "What is the gNB-CU architecture?",
    "Explain the F1 interface",
    "How does handover work in 5G?",
    "What is the difference between SA and NSA?",
]

cols = st.columns(len(example_queries))
for col, q in zip(cols, example_queries):
    if col.button(q, use_container_width=True):
        st.session_state["pending_question"] = q

st.divider()

# ---------------------------------------------------------------------------
# Render existing messages
# ---------------------------------------------------------------------------

def render_sources(sources: list):
    if not sources:
        return
    with st.expander(f"📚 Sources ({len(sources)})", expanded=False):
        for s in sources:
            spec_badge = ""
            if s.get("spec_number") and s["spec_number"] != "unknown":
                gen = s.get("generation", "")
                dom = s.get("domain", "")
                spec_badge = (
                    f' <span class="similarity-badge">'
                    f'TS {s["spec_number"]} · {gen} {dom}'
                    f'</span>'
                )
            safe_source = html.escape(s["source"])
            safe_text = html.escape(s["text"][:100]) + ("..." if len(s["text"]) > 100 else "")
            st.markdown(
                f'<div class="source-card">'
                f'<b>{safe_source}</b>{spec_badge} '
                f'<span class="similarity-badge">{s["similarity"]:.0%}</span><br>'
                f'<small>{safe_text}</small>'
                f'</div>',
                unsafe_allow_html=True,
            )


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            render_sources(msg.get("sources", []))
            if msg.get("timing"):
                t = msg["timing"]
                st.markdown(
                    f'<div class="timing-bar">⏱ total={t["query_time"]}s '
                    f'| retrieve={t["retrieve_time"]}s '
                    f'| generate={t["generate_time"]}s</div>',
                    unsafe_allow_html=True,
                )

# ---------------------------------------------------------------------------
# Handle input
# ---------------------------------------------------------------------------

# Pick up question from example chip or text input
question = st.chat_input(
    "Ask a question about 3GPP specs...",
    disabled=not st.session_state.api_online,
)
if "pending_question" in st.session_state:
    question = st.session_state.pop("pending_question")

if question:
    if not st.session_state.api_online:
        st.error("API is offline. Start it with: `uvicorn src.api.main:app --reload`")
        st.stop()

    # Show user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Generate answer
    with st.chat_message("assistant"):
        if use_streaming:
            # Streaming path
            token_placeholder = st.empty()
            sources = []
            full_answer = []
            timing = {}

            try:
                for chunk in stream_query(question, source_filter, top_k):
                    if chunk["type"] == "sources":
                        sources = chunk.get("sources", [])
                        # Update session_id from next done chunk — handled below
                    elif chunk["type"] == "token":
                        full_answer.append(chunk["token"])
                        token_placeholder.markdown("".join(full_answer) + "▌")
                    elif chunk["type"] == "done":
                        token_placeholder.markdown("".join(full_answer))
                        timing = {
                            "query_time": chunk.get("query_time", 0),
                            "retrieve_time": 0,
                            "generate_time": chunk.get("query_time", 0),
                        }
                        if chunk.get("session_id"):
                            st.session_state.session_id = chunk["session_id"]
                    elif chunk["type"] == "error":
                        st.error(chunk.get("detail", "Unknown streaming error"))
                        st.stop()

                answer = "".join(full_answer)

            except Exception as e:
                logger.error(f"Streaming error: {e}")
                st.error("Streaming failed. Please try again.")
                st.stop()

        else:
            # Blocking path
            with st.spinner("Searching specifications..."):
                try:
                    result = post_query(question, source_filter, top_k)
                except requests.HTTPError as e:
                    logger.error(f"API error: {e}")
                    st.error("API request failed. Please try again.")
                    st.stop()
                except Exception as e:
                    logger.error(f"Request failed: {e}")
                    st.error("Request failed. Please try again.")
                    st.stop()

            answer = result["answer"]
            sources = result.get("sources", [])
            timing = {
                "query_time": result["query_time"],
                "retrieve_time": result["retrieve_time"],
                "generate_time": result["generate_time"],
            }
            st.session_state.session_id = result["session_id"]
            st.markdown(answer)

        render_sources(sources)
        st.markdown(
            f'<div class="timing-bar">⏱ total={timing.get("query_time", 0):.2f}s</div>',
            unsafe_allow_html=True,
        )

    # Save to session state
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
        "timing": timing,
    })

# ---------------------------------------------------------------------------
# Legal disclaimer (footer)
# ---------------------------------------------------------------------------

st.divider()
st.caption(
    "**Disclaimer:** This tool is an independent, non-commercial project and is "
    "not affiliated with, endorsed by, or sponsored by 3GPP, ETSI, or any 3GPP "
    "Organizational Partner. 3GPP specifications are publicly available from "
    "[3gpp.org](https://www.3gpp.org). All specification content remains the "
    "intellectual property of the respective rights holders. Answers are "
    "AI-generated and may contain errors — always verify against the official "
    "specifications."
)

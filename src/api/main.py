"""
FastAPI backend for the 3GPP RAG Assistant

Endpoints:
    POST /query              - Submit a question, get answer + sources
    POST /query/stream       - Same but streams tokens via SSE
    GET  /history/{id}       - Get conversation history for a session
    DELETE /history/{id}     - Clear conversation history for a session
    GET  /health             - Health check with per-component status
    GET  /stats              - Vector store + performance metrics
    GET  /metrics            - Aggregated query performance stats

Run:
    uvicorn src.api.main:app --reload --port 8000

Interactive docs:
    http://localhost:8000/docs
"""
import json
import os
import re
import uuid
import time
import logging
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse

from src.api.models import (
    QueryRequest, QueryResponse, SourceDocument,
    HistoryResponse, HistoryMessage,
    HealthResponse, StatsResponse, EvalResponse, ErrorResponse,
    CatalogResponse, CatalogSpec,
)
from src.core.rag_chain import RAGChain
from src.core.vector_store import VectorStore
from src.core.llm import OllamaLLM
from src.core.groq_llm import GroqLLM
from src.core.retriever import DocumentRetriever
from src.core.spec_catalog import CATALOG, infer_spec_from_filename
from src.utils.logger import setup_logger
from src.utils.metrics import MetricsTracker
from src.config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Application state
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Security constants
# ---------------------------------------------------------------------------

MAX_SESSIONS = 200           # Limit total active sessions (memory DoS protection)
SESSION_TTL_SECONDS = 3600   # Sessions expire after 1 hour of inactivity
MAX_QUESTION_LENGTH = 1000   # Limit user input length

# Rate limiting: per-IP request tracking
_rate_limit_window = 60       # seconds
_rate_limit_max = 20          # max queries per IP per window
_rate_limit_max_ips = 10_000  # max tracked IPs (prevents memory exhaustion)
_rate_limit_store: OrderedDict[str, list] = OrderedDict()


def _get_client_ip(request: Request) -> str:
    """Extract client IP from request, handling proxied and test scenarios."""
    # Prefer X-Forwarded-For when behind a reverse proxy (Streamlit Cloud)
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


def _check_rate_limit(request: Request):
    """Enforce per-IP rate limiting on query endpoints.

    The store is bounded to _rate_limit_max_ips entries with LRU eviction
    to prevent memory exhaustion from IP rotation attacks.
    """
    client_ip = _get_client_ip(request)
    now = time.time()

    # LRU eviction: if store is full and this is a new IP, drop oldest
    if client_ip not in _rate_limit_store:
        while len(_rate_limit_store) >= _rate_limit_max_ips:
            _rate_limit_store.popitem(last=False)
        _rate_limit_store[client_ip] = []

    # Prune expired timestamps for this IP
    _rate_limit_store[client_ip] = [
        t for t in _rate_limit_store[client_ip]
        if now - t < _rate_limit_window
    ]
    # Move to end (LRU)
    _rate_limit_store.move_to_end(client_ip)

    if len(_rate_limit_store[client_ip]) >= _rate_limit_max:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Max {_rate_limit_max} queries per minute."
        )
    _rate_limit_store[client_ip].append(now)


class AppState:
    """Holds shared application state (initialised once at startup)."""

    def __init__(self):
        self.rag_chain: Optional[RAGChain] = None
        self.vector_store: Optional[VectorStore] = None
        self.metrics: Optional[MetricsTracker] = None
        # Session store: session_id -> (RAGChain, last_access_time)
        # LRU-bounded with TTL eviction
        self.sessions: OrderedDict[str, tuple] = OrderedDict()
        self.ready: bool = False

    def evict_expired_sessions(self):
        """Remove sessions older than TTL and enforce max count."""
        now = time.time()
        expired = [
            sid for sid, (_, last_access) in self.sessions.items()
            if now - last_access > SESSION_TTL_SECONDS
        ]
        for sid in expired:
            del self.sessions[sid]
        # If still over limit, evict oldest
        while len(self.sessions) > MAX_SESSIONS:
            self.sessions.popitem(last=False)


app_state = AppState()


# ---------------------------------------------------------------------------
# Lifespan (startup / shutdown)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise heavy components once at startup."""
    setup_logger(level=settings.log_level)
    logger.info("Starting 3GPP RAG Assistant API...")

    try:
        app_state.vector_store = VectorStore(
            persist_directory=settings.vector_db_path,
            collection_name=settings.collection_name,
        )
        stats = app_state.vector_store.get_stats()
        logger.info(f"Vector store ready: {stats['total_chunks']} chunks")

        app_state.metrics = MetricsTracker(persist_path="data/api_metrics.json")
        app_state.ready = True
        logger.info("API ready.")
    except Exception as e:
        logger.error(f"Startup error: {e}")
        app_state.ready = False

    yield

    logger.info("Shutting down API.")


# ---------------------------------------------------------------------------
# App definition
# ---------------------------------------------------------------------------

# Disable interactive docs in production (expose internal schema)
_enable_docs = os.getenv("ENABLE_API_DOCS", "false").lower() in ("1", "true", "yes")

app = FastAPI(
    title="3GPP RAG Assistant",
    description=(
        "AI-powered question answering over 3GPP technical specifications. "
        "Runs locally with Ollama or in the cloud with Groq."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs" if _enable_docs else None,
    redoc_url="/redoc" if _enable_docs else None,
    openapi_url="/openapi.json" if _enable_docs else None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",    # Local Streamlit dev
    ],
    allow_origin_regex=r"https://3gpp-rag-assistant\.streamlit\.app",  # Only our Streamlit Cloud app
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type"],
)


# ---------------------------------------------------------------------------
# Middleware: request timing
# ---------------------------------------------------------------------------

@app.middleware("http")
async def add_security_and_timing_headers(request: Request, call_next):
    """Add security headers and request timing to every response."""
    t0 = time.time()
    response = await call_next(request)
    # Timing
    response.headers["X-Process-Time"] = f"{time.time() - t0:.3f}s"
    # Security headers
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    response.headers["Content-Security-Policy"] = "default-src 'self'; frame-ancestors 'none'"
    return response


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_llm():
    """Create the right LLM instance based on settings.llm_provider."""
    if settings.llm_provider == "groq":
        logger.info(f"Using Groq LLM: {settings.groq_model}")
        return GroqLLM(
            model=settings.groq_model,
            api_key=settings.groq_api_key or None,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
        )
    else:
        logger.info(f"Using Ollama LLM: {settings.llm_model}")
        return OllamaLLM(
            model=settings.llm_model,
            base_url=settings.ollama_base_url,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
        )


def _get_or_create_session(session_id: Optional[str]) -> tuple[str, RAGChain]:
    """Return (session_id, chain), creating a new session if needed.

    Sessions are stored with a last-access timestamp and evicted
    when they exceed MAX_SESSIONS or SESSION_TTL_SECONDS.
    """
    # Evict stale sessions periodically
    app_state.evict_expired_sessions()

    if session_id and session_id in app_state.sessions:
        chain, _ = app_state.sessions[session_id]
        # Update last-access time and move to end (LRU)
        app_state.sessions.move_to_end(session_id)
        app_state.sessions[session_id] = (chain, time.time())
        return session_id, chain

    new_id = session_id or str(uuid.uuid4())
    chain = RAGChain(
        retriever=DocumentRetriever(
            vector_store=app_state.vector_store,
            top_k=settings.top_k_results,
        ),
        llm=_create_llm(),
        max_history_turns=settings.max_history_length,
    )
    app_state.sessions[new_id] = (chain, time.time())
    logger.info(f"Created session: {new_id} (active: {len(app_state.sessions)})")
    return new_id, chain


def _check_ready():
    if not app_state.ready:
        raise HTTPException(
            status_code=503,
            detail="Service not ready. Check vector store and configuration."
        )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post(
    "/query",
    response_model=QueryResponse,
    summary="Ask a question",
    description="Submit a natural language question and receive a grounded answer with citations.",
    tags=["Query"],
)
async def query(request: QueryRequest, req: Request):
    """
    Ask a question about 3GPP specifications.

    Pass `session_id` from a previous response to continue the conversation.
    Omit it to start a new session.
    """
    _check_ready()
    _check_rate_limit(req)

    session_id, chain = _get_or_create_session(request.session_id)

    try:
        result = chain.query(
            question=request.question,
            source_filter=request.source_filter,
            top_k=request.top_k,
            domain=request.domain,
            generation=request.generation,
        )
    except RuntimeError as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=503, detail="LLM generation failed. Check /health for status.")

    # Record metrics
    app_state.metrics.record(
        query=request.question,
        retrieve_time=result["retrieve_time"],
        generate_time=result["generate_time"],
        num_sources=len(result["sources"]),
        answer_length=len(result["answer"]),
    )

    return QueryResponse(
        answer=result["answer"],
        sources=[SourceDocument(**s) for s in result["sources"]],
        session_id=session_id,
        query_time=result["query_time"],
        retrieve_time=result["retrieve_time"],
        generate_time=result["generate_time"],
    )


@app.post(
    "/query/stream",
    summary="Ask a question (streaming)",
    description="Same as /query but streams the answer token-by-token via Server-Sent Events.",
    tags=["Query"],
)
async def query_stream(request: QueryRequest, req: Request):
    """
    Stream a response token-by-token using Server-Sent Events (SSE).

    The client receives a series of `data:` lines:
      - `data: {"type":"sources","sources":[...]}`  (sent first)
      - `data: {"type":"token","token":"Hello"}`    (one per LLM token)
      - `data: {"type":"done","session_id":"...","query_time":2.1}`
    """
    _check_ready()
    _check_rate_limit(req)

    session_id, chain = _get_or_create_session(request.session_id)

    async def event_generator():
        try:
            for chunk in chain.stream_query(
                question=request.question,
                source_filter=request.source_filter,
                top_k=request.top_k,
                domain=request.domain,
                generation=request.generation,
            ):
                if chunk["type"] == "done":
                    chunk["session_id"] = session_id
                yield f"data: {json.dumps(chunk)}\n\n"
        except RuntimeError as e:
            logger.error(f"Stream failed: {e}")
            yield f"data: {json.dumps({'type':'error','detail':'Generation failed. Check /health.'})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get(
    "/history/{session_id}",
    response_model=HistoryResponse,
    summary="Get conversation history",
    tags=["Session"],
)
async def get_history(session_id: str):
    """Return the full conversation history for a session."""
    if session_id not in app_state.sessions:
        raise HTTPException(status_code=404, detail="Session not found.")

    chain, _ = app_state.sessions[session_id]
    history = chain.get_history()

    return HistoryResponse(
        session_id=session_id,
        messages=[HistoryMessage(role=m["role"], content=m["content"]) for m in history],
        message_count=len(history),
    )


@app.delete(
    "/history/{session_id}",
    summary="Clear conversation history",
    tags=["Session"],
)
async def clear_history(session_id: str):
    """Clear the conversation history for a session (session remains active)."""
    if session_id not in app_state.sessions:
        raise HTTPException(status_code=404, detail="Session not found.")

    chain, _ = app_state.sessions[session_id]
    chain.clear_history()
    return {"message": "History cleared."}


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    tags=["System"],
)
async def health():
    """
    Check the health of all system components.

    Returns overall status and per-component detail.
    """
    components = {}

    # Vector store
    try:
        stats = app_state.vector_store.get_stats()
        if stats["total_chunks"] == 0:
            components["vector_store"] = {"status": "degraded", "detail": "Index is empty"}
        else:
            components["vector_store"] = {
                "status": "ok",
                "detail": f"{stats['total_chunks']} chunks indexed"
            }
    except Exception as e:
        logger.error(f"Vector store health check failed: {e}")
        components["vector_store"] = {"status": "unavailable", "detail": "connection failed"}

    # LLM (Ollama or Groq)
    try:
        llm = _create_llm()
        if llm.is_available():
            model_name = settings.groq_model if settings.llm_provider == "groq" else settings.llm_model
            components["llm"] = {
                "status": "ok",
                "detail": f"provider={settings.llm_provider}, model={model_name}"
            }
        else:
            components["llm"] = {
                "status": "degraded",
                "detail": f"LLM not available ({settings.llm_provider})"
            }
    except Exception as e:
        logger.error(f"LLM health check failed: {e}")
        components["llm"] = {"status": "unavailable", "detail": "connection failed"}

    # Embeddings
    components["embeddings"] = {
        "status": "ok",
        "detail": f"model={settings.embedding_model} (local)"
    }

    overall = (
        "ok" if all(c["status"] == "ok" for c in components.values())
        else "degraded"
    )

    return HealthResponse(
        status=overall,
        components=components,
        timestamp=datetime.utcnow(),
    )


@app.get(
    "/stats",
    response_model=StatsResponse,
    summary="System statistics",
    tags=["System"],
)
async def stats():
    """Return vector store stats, active session count, and query metrics."""
    _check_ready()

    vs_stats = app_state.vector_store.get_stats()
    metrics_summary = app_state.metrics.summary()

    return StatsResponse(
        vector_store=vs_stats,
        active_sessions=len(app_state.sessions),
        metrics=metrics_summary,
    )


@app.get(
    "/metrics",
    summary="Query performance metrics",
    tags=["System"],
)
async def metrics():
    """Return aggregated query performance metrics."""
    _check_ready()
    return app_state.metrics.summary()


@app.get(
    "/eval",
    response_model=EvalResponse,
    summary="Latest evaluation results",
    description=(
        "Returns the most recent pipeline evaluation results (retrieval quality, "
        "answer quality, and latency benchmarks). "
        "Run `python scripts/eval_retrieval.py [--full]` to refresh."
    ),
    tags=["System"],
)
async def eval_results():
    """
    Return the latest saved evaluation results from data/eval_results.json.

    If no results exist yet, returns available=False with empty fields.
    Regenerate by running: python scripts/eval_retrieval.py --full
    """
    eval_path = Path("data/eval_results.json")
    if not eval_path.exists():
        return EvalResponse(available=False)

    try:
        with open(eval_path) as f:
            data = json.load(f)
        return EvalResponse(
            available=True,
            evaluated_at=data.get("evaluated_at"),
            config=data.get("config"),
            summary=data.get("summary"),
            cases=data.get("cases"),
        )
    except Exception as e:
        logger.error(f"Failed to read eval results: {e}")
        return EvalResponse(available=False)


@app.get(
    "/catalog",
    response_model=CatalogResponse,
    summary="List all supported specs",
    description=(
        "Returns every spec in the catalog with its metadata. "
        "Use domain and generation query params to filter the list."
    ),
    tags=["System"],
)
async def catalog(
    domain: Optional[str] = None,     # "RAN" or "CORE"
    generation: Optional[str] = None,  # "5G" or "LTE"
):
    """Return the full spec catalog, optionally filtered by domain/generation.

    The ``indexed`` field reflects whether a spec's chunks are present in the
    current vector store (based on spec_number metadata).
    """
    # Determine which spec numbers are present in the vector store
    indexed_specs: set = set()
    if app_state.ready and app_state.vector_store:
        try:
            # Probe each catalog spec with a targeted where-filter query
            # (avoids fetching all 40K+ chunks just to discover which specs exist)
            for entry in CATALOG:
                result = app_state.vector_store.collection.get(
                    limit=1,
                    where={"spec_number": entry["spec_number"]},
                    include=[],
                )
                if result and result.get("ids"):
                    indexed_specs.add(entry["spec_number"])
        except Exception:
            pass  # Best-effort; if it fails just mark all as not indexed

    entries = CATALOG
    if domain:
        if domain not in ("RAN", "CORE"):
            raise HTTPException(status_code=400, detail="domain must be 'RAN' or 'CORE'")
        entries = [e for e in entries if e["domain"] == domain]
    if generation:
        if generation not in ("5G", "LTE"):
            raise HTTPException(status_code=400, detail="generation must be '5G' or 'LTE'")
        entries = [e for e in entries if e["generation"] == generation]

    specs = [
        CatalogSpec(
            spec_number=e["spec_number"],
            title=e["title"],
            domain=e["domain"],
            generation=e["generation"],
            description=e["description"],
            indexed=e["spec_number"] in indexed_specs,
        )
        for e in entries
    ]

    return CatalogResponse(total=len(specs), specs=specs)


@app.get("/", include_in_schema=False)
async def root():
    return {
        "name": "3GPP RAG Assistant API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "eval": "/eval",
    }

"""
Pydantic schemas for the 3GPP RAG Assistant API

Request and response models for all endpoints.
"""
from pydantic import BaseModel, ConfigDict, Field
from typing import List, Literal, Optional
from datetime import datetime


# ---------------------------------------------------------------------------
# Query endpoint
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    """Request body for POST /query"""
    model_config = ConfigDict(extra="forbid")

    question: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="Natural language question about 3GPP specifications",
        examples=["What is the gNB-CU architecture?"]
    )
    session_id: Optional[str] = Field(
        default=None,
        max_length=36,
        pattern=r"^[a-f0-9\-]{36}$",
        description="Session ID (UUID format). A new session is created if omitted."
    )
    source_filter: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Restrict retrieval to documents whose filename contains this string (e.g. '38300')"
    )
    top_k: Optional[int] = Field(
        default=None,
        ge=1,
        le=20,
        description="Number of document chunks to retrieve (default: 5)"
    )
    domain: Optional[Literal["RAN", "CORE"]] = Field(
        default=None,
        description="Restrict retrieval to RAN or CORE specs only. Omit to search all domains."
    )
    generation: Optional[Literal["5G", "LTE"]] = Field(
        default=None,
        description="Restrict retrieval to 5G or LTE specs only. Omit to search all generations."
    )


class SourceDocument(BaseModel):
    """A retrieved source document chunk"""
    source: str = Field(description="Source document filename")
    similarity: float = Field(description="Semantic similarity score (0-1)")
    text: str = Field(description="Preview of the retrieved chunk text")
    domain: Optional[str] = Field(default=None, description="'RAN' or 'CORE'")
    generation: Optional[str] = Field(default=None, description="'5G' or 'LTE'")
    spec_number: Optional[str] = Field(default=None, description="e.g. '38.300'")
    spec_title: Optional[str] = Field(default=None, description="Human-readable spec title")


class QueryResponse(BaseModel):
    """Response body for POST /query"""
    answer: str = Field(description="Generated answer from the LLM")
    sources: List[SourceDocument] = Field(description="Retrieved source documents")
    session_id: str = Field(description="Session ID — pass this back to continue the conversation")
    query_time: float = Field(description="Total query time in seconds")
    retrieve_time: float = Field(description="Retrieval time in seconds")
    generate_time: float = Field(description="LLM generation time in seconds")


# ---------------------------------------------------------------------------
# History endpoint
# ---------------------------------------------------------------------------

class HistoryMessage(BaseModel):
    """A single message in conversation history"""
    role: str = Field(description="'user' or 'assistant'")
    content: str = Field(description="Message content")


class HistoryResponse(BaseModel):
    """Response body for GET /history/{session_id}"""
    session_id: str
    messages: List[HistoryMessage]
    message_count: int


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------

class ComponentStatus(BaseModel):
    """Status of a single system component"""
    status: str = Field(description="'ok', 'degraded', or 'unavailable'")
    detail: Optional[str] = None


class HealthResponse(BaseModel):
    """Response body for GET /health"""
    status: str = Field(description="Overall status: 'ok' or 'degraded'")
    components: dict = Field(description="Per-component status")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Stats endpoint
# ---------------------------------------------------------------------------

class StatsResponse(BaseModel):
    """Response body for GET /stats"""
    vector_store: dict = Field(description="Vector store statistics")
    active_sessions: int = Field(description="Number of active conversation sessions")
    metrics: dict = Field(description="Aggregated query performance metrics")


# ---------------------------------------------------------------------------
# Eval endpoint
# ---------------------------------------------------------------------------

class EvalResponse(BaseModel):
    """Response body for GET /eval"""
    evaluated_at: Optional[str] = Field(
        default=None,
        description="ISO-8601 timestamp of when the evaluation was last run"
    )
    config: Optional[dict] = Field(
        default=None,
        description="Evaluation configuration (top_k, full_eval, total_chunks)"
    )
    summary: Optional[dict] = Field(
        default=None,
        description="Aggregated retrieval, answer, and latency metrics"
    )
    cases: Optional[List[dict]] = Field(
        default=None,
        description="Per-query results"
    )
    available: bool = Field(
        description="False if no eval results exist yet (run eval script first)"
    )


# ---------------------------------------------------------------------------
# Error response
# ---------------------------------------------------------------------------

class ErrorResponse(BaseModel):
    """Standard error response"""
    error: str
    detail: Optional[str] = None


# ---------------------------------------------------------------------------
# Catalog endpoint
# ---------------------------------------------------------------------------

class CatalogSpec(BaseModel):
    """A single entry from the spec catalog"""
    spec_number: str
    title: str
    domain: str
    generation: str
    description: str
    indexed: bool = Field(description="True if this spec has been indexed in the vector store")


class CatalogResponse(BaseModel):
    """Response body for GET /catalog"""
    total: int
    specs: List[CatalogSpec]

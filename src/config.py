"""
Centralised configuration for the 3GPP RAG Assistant.

All settings are read from environment variables (or a ``.env`` file in the
project root) via pydantic-settings. Unknown keys are silently ignored so
legacy ``.env`` files with e.g. ``OPENAI_API_KEY`` don't cause errors.

Environment variable names match the field names exactly (case-insensitive).
Override any default by exporting the variable before starting the server:

    export LLM_PROVIDER=openai
    uvicorn src.api.main:app --reload

See ``.env.example`` in the project root for a full reference.
"""

from urllib.parse import urlparse

from pydantic_settings import BaseSettings
from pydantic import ConfigDict, field_validator


class Settings(BaseSettings):
    """Application settings loaded from environment variables / .env file.

    Attributes:
        llm_provider: Active LLM provider. Production default is OpenAI.
        ollama_base_url: URL of the Ollama server when LLM_PROVIDER=ollama.
        llm_model: Name of the Ollama model to use when LLM_PROVIDER=ollama.
            Must be pulled first:
            ``ollama pull <model>``.
        max_tokens: Maximum number of tokens in the LLM response.
        temperature: Sampling temperature. Lower = more deterministic (0.0–1.0).
        embedding_provider: Active embedding provider. Production default is OpenAI.
        embedding_model: Shortcut key for the legacy sentence-transformer model.
        vector_store_provider: Active vector-store provider. Production default is Pinecone.
        vector_db_path: Directory where legacy ChromaDB persists its data.
        collection_name: Legacy ChromaDB collection that holds indexed chunks.
        chunk_size: Target character length of each document chunk.
        chunk_overlap: Character overlap between adjacent chunks for context
            continuity at boundaries.
        data_dir: Directory scanned by the document processor for raw specs.
        api_host: Interface the FastAPI server binds to.
        api_port: Port the FastAPI server listens on.
        log_level: Root logging level (DEBUG / INFO / WARNING / ERROR).
        max_history_length: Number of prior Q&A turns kept in each session.
        top_k_results: Default number of chunks retrieved per query.
        query_expansion: Append full forms of known 3GPP abbreviations to the
            query before embedding (src/core/query_expansion.py).
        query_decomposition: Split comparison questions into per-side
            sub-queries fused with the raw ranking
            (src/core/query_decomposition.py).
    """

    model_config = ConfigDict(env_file=".env", case_sensitive=False, extra="ignore")

    # LLM Provider: "openai" (production), "ollama" (local), or "groq" (cloud)
    llm_provider: str = "openai"

    # Ollama Configuration (local LLM - no API key needed)
    ollama_base_url: str = "http://localhost:11434"

    @field_validator("ollama_base_url")
    @classmethod
    def validate_ollama_url(cls, v: str) -> str:
        """Only allow localhost/127.0.0.1 Ollama URLs to prevent SSRF."""
        parsed = urlparse(v)
        allowed_hosts = {"localhost", "127.0.0.1", "host.docker.internal", "ollama"}
        if parsed.hostname not in allowed_hosts:
            raise ValueError(
                f"ollama_base_url must target localhost or a trusted host, "
                f"got '{parsed.hostname}'"
            )
        return v

    llm_model: str = "llama3.2"
    max_tokens: int = 1000
    temperature: float = 0.1

    # Groq Configuration (cloud LLM - free API key from https://console.groq.com/keys)
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"

    # OpenAI Configuration (cloud LLM and embeddings)
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"

    # Embedding Configuration: "openai" (production) or "local"
    embedding_provider: str = "openai"
    embedding_model: str = "bge-small"  # options: mini, mpnet, bge-small, bge-base

    # Vector Database Configuration: "pinecone" (production) or "chroma"
    vector_store_provider: str = "pinecone"
    vector_db_path: str = "./data/vectordb"
    collection_name: str = "3gpp_specs"

    # Pinecone Vector Database Configuration (cloud vector store)
    pinecone_api_key: str = ""
    pinecone_index_name: str = "3gpp-rag"
    pinecone_namespace: str = "3gpp-specs"

    # Document Processing
    chunk_size: int = 1000
    chunk_overlap: int = 200
    data_dir: str = "./data/raw"

    # API Configuration
    api_host: str = "127.0.0.1"  # Bind to localhost by default; use 0.0.0.0 in containers
    api_port: int = 8000
    log_level: str = "INFO"

    # Application Settings
    max_history_length: int = 5
    top_k_results: int = 5

    # Evidence gate settings. The reranker defaults below were calibrated for
    # the indexed TS 38.300 + TS 38.401 corpus using text-embedding-3-small and
    # cross-encoder/ms-marco-MiniLM-L6-v2. They are not universal thresholds.
    evidence_gate_enabled: bool = True
    evidence_min_top_score: float = 0.80
    evidence_min_doc_score: float = 0.80
    evidence_min_docs: int = 2
    evidence_mean_top_n: int = 3
    evidence_min_mean_score: float = 0.0
    evidence_score_source: str = "reranker"

    # Post-generation answer verification. When enabled, generated answers are
    # withheld unless citations and grounding both pass.
    answer_verification_enabled: bool = True
    openai_verifier_model: str = ""

    # Cross-encoder reranking between vector retrieval and EvidenceGate.
    reranker_enabled: bool = True
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    reranker_candidate_k: int = 10
    reranker_top_k: int = 5

    @field_validator(
        "evidence_min_top_score",
        "evidence_min_doc_score",
        "evidence_min_mean_score",
    )
    @classmethod
    def validate_evidence_score(cls, v: float) -> float:
        """Evidence score thresholds are normalized into a 0.0-1.0 range."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("evidence score thresholds must be between 0.0 and 1.0")
        return v

    @field_validator("evidence_min_docs")
    @classmethod
    def validate_evidence_min_docs(cls, v: int) -> int:
        """Minimum qualifying document count cannot be negative."""
        if v < 0:
            raise ValueError("evidence_min_docs must be greater than or equal to zero")
        return v

    @field_validator("evidence_mean_top_n")
    @classmethod
    def validate_evidence_mean_top_n(cls, v: int) -> int:
        """Mean score window must include at least one document."""
        if v <= 0:
            raise ValueError("evidence_mean_top_n must be greater than zero")
        return v

    @field_validator("evidence_score_source")
    @classmethod
    def validate_evidence_score_source(cls, v: str) -> str:
        """Evidence scores can come from vector similarity or reranker score."""
        normalized = v.strip().lower()
        if normalized not in {"vector", "reranker"}:
            raise ValueError("evidence_score_source must be 'vector' or 'reranker'")
        return normalized

    @field_validator("reranker_candidate_k", "reranker_top_k")
    @classmethod
    def validate_reranker_k(cls, v: int) -> int:
        """Reranker retrieval limits must include at least one document."""
        if v <= 0:
            raise ValueError("reranker k values must be greater than zero")
        return v

    # Query-time 3GPP vocabulary expansion (src/core/query_expansion.py):
    # appends full forms of known abbreviations before embedding the query.
    query_expansion: bool = True
    # Comparison-query decomposition (src/core/query_decomposition.py):
    # "difference between X and Y" issues one sub-query per side, merged
    # with the raw ranking via rank fusion.
    query_decomposition: bool = True


# Global settings instance
settings = Settings()

"""
Centralised configuration for the 3GPP RAG Assistant.

All settings are read from environment variables (or a ``.env`` file in the
project root) via pydantic-settings. Unknown keys are silently ignored so
legacy ``.env`` files with e.g. ``OPENAI_API_KEY`` don't cause errors.

Environment variable names match the field names exactly (case-insensitive).
Override any default by exporting the variable before starting the server:

    export LLM_MODEL=mistral
    uvicorn src.api.main:app --reload

See ``.env.example`` in the project root for a full reference.
"""
from urllib.parse import urlparse

from pydantic_settings import BaseSettings
from pydantic import ConfigDict, field_validator


class Settings(BaseSettings):
    """Application settings loaded from environment variables / .env file.

    Attributes:
        ollama_base_url: URL of the Ollama server (default: localhost).
        llm_model: Name of the Ollama model to use. Must be pulled first:
            ``ollama pull <model>``.
        max_tokens: Maximum number of tokens in the LLM response.
        temperature: Sampling temperature. Lower = more deterministic (0.0–1.0).
        embedding_model: Shortcut key for the sentence-transformer model.
            Must match the model used when the vector index was built.
        vector_db_path: Directory where ChromaDB persists its data.
        collection_name: ChromaDB collection that holds the indexed chunks.
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

    # LLM Provider: "ollama" (local, zero cost) or "groq" (cloud, free tier)
    llm_provider: str = "ollama"

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

    # Embedding Configuration (local sentence-transformers - no API key needed)
    embedding_model: str = "bge-small"  # options: mini, mpnet, bge-small, bge-base

    # Vector Database Configuration
    vector_db_path: str = "./data/vectordb"
    collection_name: str = "3gpp_specs"

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
    # Query-time 3GPP vocabulary expansion (src/core/query_expansion.py):
    # appends full forms of known abbreviations before embedding the query.
    query_expansion: bool = True
    # Comparison-query decomposition (src/core/query_decomposition.py):
    # "difference between X and Y" issues one sub-query per side, merged
    # with the raw ranking via rank fusion.
    query_decomposition: bool = True


# Global settings instance
settings = Settings()

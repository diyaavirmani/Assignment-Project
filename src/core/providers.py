"""Provider factories for embeddings, vector stores, and LLMs."""

from typing import Optional

from src.config import settings
from src.core.embeddings import LocalEmbeddingGenerator
from src.core.groq_llm import GroqLLM
from src.core.llm import OllamaLLM
from src.core.openai_embeddings import OpenAIEmbeddingGenerator
from src.core.openai_llm import OpenAILLM
from src.core.pinecone_store import PineconeVectorStore
from src.core.vector_store import VectorStore


def create_embedding_generator(provider: Optional[str] = None):
    """Create the configured embedding generator."""
    selected = _normalize_provider(provider or settings.embedding_provider)
    if selected == "local":
        return LocalEmbeddingGenerator(model_name=settings.embedding_model)
    if selected == "openai":
        return OpenAIEmbeddingGenerator(
            model_name=settings.openai_embedding_model,
            api_key=settings.openai_api_key or None,
        )
    raise ValueError(
        "Invalid EMBEDDING_PROVIDER. Expected one of: local, openai"
    )


def create_vector_store(provider: Optional[str] = None):
    """Create the configured vector store."""
    selected = _normalize_provider(provider or settings.vector_store_provider)
    if selected == "chroma":
        return VectorStore(
            persist_directory=settings.vector_db_path,
            collection_name=settings.collection_name,
        )
    if selected == "pinecone":
        return PineconeVectorStore(
            index_name=settings.pinecone_index_name,
            namespace=settings.pinecone_namespace,
            api_key=settings.pinecone_api_key or None,
        )
    raise ValueError(
        "Invalid VECTOR_STORE_PROVIDER. Expected one of: chroma, pinecone"
    )


def create_llm(provider: Optional[str] = None):
    """Create the configured LLM adapter."""
    selected = _normalize_provider(provider or settings.llm_provider)
    if selected == "ollama":
        return OllamaLLM(
            model=settings.llm_model,
            base_url=settings.ollama_base_url,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
        )
    if selected == "groq":
        return GroqLLM(
            model=settings.groq_model,
            api_key=settings.groq_api_key or None,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
        )
    if selected == "openai":
        return OpenAILLM(
            model=settings.openai_model,
            api_key=settings.openai_api_key or None,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
        )
    raise ValueError("Invalid LLM_PROVIDER. Expected one of: ollama, groq, openai")


def _normalize_provider(provider: str) -> str:
    return provider.strip().lower()

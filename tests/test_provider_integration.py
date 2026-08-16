"""Provider factory and construction tests."""

from unittest.mock import MagicMock

import pytest


class TestProviderFactories:
    def test_default_providers_use_production_path(self, monkeypatch):
        from src.core import providers

        fake_embeddings = MagicMock()
        fake_store = MagicMock()
        fake_llm = MagicMock()
        monkeypatch.setattr(providers, "OpenAIEmbeddingGenerator", fake_embeddings)
        monkeypatch.setattr(providers, "PineconeVectorStore", fake_store)
        monkeypatch.setattr(providers, "OpenAILLM", fake_llm)
        monkeypatch.setattr(providers.settings, "embedding_provider", "openai")
        monkeypatch.setattr(providers.settings, "vector_store_provider", "pinecone")
        monkeypatch.setattr(providers.settings, "llm_provider", "openai")
        monkeypatch.setattr(providers.settings, "openai_embedding_model", "embed-model")
        monkeypatch.setattr(providers.settings, "openai_model", "gpt-test")
        monkeypatch.setattr(providers.settings, "openai_api_key", "test-openai")
        monkeypatch.setattr(providers.settings, "pinecone_index_name", "3gpp-rag")
        monkeypatch.setattr(providers.settings, "pinecone_namespace", "3gpp-specs")
        monkeypatch.setattr(providers.settings, "pinecone_api_key", "test-pinecone")

        providers.create_embedding_generator()
        providers.create_vector_store()
        providers.create_llm()

        fake_embeddings.assert_called_once()
        fake_store.assert_called_once()
        fake_llm.assert_called_once()

    def test_embedding_provider_openai(self, monkeypatch):
        from src.core import providers

        fake = MagicMock()
        monkeypatch.setattr(providers, "OpenAIEmbeddingGenerator", fake)
        monkeypatch.setattr(providers.settings, "openai_embedding_model", "embed-model")
        monkeypatch.setattr(providers.settings, "openai_api_key", "test-key")

        providers.create_embedding_generator("openai")

        fake.assert_called_once_with(model_name="embed-model", api_key="test-key")

    def test_embedding_provider_local(self, monkeypatch):
        from src.core import providers

        fake = MagicMock()
        monkeypatch.setattr(providers, "LocalEmbeddingGenerator", fake)
        monkeypatch.setattr(providers.settings, "embedding_model", "bge-small")

        providers.create_embedding_generator("local")

        fake.assert_called_once_with(model_name="bge-small")

    def test_vector_store_provider_pinecone(self, monkeypatch):
        from src.core import providers

        fake = MagicMock()
        monkeypatch.setattr(providers, "PineconeVectorStore", fake)
        monkeypatch.setattr(providers.settings, "pinecone_index_name", "3gpp-rag")
        monkeypatch.setattr(providers.settings, "pinecone_namespace", "3gpp-specs")
        monkeypatch.setattr(providers.settings, "pinecone_api_key", "test-key")

        providers.create_vector_store("pinecone")

        fake.assert_called_once_with(
            index_name="3gpp-rag",
            namespace="3gpp-specs",
            api_key="test-key",
        )

    def test_vector_store_provider_chroma(self, monkeypatch):
        from src.core import providers

        fake = MagicMock()
        monkeypatch.setattr(providers, "VectorStore", fake)
        monkeypatch.setattr(providers.settings, "vector_db_path", "data/vectordb")
        monkeypatch.setattr(providers.settings, "collection_name", "3gpp_specs")

        providers.create_vector_store("chroma")

        fake.assert_called_once_with(
            persist_directory="data/vectordb",
            collection_name="3gpp_specs",
        )

    def test_llm_provider_openai(self, monkeypatch):
        from src.core import providers

        fake = MagicMock()
        monkeypatch.setattr(providers, "OpenAILLM", fake)
        monkeypatch.setattr(providers.settings, "openai_model", "gpt-test")
        monkeypatch.setattr(providers.settings, "openai_api_key", "test-key")
        monkeypatch.setattr(providers.settings, "temperature", 0.2)
        monkeypatch.setattr(providers.settings, "max_tokens", 123)

        providers.create_llm("openai")

        fake.assert_called_once_with(
            model="gpt-test",
            api_key="test-key",
            temperature=0.2,
            max_tokens=123,
        )

    def test_llm_provider_legacy_ollama(self, monkeypatch):
        from src.core import providers

        fake = MagicMock()
        monkeypatch.setattr(providers, "OllamaLLM", fake)
        monkeypatch.setattr(providers.settings, "llm_model", "llama3.2")
        monkeypatch.setattr(providers.settings, "ollama_base_url", "http://localhost:11434")
        monkeypatch.setattr(providers.settings, "temperature", 0.1)
        monkeypatch.setattr(providers.settings, "max_tokens", 1000)

        providers.create_llm("ollama")

        fake.assert_called_once_with(
            model="llama3.2",
            base_url="http://localhost:11434",
            temperature=0.1,
            max_tokens=1000,
        )

    @pytest.mark.parametrize(
        ("factory", "provider", "message"),
        [
            ("create_embedding_generator", "bad", "EMBEDDING_PROVIDER"),
            ("create_vector_store", "bad", "VECTOR_STORE_PROVIDER"),
            ("create_llm", "bad", "LLM_PROVIDER"),
        ],
    )
    def test_invalid_provider_configuration(self, factory, provider, message):
        from src.core import providers

        with pytest.raises(ValueError, match=message):
            getattr(providers, factory)(provider)


class TestRetrieverProviderIntegration:
    def test_retriever_uses_configured_factories(self, monkeypatch):
        from src.core import retriever as retriever_module

        mock_embedding = MagicMock()
        mock_embedding.generate_embedding.return_value = [0.1, 0.2, 0.3]
        mock_store = MagicMock()
        mock_store.query.return_value = {
            "documents": [["matched text"]],
            "metadatas": [
                [
                    {
                        "source": "38300-g30.docx",
                        "chunk_index": 4,
                        "domain": "RAN",
                        "generation": "5G",
                        "spec_number": "38.300",
                        "spec_title": "NR overall description",
                    }
                ]
            ],
            "distances": [[0.07]],
        }

        create_embedding = MagicMock(return_value=mock_embedding)
        create_store = MagicMock(return_value=mock_store)
        monkeypatch.setattr(retriever_module, "create_embedding_generator", create_embedding)
        monkeypatch.setattr(retriever_module, "create_vector_store", create_store)
        monkeypatch.setattr(retriever_module.settings, "query_expansion", False)
        monkeypatch.setattr(retriever_module.settings, "query_decomposition", False)

        retriever = retriever_module.DocumentRetriever(top_k=1)
        docs = retriever.retrieve("What is gNB-CU?")

        create_embedding.assert_called_once_with()
        create_store.assert_called_once_with()
        assert docs == [
            {
                "text": "matched text",
                "source": "38300-g30.docx",
                "chunk_index": 4,
                "similarity": pytest.approx(0.93),
                "domain": "RAN",
                "generation": "5G",
                "spec_number": "38.300",
                "spec_title": "NR overall description",
            }
        ]


class TestRAGChainProviderIntegration:
    def test_rag_chain_uses_configured_llm_factory(self, monkeypatch):
        from src.core import rag_chain as rag_module

        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [
            {
                "text": "context",
                "source": "38300-g30.docx",
                "chunk_index": 0,
                "similarity": 0.9,
                "vector_similarity": 0.9,
                "reranker_score": 0.9,
            },
            {
                "text": "more context",
                "source": "38300-g30.docx",
                "chunk_index": 1,
                "similarity": 0.88,
                "vector_similarity": 0.88,
                "reranker_score": 0.88,
            },
        ]
        mock_retriever.format_context.return_value = "context"
        mock_llm = MagicMock()
        mock_llm.model = "gpt-test"
        mock_llm.generate.return_value = "answer"
        create_llm = MagicMock(return_value=mock_llm)
        monkeypatch.setattr(rag_module, "create_llm", create_llm)
        mock_verifier = MagicMock()
        mock_verifier.enabled = True
        mock_verifier.verify.return_value.to_dict.return_value = {
            "passed": True,
            "reason": "verified",
            "total_claims": 1,
            "supported_claims": 1,
            "unsupported_claims": [],
            "contradicted_claims": [],
            "citation_valid": True,
            "cited_sources": ["S1"],
            "invalid_citations": [],
            "detail": "",
        }
        mock_verifier.verify.return_value.passed = True
        monkeypatch.setattr(rag_module, "AnswerVerifier", MagicMock(return_value=mock_verifier))

        chain = rag_module.RAGChain(retriever=mock_retriever)
        result = chain.query("question")

        create_llm.assert_called_once_with()
        assert result["answer"] == "answer"

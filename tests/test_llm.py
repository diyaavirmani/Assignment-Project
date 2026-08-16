"""
Tests for src/core/llm.py

These tests mock the Ollama client so they run without a live Ollama server.
"""
import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_chat_response(content: str):
    """Build a minimal mock that mimics ollama.Client.chat() return value."""
    msg = MagicMock()
    msg.content = content
    resp = MagicMock()
    resp.message = msg
    return resp


def _make_stream_chunks(tokens: list):
    """Build a list of streaming chunk mocks."""
    chunks = []
    for token in tokens:
        msg = MagicMock()
        msg.content = token
        chunk = MagicMock()
        chunk.message = msg
        chunks.append(chunk)
    return chunks


def _make_model_list(names: list):
    """Build a mock ollama.list() response."""
    models = []
    for name in names:
        m = MagicMock()
        m.model = name
        models.append(m)
    resp = MagicMock()
    resp.models = models
    return resp


# ---------------------------------------------------------------------------
# OllamaLLM unit tests
# ---------------------------------------------------------------------------

@patch("src.core.llm.OllamaLLM.__init__", lambda self, **kw: None)
def _build_llm():
    """Create an OllamaLLM with a mocked internal client."""
    from src.core.llm import OllamaLLM
    llm = OllamaLLM.__new__(OllamaLLM)
    llm.model = "llama3.2"
    llm.base_url = "http://localhost:11434"
    llm.temperature = 0.1
    llm.max_tokens = 100
    llm._ollama = MagicMock()
    llm._client = MagicMock()
    return llm


class TestOllamaLLMAvailability:
    def test_is_available_true_when_model_present(self):
        llm = _build_llm()
        llm._client.list.return_value = _make_model_list(["llama3.2:latest"])
        assert llm.is_available() is True

    def test_is_available_false_when_model_missing(self):
        llm = _build_llm()
        llm._client.list.return_value = _make_model_list(["mistral:latest"])
        assert llm.is_available() is False

    def test_is_available_false_when_server_down(self):
        llm = _build_llm()
        llm._client.list.side_effect = ConnectionError("refused")
        assert llm.is_available() is False


class TestOllamaLLMGenerate:
    def test_generate_returns_string(self):
        llm = _build_llm()
        llm._client.chat.return_value = _make_chat_response("gNB is a base station.")
        result = llm.generate("What is a gNB?")
        assert isinstance(result, str)
        assert "gNB" in result

    def test_generate_passes_system_prompt(self):
        llm = _build_llm()
        llm._client.chat.return_value = _make_chat_response("ok")
        llm.generate("hi", system_prompt="Custom system")
        call_args = llm._client.chat.call_args
        messages = call_args.kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "Custom system"

    def test_generate_includes_history(self):
        llm = _build_llm()
        llm._client.chat.return_value = _make_chat_response("follow-up answer")
        history = [
            {"role": "user", "content": "first question"},
            {"role": "assistant", "content": "first answer"},
        ]
        llm.generate("second question", history=history)
        call_args = llm._client.chat.call_args
        messages = call_args.kwargs["messages"]
        roles = [m["role"] for m in messages]
        assert roles == ["system", "user", "assistant", "user"]

    def test_generate_raises_on_server_error(self):
        from src.core.llm import OllamaLLM
        llm = _build_llm()
        llm._client.chat.side_effect = Exception("connection refused")
        with pytest.raises(RuntimeError, match="Failed to generate"):
            llm.generate("hello")


class TestOllamaLLMStream:
    def test_stream_yields_tokens(self):
        llm = _build_llm()
        llm._client.chat.return_value = _make_stream_chunks(["Hello", " world", "!"])
        tokens = list(llm.stream("say hello"))
        assert tokens == ["Hello", " world", "!"]

    def test_stream_raises_on_error(self):
        llm = _build_llm()
        llm._client.chat.side_effect = Exception("timeout")
        with pytest.raises(RuntimeError, match="Streaming failed"):
            list(llm.stream("hello"))

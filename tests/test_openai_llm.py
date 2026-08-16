"""Tests for src/core/openai_llm.py."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.core.openai_llm import OpenAILLM


def _chat_response(content: str):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


def _stream_chunk(content):
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=SimpleNamespace(content=content))]
    )


@pytest.fixture
def openai_chat_client():
    client = MagicMock()
    client.chat.completions.create.return_value = _chat_response(
        "The gNB-CU handles RRC and PDCP."
    )
    return client


class TestOpenAILLM:
    def test_generate_returns_text(self, openai_chat_client):
        llm = OpenAILLM(
            model="gpt-test",
            temperature=0.2,
            max_tokens=50,
            client=openai_chat_client,
        )

        result = llm.generate("Explain gNB-CU")

        assert result == "The gNB-CU handles RRC and PDCP."
        openai_chat_client.chat.completions.create.assert_called_once()
        kwargs = openai_chat_client.chat.completions.create.call_args.kwargs
        assert kwargs["model"] == "gpt-test"
        assert kwargs["temperature"] == 0.2
        assert kwargs["max_tokens"] == 50
        assert kwargs["messages"][-1] == {
            "role": "user",
            "content": "Explain gNB-CU",
        }

    def test_generate_includes_history(self, openai_chat_client):
        llm = OpenAILLM(model="gpt-test", client=openai_chat_client)
        history = [
            {"role": "user", "content": "First"},
            {"role": "assistant", "content": "Answer"},
        ]

        llm.generate("Follow up", history=history)

        roles = [
            message["role"]
            for message in openai_chat_client.chat.completions.create.call_args.kwargs[
                "messages"
            ]
        ]
        assert roles == ["system", "user", "assistant", "user"]

    def test_stream_yields_tokens(self):
        client = MagicMock()
        client.chat.completions.create.return_value = [
            _stream_chunk("Hello"),
            _stream_chunk(None),
            _stream_chunk(" world"),
        ]
        llm = OpenAILLM(model="gpt-test", client=client)

        tokens = list(llm.stream("Say hello"))

        assert tokens == ["Hello", " world"]
        assert client.chat.completions.create.call_args.kwargs["stream"] is True

    def test_is_available_true_on_success(self, openai_chat_client):
        llm = OpenAILLM(model="gpt-test", client=openai_chat_client)

        assert llm.is_available() is True
        assert openai_chat_client.chat.completions.create.call_args.kwargs[
            "max_tokens"
        ] == 1

    def test_is_available_false_on_error(self, openai_chat_client):
        openai_chat_client.chat.completions.create.side_effect = Exception("boom")
        llm = OpenAILLM(model="gpt-test", client=openai_chat_client)

        assert llm.is_available() is False

    def test_missing_api_key_raises_clear_error(self):
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            OpenAILLM(model="gpt-test", api_key="")

    def test_missing_model_raises_clear_error(self, openai_chat_client):
        with pytest.raises(ValueError, match="OPENAI_MODEL"):
            OpenAILLM(model="", client=openai_chat_client)

    def test_invalid_max_tokens_raises_clear_error(self, openai_chat_client):
        with pytest.raises(ValueError, match="max_tokens"):
            OpenAILLM(model="gpt-test", max_tokens=0, client=openai_chat_client)

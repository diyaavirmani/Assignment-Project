"""
OpenAI LLM client for the 3GPP RAG pipeline.

This adapter mirrors the OllamaLLM/GroqLLM interface used by RAGChain:
``generate(...)``, ``stream(...)``, and ``is_available()``. Unit tests can
inject a fake client, so no real OpenAI calls are needed.
"""

import logging
from typing import Any, Dict, Iterator, List, Optional

from src.config import settings
from src.core.llm import SYSTEM_PROMPT

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - exercised only when dependency is absent
    OpenAI = None

logger = logging.getLogger(__name__)


class OpenAILLM:
    """Cloud LLM client via the official OpenAI Python SDK."""

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        client: Optional[Any] = None,
    ) -> None:
        """
        Args:
            model: OpenAI chat model. Defaults to Settings.openai_model.
            api_key: OpenAI API key. Defaults to Settings.openai_api_key.
            temperature: Sampling temperature. Defaults to Settings.temperature.
            max_tokens: Maximum response tokens. Defaults to Settings.max_tokens.
            client: Optional prebuilt OpenAI-compatible client for tests.

        Raises:
            ImportError: If the OpenAI SDK is missing and no client is injected.
            ValueError: If required OpenAI configuration is missing.
        """
        self.model = model if model is not None else settings.openai_model
        self.temperature = (
            temperature if temperature is not None else settings.temperature
        )
        self.max_tokens = max_tokens if max_tokens is not None else settings.max_tokens

        if not self.model:
            raise ValueError("OPENAI_MODEL is required to use OpenAILLM")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be greater than zero")

        if client is not None:
            self._client = client
            return

        resolved_api_key = api_key if api_key is not None else settings.openai_api_key
        if not resolved_api_key:
            raise ValueError("OPENAI_API_KEY is required to use OpenAILLM")
        if OpenAI is None:
            raise ImportError("openai is required. Install with: pip install openai")

        self._client = OpenAI(api_key=resolved_api_key)
        logger.info("Initialized OpenAILLM: model=%s", self.model)

    def is_available(self) -> bool:
        """Check whether the configured OpenAI model can answer a tiny request."""
        try:
            self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=1,
            )
            return True
        except Exception as e:
            logger.warning("OpenAI API not reachable: %s", e)
            return False

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """Generate a response for one prompt."""
        messages = self._build_messages(prompt, system_prompt, history)

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            answer = response.choices[0].message.content
            return answer or ""
        except Exception as e:
            logger.error("OpenAI generation failed: %s", e)
            raise RuntimeError("LLM generation failed. Please try again.")

    def stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Iterator[str]:
        """Stream a response token by token."""
        messages = self._build_messages(prompt, system_prompt, history)

        try:
            stream = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True,
            )
            for chunk in stream:
                content = chunk.choices[0].delta.content
                if content:
                    yield content
        except Exception as e:
            logger.error("OpenAI streaming failed: %s", e)
            raise RuntimeError("LLM streaming failed. Please try again.")

    @staticmethod
    def _build_messages(
        prompt: str,
        system_prompt: Optional[str],
        history: Optional[List[Dict[str, str]]],
    ) -> List[Dict[str, str]]:
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt or SYSTEM_PROMPT}
        ]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": prompt})
        return messages

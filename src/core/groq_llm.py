"""
Cloud LLM client using Groq API (free tier, runs Llama/Mixtral at inference speed)

Groq offers free API access to open-source models:
  - llama-3.3-70b-versatile : Meta's Llama 3.3 70B (best quality, free)
  - llama-3.1-8b-instant    : Meta's Llama 3.1 8B (fastest, free)
  - mixtral-8x7b-32768      : Mistral's Mixtral 8x7B (good for technical text)

Get a free API key: https://console.groq.com/keys
"""
import logging
from typing import List, Dict, Optional, Iterator

logger = logging.getLogger(__name__)

# System prompt tailored for 3GPP technical queries
SYSTEM_PROMPT = """You are a 3GPP technical specification assistant. You help engineers and \
researchers understand 3GPP standards by answering questions based on provided specification \
excerpts.

Guidelines:
- Answer based ONLY on the provided context excerpts
- Always cite which document/source you are drawing from
- Use precise technical language appropriate for telecom engineers
- If the context does not contain enough information, say so clearly
- Do not speculate beyond what is stated in the specifications
- Format answers clearly with relevant section references when available"""


class GroqLLM:
    """Cloud LLM client via Groq API — free tier, fast inference"""

    def __init__(
        self,
        model: str = "llama-3.3-70b-versatile",
        api_key: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 1000,
    ):
        """
        Initialize Groq LLM client.

        Args:
            model: Groq model name (see https://console.groq.com/docs/models)
            api_key: Groq API key (or set GROQ_API_KEY env var)
            temperature: Sampling temperature (lower = more deterministic)
            max_tokens: Maximum tokens in the response
        """
        try:
            from groq import Groq
            self._groq_cls = Groq
        except ImportError:
            raise ImportError(
                "groq package not installed. Run: pip install groq"
            )

        import os
        self.model = model
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.temperature = temperature
        self.max_tokens = max_tokens

        if not self.api_key:
            raise ValueError(
                "Groq API key required. Set GROQ_API_KEY env var or pass api_key. "
                "Get a free key at: https://console.groq.com/keys"
            )

        self._client = self._groq_cls(api_key=self.api_key)
        logger.info(f"Initialized GroqLLM: model={model}")

    def is_available(self) -> bool:
        """Check whether the Groq API is reachable and the model works."""
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=5,
            )
            return True
        except Exception as e:
            logger.warning(f"Groq API not reachable: {e}")
            return False

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """
        Generate a response (blocking).

        Args:
            prompt: User question / current turn
            system_prompt: Override the default system prompt
            history: Prior conversation turns as [{"role": ..., "content": ...}]

        Returns:
            Generated answer string
        """
        messages = self._build_messages(prompt, system_prompt, history)

        logger.debug(f"Sending {len(messages)} messages to Groq ({self.model})")

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            answer = response.choices[0].message.content
            logger.debug(f"Received response ({len(answer)} chars)")
            return answer

        except Exception as e:
            logger.error(f"Groq generation failed: {e}")
            raise RuntimeError("LLM generation failed. Please try again.")

    def stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Iterator[str]:
        """
        Stream a response token by token.

        Yields:
            Successive text chunks as they are generated
        """
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
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            logger.error(f"Groq streaming failed: {e}")
            raise RuntimeError("LLM streaming failed. Please try again.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_messages(
        self,
        prompt: str,
        system_prompt: Optional[str],
        history: Optional[List[Dict[str, str]]],
    ) -> List[Dict[str, str]]:
        messages: List[Dict[str, str]] = []

        messages.append({
            "role": "system",
            "content": system_prompt or SYSTEM_PROMPT,
        })

        if history:
            messages.extend(history)

        messages.append({"role": "user", "content": prompt})
        return messages


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    print("\n" + "=" * 60)
    print("Groq LLM Client Test")
    print("=" * 60 + "\n")

    try:
        llm = GroqLLM()
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    if not llm.is_available():
        print("Groq API is not available.")
        sys.exit(1)

    print(f"Model '{llm.model}' is ready.\n")
    print("Streaming response:")
    print("-" * 40)

    for token in llm.stream("What is a gNB in 5G NR? Answer in one sentence."):
        print(token, end="", flush=True)

    print("\n" + "=" * 60 + "\n")

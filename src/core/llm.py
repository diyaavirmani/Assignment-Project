"""
Local LLM client using Ollama (zero cost, runs entirely on your machine)

Ollama runs open-source models locally:
  - llama3.2   : Meta's Llama 3.2 (default, 2GB, great general purpose)
  - mistral    : Mistral 7B (good for technical text)
  - phi3       : Microsoft Phi-3 Mini (very fast, 2.2GB)

Install Ollama: https://ollama.com
Pull a model:  ollama pull llama3.2
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


class OllamaLLM:
    """Local LLM client via Ollama - zero cost, no API key required"""

    def __init__(
        self,
        model: str = "llama3.2",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.1,
        max_tokens: int = 1000,
    ):
        """
        Initialize Ollama LLM client.

        Args:
            model: Ollama model name (must be pulled first: ollama pull <model>)
            base_url: Ollama server URL (default: local)
            temperature: Sampling temperature (lower = more deterministic)
            max_tokens: Maximum tokens in the response
        """
        try:
            import ollama
            self._ollama = ollama
        except ImportError:
            raise ImportError(
                "ollama package not installed. Run: pip install ollama"
            )

        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Point the ollama client at the right host
        self._client = self._ollama.Client(host=base_url)

        logger.info(f"Initialized OllamaLLM: model={model}, url={base_url}")

    def is_available(self) -> bool:
        """Check whether Ollama server is running and the model is available."""
        try:
            models = self._client.list()
            available = [m.model for m in models.models]
            if not any(self.model in m for m in available):
                logger.warning(
                    f"Model '{self.model}' not found. "
                    f"Run: ollama pull {self.model}"
                )
                return False
            return True
        except Exception as e:
            logger.warning(f"Ollama not reachable at {self.base_url}: {e}")
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

        logger.debug(f"Sending {len(messages)} messages to {self.model}")

        try:
            response = self._client.chat(
                model=self.model,
                messages=messages,
                options={
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens,
                },
            )
            answer = response.message.content
            logger.debug(f"Received response ({len(answer)} chars)")
            return answer

        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            raise RuntimeError(
                f"Failed to generate response. Is Ollama running?\n"
                f"Start with: ollama serve\n"
                f"Pull model:  ollama pull {self.model}\n"
                f"Error: {e}"
            )

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
            for chunk in self._client.chat(
                model=self.model,
                messages=messages,
                stream=True,
                options={
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens,
                },
            ):
                yield chunk.message.content

        except Exception as e:
            logger.error(f"Ollama streaming failed: {e}")
            raise RuntimeError(
                f"Streaming failed. Is Ollama running?\n"
                f"Start with: ollama serve\n"
                f"Error: {e}"
            )

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
    print("Ollama LLM Client Test")
    print("=" * 60 + "\n")

    llm = OllamaLLM()

    if not llm.is_available():
        print("Ollama is not available.")
        print("Install: https://ollama.com")
        print(f"Then run: ollama pull {llm.model}")
        sys.exit(1)

    print(f"Model '{llm.model}' is ready.\n")
    print("Streaming response:")
    print("-" * 40)

    for token in llm.stream("What is a gNB in 5G NR? Answer in one sentence."):
        print(token, end="", flush=True)

    print("\n" + "=" * 60 + "\n")

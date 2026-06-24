"""
LLM Functionality.

@author: Cyrus Anderson & Anthony Nguyen
Seattle University, ARIN 5360
@see: https://catalog.seattleu.edu/preview_course_nopop.php?catoid=55&coid=190380
@version: 3.1.0+w26
"""

import os
from typing import Any, Dict, List, Optional

import httpx
from dotenv import load_dotenv

load_dotenv()


class LLMClient:
    """
    LLM Client for interacting with language models.

    Supports Ollama and OpenAI-compatible APIs. Default is Ollama client for qwen2.5:3b.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: Optional[str] = None,
        timeout: float = 30.0,
    ):
        """Initialize the LLM client. URL, model, and API key can be set via environment variables.

        Args:
            base_url (str): Base URL of the LLM service.
            model (str): Model name to use for generation.
            api_key (Optional[str]): API key for authentication.
            timeout (float): Request timeout in seconds.
        """
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self._client = httpx.Client(timeout=self.timeout)

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_completion_tokens: int = 500,
    ) -> str:
        """Generate a response from the LLM.

        Args:
            prompt (str): The user prompt to generate a response for.
            system_prompt (Optional[str]): An optional system prompt to guide the LLM.
            temperature (float): Sampling temperature for response variability.
            max_completion_tokens (int): Maximum tokens to generate in the response.
        """

        messages = self._build_messages(prompt, system_prompt)

        # Construct OpenAI Chat Completions endpoint
        url = f"{self.base_url}/v1/chat/completions"
        headers = self._build_headers()

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_completion_tokens": max_completion_tokens,
        }

        try:
            response = self._client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()

            # OpenAI-compatible result
            return data["choices"][0]["message"]["content"]

        except httpx.TimeoutException:
            raise RuntimeError("LLM request timed out.")
        except httpx.HTTPStatusError as e:
            raise RuntimeError(
                f"LLM request failed with status {e.response.status_code}: {e.response.text}"
            )
        except Exception as e:
            raise RuntimeError(f"Unexpected error: {str(e)}")

    def is_available(self) -> bool:
        """Check if the LLM service is available."""
        try:
            url = f"{self.base_url}/v1/models"
            headers = self._build_headers()
            response = self._client.get(url, headers=headers)
            return response.status_code == 200
        except Exception:
            return False

    def _build_messages(self, prompt: str, system_prompt: Optional[str]) -> List[Dict[str, str]]:
        """Build messages for the LLM request.

        Args:
            prompt (str): The user prompt to generate a response for.
            system_prompt (Optional[str]): An optional system prompt to guide the LLM.
        """
        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages

    def _build_headers(self) -> Dict[str, str]:
        """Build headers for the LLM request."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def close(self):
        """Close the HTTP client."""
        self._client.close()


def create_llm_client(
    provider: str = "ollama",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> LLMClient:
    """Create an LLM client based on the provider. Model and API key can be set via environment variables.

    Args:
        provider (str): The LLM provider ("ollama" or "openai").
        model (Optional[str]): The model name to use.
        api_key (Optional[str]): The API key for authentication.
    """
    provider = provider.lower()

    if provider == "ollama":
        resolved_model = model or os.getenv("OLLAMA_MODEL") or "qwen2.5:3b"
        return LLMClient(
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            model=resolved_model,
            api_key="ollama",
        )

    elif provider == "openai":
        resolved_model = model or os.getenv("OPENAI_MODEL") or "gpt-5.2"
        resolved_api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not resolved_api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY in .env file.")
        return LLMClient(
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com"),
            model=resolved_model,
            api_key=resolved_api_key,
        )

    else:
        raise ValueError(f"Unknown provider: {provider}. Use 'ollama' or 'openai'")

"""
Unit tests for LLM functionality.

@author: Cyrus Anderson & Anthony Nguyen
Seattle University, ARIN 5360
@version: 3.1.0+w26
"""

from unittest.mock import MagicMock, Mock, patch

import httpx
import pytest

from retrieval.llm import LLMClient, create_llm_client


class TestLLMClient:
    """Test suite for LLMClient class."""

    @patch("src.retrieval.llm.httpx.Client")
    def test_generate_success(self, mock_client_class):
        """Test successful generation with OpenAI-compatible format."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.json.return_value = {"choices": [{"message": {"content": "Test answer"}}]}
        mock_response.status_code = 200
        mock_client.post.return_value = mock_response
        mock_client_class.return_value = mock_client

        llm = LLMClient(base_url="http://localhost:11434", model="qwen2.5:3b")
        response = llm.generate("Test prompt")

        assert response == "Test answer"
        mock_client.post.assert_called_once()

    @patch("src.retrieval.llm.httpx.Client")
    def test_generate_with_system_prompt(self, mock_client_class):
        """Test generation with system prompt."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.json.return_value = {"choices": [{"message": {"content": "Response"}}]}
        mock_client.post.return_value = mock_response
        mock_client_class.return_value = mock_client

        llm = LLMClient(base_url="http://localhost:11434", model="qwen2.5:3b")
        llm.generate("User prompt", system_prompt="System instruction")

        # Verify system message was included
        call_args = mock_client.post.call_args
        payload = call_args.kwargs["json"]
        assert len(payload["messages"]) == 2
        assert payload["messages"][0]["role"] == "system"
        assert payload["messages"][0]["content"] == "System instruction"
        assert payload["messages"][1]["role"] == "user"

    @patch("src.retrieval.llm.httpx.Client")
    def test_generate_with_parameters(self, mock_client_class):
        """Test generation with custom temperature and max_tokens."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.json.return_value = {"choices": [{"message": {"content": "Response"}}]}
        mock_client.post.return_value = mock_response
        mock_client_class.return_value = mock_client

        llm = LLMClient(base_url="http://localhost:11434", model="qwen2.5:3b")
        llm.generate("Prompt", temperature=0.5, max_completion_tokens=1000)

        call_args = mock_client.post.call_args
        payload = call_args.kwargs["json"]
        assert payload["temperature"] == 0.5
        assert payload["max_completion_tokens"] == 1000

    @patch("src.retrieval.llm.httpx.Client")
    def test_generate_timeout(self, mock_client_class):
        """Test timeout handling."""
        mock_client = Mock()
        mock_client.post.side_effect = httpx.TimeoutException("Timeout")
        mock_client_class.return_value = mock_client

        llm = LLMClient(base_url="http://localhost:11434", model="qwen2.5:3b")

        with pytest.raises(RuntimeError, match="LLM request timed out."):
            llm.generate("Test prompt")

    @patch("src.retrieval.llm.httpx.Client")
    def test_generate_http_error(self, mock_client_class):
        """Test HTTP status error handling."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_client.post.side_effect = httpx.HTTPStatusError(
            "Error", request=Mock(), response=mock_response
        )
        mock_client_class.return_value = mock_client

        llm = LLMClient(base_url="http://localhost:11434", model="qwen2.5:3b")

        with pytest.raises(
            RuntimeError, match="LLM request failed with status 500: Internal Server Error"
        ):
            llm.generate("Test prompt")

    @patch("src.retrieval.llm.httpx.Client")
    def test_generate_malformed_response(self, mock_client_class):
        """Test handling of malformed API response."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.json.return_value = {"invalid": "format"}
        mock_client.post.return_value = mock_response
        mock_client_class.return_value = mock_client

        llm = LLMClient(base_url="http://localhost:11434", model="qwen2.5:3b")

        with pytest.raises(RuntimeError, match="Unexpected error:"):
            llm.generate("Test prompt")

    @patch("src.retrieval.llm.httpx.Client")
    def test_generate_unexpected_error(self, mock_client_class):
        """Test handling of unexpected errors."""
        mock_client = Mock()
        mock_client.post.side_effect = Exception("Unexpected error")
        mock_client_class.return_value = mock_client

        llm = LLMClient(base_url="http://localhost:11434", model="qwen2.5:3b")

        with pytest.raises(RuntimeError, match="Unexpected error: Unexpected error"):
            llm.generate("Test prompt")

    @patch("src.retrieval.llm.httpx.Client")
    def test_is_available_success(self, mock_client_class):
        """Test availability check when server is reachable."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_client.get.return_value = mock_response
        mock_client_class.return_value = mock_client

        llm = LLMClient(base_url="http://localhost:11434", model="qwen2.5:3b")
        assert llm.is_available() is True
        mock_client.get.assert_called_once_with(
            "http://localhost:11434/v1/models", headers={"Content-Type": "application/json"}
        )

    @patch("src.retrieval.llm.httpx.Client")
    def test_is_available_failure(self, mock_client_class):
        """Test availability check when server is unreachable."""
        mock_client = Mock()
        mock_client.get.side_effect = Exception("Connection error")
        mock_client_class.return_value = mock_client

        llm = LLMClient(base_url="http://localhost:11434", model="qwen2.5:3b")
        assert llm.is_available() is False

    @patch("src.retrieval.llm.httpx.Client")
    def test_is_available_non_200(self, mock_client_class):
        """Test availability check with non-200 response."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.status_code = 404
        mock_client.get.return_value = mock_response
        mock_client_class.return_value = mock_client

        llm = LLMClient(base_url="http://localhost:11434", model="qwen2.5:3b")
        assert llm.is_available() is False

    @patch("src.retrieval.llm.httpx.Client")
    def test_build_headers_with_api_key(self, mock_client_class):
        """Test header construction with API key."""
        mock_client_class.return_value = Mock()

        llm = LLMClient(
            base_url="http://localhost:11434", model="qwen2.5:3b", api_key="test-key-123"
        )
        headers = llm._build_headers()

        assert headers["Content-Type"] == "application/json"
        assert headers["Authorization"] == "Bearer test-key-123"

    @patch("src.retrieval.llm.httpx.Client")
    def test_build_headers_without_api_key(self, mock_client_class):
        """Test header construction without API key."""
        mock_client_class.return_value = Mock()

        llm = LLMClient(base_url="http://localhost:11434", model="qwen2.5:3b")
        headers = llm._build_headers()

        assert headers["Content-Type"] == "application/json"
        assert "Authorization" not in headers

    @patch("src.retrieval.llm.httpx.Client")
    def test_close(self, mock_client_class):
        """Test client closure."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        llm = LLMClient(base_url="http://localhost:11434", model="qwen2.5:3b")
        llm.close()

        mock_client.close.assert_called_once()

    @patch("src.retrieval.llm.httpx.Client")
    def test_base_url_trailing_slash_removed(self, mock_client_class):
        """Test that trailing slash is removed from base_url."""
        mock_client_class.return_value = Mock()

        llm = LLMClient(base_url="http://localhost:11434/", model="qwen2.5:3b")
        assert llm.base_url == "http://localhost:11434"


class TestCreateLLMClient:
    """Test suite for create_llm_client factory function."""

    @patch.dict("os.environ", {}, clear=True)
    @patch("src.retrieval.llm.httpx.Client")
    def test_create_ollama_client_defaults(self, mock_client_class):
        """Test Ollama client creation with defaults."""
        mock_client_class.return_value = Mock()

        client = create_llm_client(provider="ollama")

        assert client.base_url == "http://localhost:11434"
        assert client.model == "qwen2.5:3b"
        assert client.api_key == "ollama"

    @patch.dict(
        "os.environ", {"OLLAMA_MODEL": "llama2:7b", "OLLAMA_BASE_URL": "http://custom:8080"}
    )
    @patch("src.retrieval.llm.httpx.Client")
    def test_create_ollama_client_from_env(self, mock_client_class):
        """Test Ollama client creation from environment variables."""
        mock_client_class.return_value = Mock()

        client = create_llm_client(provider="ollama")

        assert client.base_url == "http://custom:8080"
        assert client.model == "llama2:7b"

    @patch.dict("os.environ", {}, clear=True)
    @patch("src.retrieval.llm.httpx.Client")
    def test_create_ollama_client_with_overrides(self, mock_client_class):
        """Test Ollama client creation with explicit overrides."""
        mock_client_class.return_value = Mock()

        client = create_llm_client(provider="ollama", model="custom-model")

        assert client.model == "custom-model"

    @patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}, clear=True)
    @patch("src.retrieval.llm.httpx.Client")
    def test_create_openai_client_defaults(self, mock_client_class):
        """Test OpenAI client creation with defaults."""
        mock_client_class.return_value = Mock()

        client = create_llm_client(provider="openai")

        assert client.base_url == "https://api.openai.com"
        assert client.model == "gpt-5.2"

    @patch.dict(
        "os.environ",
        {
            "OPENAI_MODEL": "gpt-4",
            "OPENAI_BASE_URL": "https://custom.openai.com",
            "OPENAI_API_KEY": "sk-test",
        },
    )
    @patch("src.retrieval.llm.httpx.Client")
    def test_create_openai_client_from_env(self, mock_client_class):
        """Test OpenAI client creation from environment variables."""
        mock_client_class.return_value = Mock()

        client = create_llm_client(provider="openai")

        assert client.base_url == "https://custom.openai.com"
        assert client.model == "gpt-4"
        assert client.api_key == "sk-test"

    @patch.dict("os.environ", {}, clear=True)
    @patch("src.retrieval.llm.httpx.Client")
    def test_create_openai_client_with_overrides(self, mock_client_class):
        """Test OpenAI client creation with explicit overrides."""
        mock_client_class.return_value = Mock()

        client = create_llm_client(provider="openai", model="gpt-3.5-turbo", api_key="sk-override")

        assert client.model == "gpt-3.5-turbo"
        assert client.api_key == "sk-override"

    @patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}, clear=True)
    @patch("src.retrieval.llm.httpx.Client")
    def test_create_client_case_insensitive(self, mock_client_class):
        """Test provider name is case-insensitive."""
        mock_client_class.return_value = Mock()

        client1 = create_llm_client(provider="OLLAMA")
        client2 = create_llm_client(provider="OpenAI")

        assert client1.base_url == "http://localhost:11434"
        assert client2.base_url == "https://api.openai.com"

    def test_create_client_invalid_provider(self):
        """Test error handling for invalid provider."""
        with pytest.raises(ValueError, match="Unknown provider: invalid. Use 'ollama' or 'openai'"):
            create_llm_client(provider="invalid")

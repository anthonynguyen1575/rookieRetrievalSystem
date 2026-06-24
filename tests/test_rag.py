"""
Unit tests for RAG functionality.

@author: Cyrus Anderson & Anthony Nguyen
Seattle University, ARIN 5360
@version: 3.1.0+w26
"""

from unittest.mock import MagicMock, Mock, patch

import pytest
from fastapi.testclient import TestClient

from retrieval.llm import LLMClient
from retrieval.main import app
from retrieval.rag import RAGSystem
from retrieval.retriever import DocumentRetriever


@pytest.fixture
def mock_retriever():
    """Create a mock DocumentRetriever."""
    retriever = Mock(spec=DocumentRetriever)
    return retriever


@pytest.fixture
def mock_llm_client():
    """Create a mock LLMClient."""
    llm_client = Mock(spec=LLMClient)
    return llm_client


@pytest.fixture
def sample_retrieval_results():
    """Sample documents returned by retriever."""
    return [
        {
            "text": "Python is a high-level programming language.",
            "metadata": {"source": "python_basics.txt"},
            "score": 0.95,
        },
        {
            "text": "Python was created by Guido van Rossum.",
            "metadata": {"source": "python_history.txt"},
            "score": 0.87,
        },
        {
            "text": "Python supports multiple programming paradigms.",
            "metadata": {"source": "python_features.txt"},
            "score": 0.82,
        },
    ]


@pytest.fixture
def rag_system(mock_retriever, mock_llm_client):
    """Create a RAGSystem instance with mocked dependencies."""
    return RAGSystem(
        retriever=mock_retriever,
        llm_client=mock_llm_client,
        n_context_docs=3,
    )


@pytest.fixture
def client():
    """Fixture provides a fresh test client for each test"""
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client


class TestRAGSystemInitialization:
    """Test RAGSystem initialization."""

    def test_init_with_all_params(self, mock_retriever, mock_llm_client):
        rag = RAGSystem(
            retriever=mock_retriever,
            llm_client=mock_llm_client,
            n_context_docs=5,
        )
        assert rag.retriever == mock_retriever
        assert rag.llm_client == mock_llm_client
        assert rag.n_context_docs == 5

    def test_init_without_llm_client(self, mock_retriever):
        rag = RAGSystem(retriever=mock_retriever, n_context_docs=3)
        assert rag.retriever == mock_retriever
        assert rag.llm_client is None
        assert rag.n_context_docs == 3

    def test_init_default_n_context_docs(self, mock_retriever, mock_llm_client):
        rag = RAGSystem(retriever=mock_retriever, llm_client=mock_llm_client)
        assert rag.n_context_docs == 3


class TestContextBuilding:
    """Test context building from retrieval results."""

    def test_build_context_with_results(self, rag_system, sample_retrieval_results):
        """Test building context with sample retrieval results."""
        context = rag_system._build_context(sample_retrieval_results)

        # Check that all documents are included
        assert "[Document 1]" in context
        assert "[Document 2]" in context
        assert "[Document 3]" in context

        # Check that sources are included
        assert "python_basics.txt" in context
        assert "python_history.txt" in context
        assert "python_features.txt" in context

        # Check that text content is included
        assert "high-level programming language" in context
        assert "Guido van Rossum" in context
        assert "multiple programming paradigms" in context

    def test_build_context_empty_results(self, rag_system):
        """Test building context with empty retrieval results."""
        context = rag_system._build_context([])
        assert context == ""

    def test_build_context_with_content_field(self, rag_system):
        """Test that 'content' field is used if 'text' is not present."""
        results = [
            {
                "content": "This uses content field",
                "metadata": {"source": "test.txt"},
                "score": 0.9,
            }
        ]
        context = rag_system._build_context(results)
        assert "This uses content field" in context

    def test_build_context_missing_metadata(self, rag_system):
        """Test handling of missing metadata."""
        results = [{"text": "Text without metadata", "score": 0.9}]
        context = rag_system._build_context(results)
        assert "Unknown" in context
        assert "Text without metadata" in context


class TestPromptCreation:
    """Test prompt creation."""

    def test_create_prompt_structure(self, rag_system):
        """Test creating a prompt with a question and context."""
        question = "What is Python?"
        context = "Python is a programming language."

        prompt = rag_system._create_prompt(question, context)

        # Check that prompt contains key elements
        assert "Context information" in prompt
        assert context in prompt
        assert question in prompt
        assert "Question:" in prompt
        assert "Answer:" in prompt

    def test_create_prompt_with_multiline_context(self, rag_system):
        """Test creating a prompt with a question and multiline context."""
        question = "Test question?"
        context = "Line 1\nLine 2\nLine 3"

        prompt = rag_system._create_prompt(question, context)

        assert "Line 1" in prompt
        assert "Line 2" in prompt
        assert "Line 3" in prompt

    def test_get_system_prompt(self, rag_system):
        """Test getting the system prompt."""
        system_prompt = rag_system._get_system_prompt()

        assert "You are a helpful AI assistant" in system_prompt
        assert "context" in system_prompt.lower()
        assert len(system_prompt) > 0


class TestReadyStateChecking:
    """Test that RAG system checks if LLM is configured before querying."""

    def test_query_without_llm_raises_error(self, mock_retriever):
        rag = RAGSystem(retriever=mock_retriever, llm_client=None)

        with pytest.raises(ValueError, match="LLM client is not configured"):
            rag.query("What is Python?")

    def test_query_with_llm_configured(
        self, rag_system, mock_retriever, mock_llm_client, sample_retrieval_results
    ):
        """Test querying with LLM configured does not raise an error."""
        # Setup mocks
        mock_retriever.search.return_value = sample_retrieval_results
        mock_llm_client.generate.return_value = "Python is a programming language."

        # Should not raise an error
        result = rag_system.query("What is Python?")
        assert result is not None


class TestRAGQuery:
    """Test the main query method."""

    def test_query_full_pipeline(
        self, rag_system, mock_retriever, mock_llm_client, sample_retrieval_results
    ):
        """Test the full query pipeline with retrieval and LLM."""
        # Setup mocks
        mock_retriever.search.return_value = sample_retrieval_results
        expected_answer = "Python is a high-level programming language."
        mock_llm_client.generate.return_value = expected_answer

        # Execute query
        result = rag_system.query("What is Python?")

        # Verify retriever was called
        mock_retriever.search.assert_called_once_with("What is Python?", n_results=3)

        # Verify LLM was called
        assert mock_llm_client.generate.called
        call_args = mock_llm_client.generate.call_args

        # Check that prompt and system_prompt were passed
        assert "prompt" in call_args.kwargs
        assert "system_prompt" in call_args.kwargs
        assert "temperature" in call_args.kwargs

        # Verify result structure
        assert result["question"] == "What is Python?"
        assert result["answer"] == expected_answer
        assert isinstance(result["context"], list)
        assert len(result["context"]) == 3

    def test_query_with_custom_n_results(
        self, rag_system, mock_retriever, mock_llm_client, sample_retrieval_results
    ):
        """Test querying with custom number of results."""
        mock_retriever.search.return_value = sample_retrieval_results[:2]
        mock_llm_client.generate.return_value = "Answer"

        result = rag_system.query("Test question?", n_results=2)

        # Verify retriever was called with custom n_results
        mock_retriever.search.assert_called_once_with("Test question?", n_results=2)
        assert len(result["context"]) == 2

    def test_query_with_custom_temperature(
        self, rag_system, mock_retriever, mock_llm_client, sample_retrieval_results
    ):
        """Test querying with custom temperature."""
        mock_retriever.search.return_value = sample_retrieval_results
        mock_llm_client.generate.return_value = "Answer"

        rag_system.query("Test question?", temperature=0.5)

        # Verify temperature was passed to LLM
        call_args = mock_llm_client.generate.call_args
        assert call_args.kwargs["temperature"] == 0.5

    def test_query_sources_structure(
        self, rag_system, mock_retriever, mock_llm_client, sample_retrieval_results
    ):
        """Test the structure of the sources in the result."""
        mock_retriever.search.return_value = sample_retrieval_results
        mock_llm_client.generate.return_value = "Answer"

        result = rag_system.query("Test question?")

        # Verify sources structure
        for i, source in enumerate(result["context"]):
            assert "source" in source
            assert "text" in source
            # Check for any of the score fields that might be present
            score_fields = ["distance", "bm25_score", "final_score", "rerank_score"]
            assert any(field in source for field in score_fields)
            assert source["source"] == sample_retrieval_results[i]["metadata"]["source"]
            assert source["text"] == sample_retrieval_results[i]["text"]

    def test_query_empty_retrieval_results(self, rag_system, mock_retriever, mock_llm_client):
        """Test building context with empty retrieval results."""
        mock_retriever.search.return_value = []
        mock_llm_client.generate.return_value = "No information found."

        result = rag_system.query("Obscure question?")

        assert result["context"] == []
        assert result["context_count"] == 0
        assert result["answer"] == "No information found."


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_query_with_none_retrieval_results(self, rag_system, mock_retriever, mock_llm_client):
        """Test handling when retriever returns results with missing fields."""
        mock_retriever.search.return_value = [{"text": "Some text", "metadata": {}, "score": None}]
        mock_llm_client.generate.return_value = "Answer"

        result = rag_system.query("Question?")

        assert result["context"][0]["source"] is None
        assert result["context"][0]["text"] == "Some text"
        # Check that score fields are None when not present in retrieval results
        assert result["context"][0]["distance"] is None
        assert result["context"][0]["bm25_score"] is None
        assert result["context"][0]["final_score"] is None


class TestRAGEndpoint:
    """Test RAG Endpoint for prompt structure & enforcement."""

    def test_rag_enforced_prompt_format(self, client):
        """Verify enforced_prompt includes formatting rules"""
        mock_rag = MagicMock()
        mock_rag.query.return_value = {
            "question": "Test?",
            "answer": "Answer",
            "context": [],
        }

        with patch("retrieval.main.rag_system", mock_rag):
            res = client.post("/rag", json={"question": "Test?", "system_prompt": "Be helpful"})
            assert res.status_code == 200

            # Verify the enforced_prompt contains both custom and formatting rules
            call_args = mock_rag.query.call_args[1]
            enforced_prompt = call_args["system_prompt"]
            assert "Be helpful" in enforced_prompt
            assert "Preserve line breaks" in enforced_prompt
            assert "[Document X]" in enforced_prompt

    def test_rag_successful_query_with_all_params(self, client):
        """Verify query contains all parameters"""
        mock_rag = MagicMock()
        mock_rag.query.return_value = {
            "question": "What is AI?",
            "answer": "AI is artificial intelligence",
            "context": [{"text": "AI context", "source": "doc1.txt"}],
        }

        with patch("retrieval.main.rag_system", mock_rag):
            res = client.post(
                "/rag",
                json={
                    "question": "What is AI?",
                    "n_context_docs": 5,
                    "temperature": 0.9,
                    "system_prompt": "Custom prompt",
                    "llm_provider": "openai",
                },
            )
            assert res.status_code == 200
            data = res.json()
            assert data["question"] == "What is AI?"
            assert data["answer"] == "AI is artificial intelligence"
            assert data["context_count"] == 1

            # Verify all params passed to rag_system.query
            mock_rag.query.assert_called_once()
            call_kwargs = mock_rag.query.call_args[1]
            assert call_kwargs["question"] == "What is AI?"
            assert call_kwargs["n_results"] == 5
            assert call_kwargs["temperature"] == 0.9
            assert call_kwargs["llm_provider"] == "openai"


class TestHealthEndpoint:
    """Tests for /health endpointto verify RAG availability"""

    def test_health_when_retriever_none(self, client):
        """Verify retriever is None branch"""
        with patch("retrieval.main.retriever", None):
            res = client.get("/health")
            assert res.status_code == 200
            data = res.json()
            assert data["status"] == "unhealthy"
            assert data["rag_available"] is False

    def test_health_component_rag_when_available(self, client):
        """Verify component == 'rag' and rag_available is True"""
        mock_retriever = MagicMock()
        mock_retriever.document_count = 10
        mock_rag = MagicMock()

        with (
            patch("retrieval.main.retriever", mock_retriever),
            patch("retrieval.main.rag_system", mock_rag),
        ):
            res = client.get("/health?component=rag")
            assert res.status_code == 200
            data = res.json()
            assert data["message"] == "RAG is running and ready"
            assert data["status"] == "healthy"

    def test_health_component_rag_when_unavailable(self, client):
        """Verify component == 'rag' and rag_available is False"""
        mock_retriever = MagicMock()
        mock_retriever.document_count = 10

        with (
            patch("retrieval.main.retriever", mock_retriever),
            patch("retrieval.main.rag_system", None),
        ):
            res = client.get("/health?component=rag")
            assert res.status_code == 200
            data = res.json()
            assert data["message"] == "RAG system not available"
            assert data["status"] == "degraded"

    def test_health_component_search(self, client):
        """Verify component == 'search'"""
        mock_retriever = MagicMock()
        mock_retriever.document_count = 10

        with patch("retrieval.main.retriever", mock_retriever):
            res = client.get("/health?component=search")
            assert res.status_code == 200
            data = res.json()
            assert data["message"] == "Search API is running and ready"
            assert data["status"] == "healthy"

    def test_health_default_with_rag_unavailable(self, client):
        """Verify else branch when rag_available is False"""
        mock_retriever = MagicMock()
        mock_retriever.document_count = 10

        with (
            patch("retrieval.main.retriever", mock_retriever),
            patch("retrieval.main.rag_system", None),
        ):
            res = client.get("/health")
            assert res.status_code == 200
            data = res.json()
            assert data["message"] == "API is running (RAG unavailable)"

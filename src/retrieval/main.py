"""
Lab 4 FastAPI API.

@author: Cyrus Anderson, Anthony Nguyen, Kevin Lundeen
Seattle University, ARIN 5360
@see: https://catalog.seattleu.edu/preview_course_nopop.php?catoid=55&coid=190380
@version: 2.0.0+w26
"""

import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse
from starlette.staticfiles import StaticFiles

from retrieval.llm import LLMClient
from retrieval.rag import RAGSystem
from retrieval.retriever import DocumentRetriever

# Load environment variables from .env file
load_dotenv()

# Set default variables for RAG Health Check
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
PORT = int(os.getenv("PORT", 8000))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global retriever instance
retriever = None
rag_system = None


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str
    documents_indexed: int
    message: str
    rag_available: bool


class SearchRequest(BaseModel):
    """Request model for search."""

    query: str
    n_results: int = 5
    use_hybrid: bool = True
    use_reranking: bool = True


class SearchResponse(BaseModel):
    """Response model for search."""

    query: str
    results: list[dict]
    count: int


class RAGRequest(BaseModel):
    """Request model for RAG query."""

    question: str
    n_context_docs: int = 3
    temperature: float = 0.7
    system_prompt: Optional[str] = None
    llm_provider: Optional[str] = "ollama"


class RAGResponse(BaseModel):
    """Response model for RAG query."""

    question: str
    answer: str
    context: list[dict]
    context_count: int


# Define lifespan function to load models on startup
@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Code before the 'yield' is executed during application startup
    try:
        logger.info("Loading models...")

        # Index documents from the documents/ directory
        global retriever, rag_system
        retriever = DocumentRetriever()
        num_docs = retriever.index_documents("documents")
        logger.info(f"Indexed {num_docs} chunks successfully!")

        # Initialize RAG system with default LLM client
        try:
            llm_client = LLMClient(base_url=OLLAMA_BASE_URL, model=OLLAMA_MODEL)
            rag_system = RAGSystem(retriever=retriever, llm_client=llm_client)
            logger.info("RAG system initialized successfully!")
        except Exception as e:
            logger.warning(f"Failed to initialize RAG system: {str(e)}")
            logger.warning("RAG endpoint will not be available")
            rag_system = None
    except Exception as e:
        # Don't crash the server, but log the error
        logger.error(f"Failed to load model: {str(e)}")

    yield  # The application starts receiving requests after the yield

    # Code after the 'yield' is executed during application shutdown
    logger.info("Application shutting down (lifespan)...")


# Initialize FastAPI app
app = FastAPI(
    title="RAG Playground",
    description="Lab6: RAG Playground API",
    version="1.0.0",
    lifespan=lifespan,
)

# Add cross-origin resource sharing (CORS) middleware
# (gives browser permission to call our API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """
    Search for documents relevant to the query.

    Args:
        request: SearchRequest with query and optional n_results

    Returns:
        SearchResponse with results
    """
    if retriever is None:
        raise HTTPException(status_code=503, detail="Retriever not initialized")

    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    if request.n_results < 1 or request.n_results > 20:
        raise HTTPException(status_code=400, detail="n_results must be between 1 and 20")

    try:
        results = retriever.search(
            request.query,
            request.n_results,
            use_hybrid=request.use_hybrid,
            use_reranking=request.use_reranking,
        )
        return SearchResponse(query=request.query, results=results, count=len(results))
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        raise HTTPException(status_code=500, detail="Search failed")


@app.post("/rag", response_model=RAGResponse)
async def rag_query(request: RAGRequest):
    """
    Answer a question using RAG (Retrieval-Augmented Generation).

    Args:
        request: RAGRequest with question and optional parameters

    Returns:
        RAGResponse with answer and source documents
    """
    if rag_system is None:
        raise HTTPException(
            status_code=503, detail="RAG system not initialized. Please check LLM configuration."
        )

    # Validate question
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    # Validate parameters
    if request.n_context_docs < 1 or request.n_context_docs > 10:
        raise HTTPException(status_code=400, detail="n_context_docs must be between 1 and 10")

    if request.temperature < 0.0 or request.temperature > 2.0:
        raise HTTPException(status_code=400, detail="temperature must be between 0.0 and 2.0")

    try:
        logger.info(f"Processing RAG query: {request.question[:100]}...")

        base_prompt = request.system_prompt or ""

        enforced_prompt = (
            base_prompt
            + """
        Format your answers as follows:
        - Preserve line breaks
        - Use [Document X] to cite sources
        """
        )

        result = rag_system.query(
            question=request.question,
            n_results=request.n_context_docs,
            temperature=request.temperature,
            system_prompt=enforced_prompt,
            llm_provider=request.llm_provider,
        )

        # Extract context documents from sources
        context_docs = result.get("context", [])

        return RAGResponse(
            question=result["question"],
            answer=result["answer"],
            context=context_docs,
            context_count=len(context_docs),
        )
    except ValueError as e:
        logger.error(f"RAG validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"RAG query error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to generate answer. Please try again.")


# Implement health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check(component: Optional[str] = None):
    """
    Check if the API is running.

    Args:
        component: Optional component to check ("search" or "rag")

    Returns:
        Health status including RAG availability
    """
    if retriever is None:
        return HealthResponse(
            status="unhealthy",
            message="Retriever not initialized",
            documents_indexed=0,
            rag_available=False,
        )

    # Check if RAG system is available
    rag_available = rag_system is not None

    # Determine message based on component parameter
    if component == "rag":
        if rag_available:
            message = "RAG is running and ready"
            status = "healthy"
        else:
            message = "RAG system not available"
            status = "degraded"
    elif component == "search":
        message = "Search API is running and ready"
        status = "healthy"
    else:
        # Default message when no component specified
        if rag_available:
            message = "API is running and ready"
        else:
            message = "API is running (RAG unavailable)"
        status = "healthy"

    return HealthResponse(
        status=status,
        message=message,
        documents_indexed=retriever.document_count,
        rag_available=rag_available,
    )


# Add error handler for general exceptions
@app.exception_handler(Exception)
async def general_exception_handler(_request, exc):
    """Handle unexpected exceptions."""
    logger.error(f"Unexpected error: {str(exc)}")
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


# Create a test endpoint that raises exceptions (only for testing!)
@app.get("/test/error")
async def test_error():
    raise RuntimeError("Something went wrong")


# Mount static files LAST - catches all remaining routes
# including / --> /static/index.html, and
#  /stlye.css --> /static/style.css
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    print("To run this application:")
    print("uv run uvicorn src.retrieval.main:app --reload")
    print("\nThen open: http://localhost:8000")

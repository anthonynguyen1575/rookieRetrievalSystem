"""
RAG Functionality.

@author: Cyrus Anderson & Anthony Nguyen
Seattle University, ARIN 5360
@see: https://catalog.seattleu.edu/preview_course_nopop.php?catoid=55&coid=190380
@version: 3.1.0+w26
"""

from typing import Any, Dict, List, Optional

from retrieval.llm import LLMClient, create_llm_client
from retrieval.retriever import DocumentRetriever


class RAGSystem:
    """
    RAG System for Retrieval-Augmented Generation.
    """

    def __init__(
        self,
        retriever: DocumentRetriever,
        llm_client: Optional[LLMClient] = None,
        n_context_docs: int = 3,
    ) -> None:
        """
        Initialize the RAG system with a document retriever and optional LLM client.
        """
        self.retriever = retriever
        self.llm_client = llm_client
        self.n_context_docs = n_context_docs

    def query(
        self,
        question: str,
        n_results: Optional[int] = None,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
        llm_provider: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        RAG Pipeline:
        1. Retrieve relevant documents
        2. Build context
        3. Create prompt
        4. Generate answer
        5. Return answer + sources
        """

        # Create LLM client based on provider if specified
        llm_client = self.llm_client
        if llm_provider:
            llm_client = create_llm_client(provider=llm_provider)

        if llm_client is None:
            raise ValueError("LLM client is not configured.")

        k = n_results if n_results is not None else self.n_context_docs

        # STEP 1: Retrieve
        results = self.retriever.search(question, n_results=k)

        # STEP 2: Build context
        context = self._build_context(results)

        # STEP 3: Build prompt
        prompt = self._create_prompt(question, context)

        # STEP 4: Generate answer
        answer = llm_client.generate(
            prompt=prompt,
            system_prompt=system_prompt or self._get_system_prompt(),
            temperature=temperature,
        )

        # STEP 5: Return structured result with full context documents
        return {
            "question": question,
            "answer": answer,
            "context": [
                {
                    "source": r.get("metadata", {}).get("source")
                    or r.get("metadata", {}).get("filename"),
                    "text": r.get("text", ""),
                    "distance": r.get("distance"),
                    "bm25_score": r.get("bm25_score"),
                    "final_score": r.get("final_score"),
                    "rerank_score": r.get("rerank_score"),
                    "page": r.get("metadata", {}).get("page"),
                    "chunk_index": r.get("metadata", {}).get("chunk")
                    or r.get("metadata", {}).get("chunk_index"),
                }
                for r in results
            ],
            "context_count": len(results),
        }

    # -------------------------
    # Helpers
    # -------------------------

    def _build_context(self, results: List[Dict[str, Any]]) -> str:
        """
        Convert retriever results into a single context string.
        Assumes each result dict contains:
            - "text" or "content"
            - "metadata"
            - optionally "score"
        """

        context_parts = []

        for i, r in enumerate(results, start=1):
            text = r.get("text") or r.get("content") or ""
            source = r.get("metadata", {}).get("source", "Unknown")

            context_parts.append(f"[Document {i}] (Source: {source})\n{text}")

        return "\n\n".join(context_parts)

    def _create_prompt(self, question: str, context: str) -> str:
        return f"""Context information from relevant documents:

{context}

Based only on the context above, answer the question below.
If the context does not contain sufficient information, say so clearly.

Question: {question}

Answer:"""

    def _get_system_prompt(self) -> str:
        return (
            "You are a helpful AI assistant that answers questions based on the provided context documents."
            "Key guidelines:"
            "- Answer directly and concisely"
            "- Cite specific documents when possible"
            "- If the context doesn't contain the answer, say so"
            "- Be honest about uncertainty"
        )

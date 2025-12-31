"""RAG retriever for context-aware retrieval."""
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass

from .knowledge_base import KnowledgeBase

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Result from RAG retrieval."""
    content: str
    score: float
    source: str
    metadata: Dict


class RAGRetriever:
    """
    RAG retriever for context-aware retrieval.
    """
    
    def __init__(
        self,
        knowledge_base: KnowledgeBase,
        min_relevance_score: float = 0.5,
        max_context_length: int = 4000
    ):
        """
        Initializes RAG retriever.
        
        Parameters:
            knowledge_base: Knowledge base instance
            min_relevance_score: Minimum score to include
            max_context_length: Max characters in context
        """
        self.knowledge_base = knowledge_base
        self.min_relevance_score = min_relevance_score
        self.max_context_length = max_context_length
        
        # Retrieval cache
        self._cache: Dict[str, List[RetrievalResult]] = {}
        
        logger.info("RAGRetriever initialized")
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        use_cache: bool = True
    ) -> List[RetrievalResult]:
        """
        Retrieves relevant documents for query.
        
        Parameters:
            query: Search query
            top_k: Maximum results to return
            use_cache: Whether to use cached results
        
        Returns:
            List of RetrievalResults
        """
        # Check cache
        cache_key = f"{query}:{top_k}"
        if use_cache and cache_key in self._cache:
            logger.debug(f"Cache hit for query: {query[:50]}...")
            return self._cache[cache_key]
        
        # Search knowledge base
        results = self.knowledge_base.search(query, top_k=top_k)
        
        # Filter by relevance score
        retrieval_results = []
        for doc in results:
            if doc['score'] >= self.min_relevance_score:
                retrieval_results.append(RetrievalResult(
                    content=doc['content'],
                    score=doc['score'],
                    source=doc.get('metadata', {}).get('source', 'unknown'),
                    metadata=doc.get('metadata', {})
                ))
        
        # Cache results
        if use_cache:
            self._cache[cache_key] = retrieval_results
        
        logger.debug(f"Retrieved {len(retrieval_results)} documents for query")
        return retrieval_results
    
    def retrieve_with_context(
        self,
        query: str,
        conversation_context: str = "",
        top_k: int = 5
    ) -> List[RetrievalResult]:
        """
        Retrieves with conversation context enhancement.
        
        Parameters:
            query: User query
            conversation_context: Recent conversation text
            top_k: Maximum results
        
        Returns:
            List of RetrievalResults
        """
        # Enhance query with context
        if conversation_context:
            enhanced_query = f"{conversation_context}\n\nCurrent question: {query}"
        else:
            enhanced_query = query
        
        return self.retrieve(enhanced_query, top_k=top_k, use_cache=False)
    
    def build_context_string(
        self,
        results: List[RetrievalResult],
        include_sources: bool = True
    ) -> str:
        """
        Builds context string from retrieval results.
        
        Parameters:
            results: Retrieval results
            include_sources: Whether to include source citations
        
        Returns:
            Formatted context string
        """
        if not results:
            return ""
        
        context_parts = []
        total_length = 0
        
        for i, result in enumerate(results, 1):
            content = result.content
            
            # Check length limit
            if total_length + len(content) > self.max_context_length:
                # Truncate last document
                remaining = self.max_context_length - total_length
                if remaining > 100:  # Only include if meaningful
                    content = content[:remaining] + "..."
                else:
                    break
            
            if include_sources:
                source = result.source or f"Document {i}"
                context_parts.append(f"[{source}]\n{content}")
            else:
                context_parts.append(content)
            
            total_length += len(content)
        
        return "\n\n---\n\n".join(context_parts)
    
    def rerank(
        self,
        query: str,
        results: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """
        Re-ranks results using cross-encoder.
        
        Parameters:
            query: Original query
            results: Initial retrieval results
        
        Returns:
            Re-ranked results
        """
        if not results:
            return results
        
        try:
            from sentence_transformers import CrossEncoder
            
            # Use cross-encoder for re-ranking
            model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            
            # Score each result
            pairs = [(query, r.content) for r in results]
            scores = model.predict(pairs)
            
            # Update scores and sort
            for result, score in zip(results, scores):
                result.score = float(score)
            
            return sorted(results, key=lambda x: x.score, reverse=True)
            
        except ImportError:
            logger.warning("Cross-encoder not available, skipping re-ranking")
            return results
    
    def clear_cache(self) -> None:
        """Clear retrieval cache."""
        self._cache.clear()
        logger.debug("Retrieval cache cleared")

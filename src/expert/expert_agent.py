"""Main expert agent coordinating RAG and LLM."""
import logging
from typing import Optional, Dict, List, AsyncGenerator
import asyncio

from .knowledge_base import KnowledgeBase
from .rag_retriever import RAGRetriever
from .context_manager import ContextManager
from .prompt_builder import PromptBuilder
from ..llm.llm_handler import LLMHandler

logger = logging.getLogger(__name__)


class ExpertAgent:
    """
    Main expert agent coordinating RAG and LLM.
    """
    
    def __init__(
        self,
        knowledge_base: KnowledgeBase,
        llm_handler: LLMHandler,
        rag_retriever: Optional[RAGRetriever] = None,
        context_manager: Optional[ContextManager] = None,
        prompt_builder: Optional[PromptBuilder] = None
    ):
        """
        Initializes expert agent.
        
        Parameters:
            knowledge_base: Knowledge base instance
            llm_handler: LLM handler instance
            rag_retriever: Optional RAG retriever
            context_manager: Optional context manager
            prompt_builder: Optional prompt builder
        """
        self.knowledge_base = knowledge_base
        self.llm_handler = llm_handler
        
        # Initialize components
        self.rag_retriever = rag_retriever or RAGRetriever(knowledge_base)
        self.context_manager = context_manager or ContextManager()
        self.prompt_builder = prompt_builder or PromptBuilder()
        
        # Agent state
        self._is_active = False
        self._last_query_time: Optional[float] = None
        self._query_count = 0
        
        logger.info("ExpertAgent initialized")
    
    async def process_query(
        self,
        query: str,
        speaker: str = "user",
        use_rag: bool = True,
        use_context: bool = True
    ) -> str:
        """
        Processes user query and generates response.
        
        Parameters:
            query: User's question
            speaker: Who is asking
            use_rag: Whether to use RAG retrieval
            use_context: Whether to include conversation context
        
        Returns:
            Expert response
        """
        import time
        start_time = time.time()
        
        try:
            # Add question to context
            self.context_manager.add_question(query, speaker)
            
            # Build variables for prompt
            variables = {
                "question": query,
                "speaker": speaker
            }
            
            # Get RAG context if enabled
            if use_rag:
                retrieval_results = self.rag_retriever.retrieve(query)
                knowledge_context = self.rag_retriever.build_context_string(retrieval_results)
                variables["knowledge_context"] = knowledge_context
            
            # Get conversation context if enabled
            if use_context:
                conversation = self.context_manager.get_recent_context(num_entries=10)
                variables["conversation_context"] = conversation
            
            # Build messages
            messages = self.prompt_builder.build_chat_messages(
                template_name="expert_answer",
                variables=variables
            )
            
            # Generate response
            response = await self.llm_handler.generate(messages)
            
            # Add response to context
            self.context_manager.add_answer(response, "assistant")
            
            # Update stats
            self._last_query_time = time.time()
            self._query_count += 1
            
            elapsed = time.time() - start_time
            logger.info(f"Query processed in {elapsed:.2f}s")
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            raise
    
    async def process_query_streaming(
        self,
        query: str,
        speaker: str = "user"
    ) -> AsyncGenerator[str, None]:
        """
        Streams response for lower latency.
        
        Yields:
            Response chunks
        """
        # Add question to context
        self.context_manager.add_question(query, speaker)
        
        # Build variables
        variables = {
            "question": query,
            "speaker": speaker
        }
        
        # Get RAG context
        retrieval_results = self.rag_retriever.retrieve(query)
        variables["knowledge_context"] = self.rag_retriever.build_context_string(retrieval_results)
        
        # Get conversation context
        variables["conversation_context"] = self.context_manager.get_recent_context()
        
        # Build messages
        messages = self.prompt_builder.build_chat_messages(
            template_name="expert_answer",
            variables=variables
        )
        
        # Stream response
        full_response = []
        async for chunk in self.llm_handler.generate_streaming(messages):
            full_response.append(chunk)
            yield chunk
        
        # Add complete response to context
        self.context_manager.add_answer("".join(full_response), "assistant")
    
    def add_transcript(
        self,
        speaker: str,
        content: str
    ) -> None:
        """
        Adds transcript entry to context.
        
        Parameters:
            speaker: Speaker name
            content: Transcript text
        """
        self.context_manager.add_transcript(speaker, content)
    
    async def generate_summary(self) -> str:
        """
        Generates meeting summary.
        
        Returns:
            Summary string
        """
        # Get full context
        context = self.context_manager.get_recent_context(num_entries=50)
        
        # Build summary prompt
        messages = self.prompt_builder.build_chat_messages(
            template_name="summary",
            variables={"conversation_context": context}
        )
        
        return await self.llm_handler.generate(messages)
    
    async def fact_check(
        self,
        statement: str
    ) -> str:
        """
        Fact checks a statement against knowledge base.
        
        Parameters:
            statement: Statement to verify
        
        Returns:
            Fact check result
        """
        # Retrieve relevant knowledge
        results = self.rag_retriever.retrieve(statement, top_k=3)
        knowledge = self.rag_retriever.build_context_string(results)
        
        # Build fact check prompt
        messages = self.prompt_builder.build_chat_messages(
            template_name="fact_check",
            variables={
                "statement": statement,
                "knowledge_context": knowledge
            }
        )
        
        return await self.llm_handler.generate(messages)
    
    def get_stats(self) -> Dict:
        """Get agent statistics."""
        return {
            "is_active": self._is_active,
            "query_count": self._query_count,
            "last_query_time": self._last_query_time,
            "knowledge_base_stats": self.knowledge_base.get_stats(),
            "context_entries": len(self.context_manager._entries)
        }
    
    def reset(self) -> None:
        """Reset agent state."""
        self.context_manager.clear()
        self.rag_retriever.clear_cache()
        self._query_count = 0
        self._last_query_time = None
        logger.info("ExpertAgent reset")

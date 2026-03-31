"""
Multi-provider LLM handler adapter (wraps voice_MVP LLMHandler).
Adds meeting context + RAG context injection on top of the base handler.
"""
import sys
import logging
from pathlib import Path
from typing import List, Dict, Optional

# Import MVP LLM handler
_MVP_PATH = Path(__file__).parent.parent.parent.parent / "voice_engine_MVP" / "src"
if str(_MVP_PATH) not in sys.path:
    sys.path.insert(0, str(_MVP_PATH))

try:
    from llm_handler import LLMHandler as _MVPLLMHandler
except ImportError as e:
    logging.error(f"Failed to import MVP LLMHandler: {e}")
    _MVPLLMHandler = None

logger = logging.getLogger(__name__)


class MeetingLLMHandler:
    """
    LLM handler for Voice_SME_expert.
    Inherits multi-provider fallback from voice_MVP and adds:
      - RAG context injection
      - Meeting transcript context injection
      - Speaker-aware prompt building
    """

    SYSTEM_PROMPT = """You are an AI Subject Matter Expert (SME) assistant 
participating silently in a meeting. You are activated by a wake word.

Rules:
- Answer questions using the provided knowledge base context first.
- Reference the meeting transcript for context about what was discussed.
- Keep responses concise and suitable for spoken voice output (1–3 sentences).
- Address the person who asked by name if known.
- If you don't have enough information, say so briefly."""

    def __init__(self, config):
        """
        Initialize the MeetingLLMHandler.
        
        Args:
            config: AppConfig instance with LLM settings
        """
        if _MVPLLMHandler is None:
            raise ImportError("voice_MVP LLMHandler not available")
        
        self._base = _MVPLLMHandler()
        self._config = config
        self._conversation_history: List[str] = []
        self._system_prompt_added = False
        
        logger.info("✅ MeetingLLMHandler initialized (multi-provider fallback)")

    async def process_query(
        self,
        query: str,
        speaker_id: str = "Speaker",
        knowledge_context: str = "",
        meeting_context: str = "",
    ) -> str:
        """
        Process an expert query with full context injection.
        
        Args:
            query: The user's question
            speaker_id: Who asked (for personalisation)
            knowledge_context: RAG-retrieved knowledge chunks
            meeting_context: Recent meeting transcript
            
        Returns:
            LLM response text
        """
        # Build enriched user message
        parts = []
        
        if knowledge_context:
            parts.append(f"[KNOWLEDGE BASE]\n{knowledge_context}")
        
        if meeting_context:
            parts.append(f"[MEETING CONTEXT]\n{meeting_context}")
        
        parts.append(f"[{speaker_id} ASKS]: {query}")
        enriched_message = "\n\n".join(parts)

        # Add system prompt on first call
        if not self._system_prompt_added:
            self._conversation_history.append(f"System: {self.SYSTEM_PROMPT}")
            self._system_prompt_added = True

        try:
            # Use MVP handler's process method
            if hasattr(self._base, 'process_text_with_history'):
                response = await self._base.process_text_with_history(
                    enriched_message, self._conversation_history
                )
            elif hasattr(self._base, 'process_text'):
                response = await self._base.process_text(enriched_message)
            else:
                logger.error("MVP LLMHandler has no process method")
                return "I'm having trouble processing that. Could you try again?"
            
            # Update conversation history
            self._conversation_history.append(f"User: {query}")
            self._conversation_history.append(f"Agent: {response}")
            
            # Keep history bounded (system prompt + last 18 turns)
            if len(self._conversation_history) > 20:
                self._conversation_history = (
                    self._conversation_history[:1] + self._conversation_history[-18:]
                )
            
            return response
            
        except Exception as e:
            logger.error(f"❌ LLM query error: {e}")
            return "I'm having trouble processing that. Could you try again?"

    async def process_simple(self, text: str) -> str:
        """
        Process a simple text query without context injection.
        
        Args:
            text: The text to process
            
        Returns:
            LLM response text
        """
        try:
            if hasattr(self._base, 'process_text'):
                return await self._base.process_text(text)
            return "I'm having trouble processing that."
        except Exception as e:
            logger.error(f"❌ Simple query error: {e}")
            return "I'm having trouble processing that."

    def clear_history(self) -> None:
        """Clear conversation history."""
        self._conversation_history = []
        self._system_prompt_added = False
        logger.info("🗑️ Conversation history cleared")

    def get_history(self) -> List[str]:
        """Get current conversation history."""
        return self._conversation_history.copy()

    async def shutdown(self) -> None:
        """Shutdown the LLM handler and release resources."""
        try:
            if hasattr(self._base, 'shutdown'):
                await self._base.shutdown()
            logger.info("🔌 MeetingLLMHandler shutdown complete")
        except Exception as e:
            logger.error(f"❌ LLM shutdown error: {e}")

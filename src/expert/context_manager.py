"""Manages conversation and meeting context."""
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class ContextEntry:
    """Single context entry."""
    timestamp: datetime
    speaker: str
    content: str
    entry_type: str = "transcript"  # transcript, question, answer, note
    metadata: Dict = field(default_factory=dict)


class ContextManager:
    """
    Manages conversation and meeting context.
    """
    
    def __init__(
        self,
        max_context_entries: int = 100,
        max_context_tokens: int = 4000,
        summary_threshold: int = 50
    ):
        """
        Initializes context manager.
        
        Parameters:
            max_context_entries: Maximum entries to keep
            max_context_tokens: Maximum tokens in context
            summary_threshold: Entries before summarization
        """
        self.max_context_entries = max_context_entries
        self.max_context_tokens = max_context_tokens
        self.summary_threshold = summary_threshold
        
        # Context storage
        self._entries: deque = deque(maxlen=max_context_entries)
        self._summaries: List[str] = []
        
        # Meeting metadata
        self._meeting_info: Dict = {}
        self._participants: Dict[str, Dict] = {}
        
        # Current topic tracking
        self._current_topic: Optional[str] = None
        self._topic_history: List[str] = []
        
        logger.info("ContextManager initialized")
    
    def add_transcript(
        self,
        speaker: str,
        content: str,
        timestamp: Optional[datetime] = None
    ) -> None:
        """
        Adds transcript entry to context.
        
        Parameters:
            speaker: Speaker name/ID
            content: Transcript text
            timestamp: Entry timestamp
        """
        entry = ContextEntry(
            timestamp=timestamp or datetime.now(),
            speaker=speaker,
            content=content,
            entry_type="transcript"
        )
        
        self._entries.append(entry)
        self._update_participant(speaker)
        
        # Check if summarization needed
        if len(self._entries) >= self.summary_threshold:
            self._maybe_summarize()
    
    def add_question(
        self,
        question: str,
        asker: str = "user"
    ) -> None:
        """
        Adds a question to context.
        
        Parameters:
            question: Question text
            asker: Who asked the question
        """
        entry = ContextEntry(
            timestamp=datetime.now(),
            speaker=asker,
            content=question,
            entry_type="question"
        )
        self._entries.append(entry)
    
    def add_answer(
        self,
        answer: str,
        responder: str = "assistant"
    ) -> None:
        """
        Adds an answer to context.
        
        Parameters:
            answer: Answer text
            responder: Who answered
        """
        entry = ContextEntry(
            timestamp=datetime.now(),
            speaker=responder,
            content=answer,
            entry_type="answer"
        )
        self._entries.append(entry)
    
    def get_recent_context(
        self,
        num_entries: int = 20,
        entry_types: Optional[List[str]] = None
    ) -> str:
        """
        Gets recent context as formatted string.
        
        Parameters:
            num_entries: Number of recent entries
            entry_types: Filter by entry types
        
        Returns:
            Formatted context string
        """
        entries = list(self._entries)
        
        # Filter by type if specified
        if entry_types:
            entries = [e for e in entries if e.entry_type in entry_types]
        
        # Get most recent
        recent = entries[-num_entries:]
        
        # Format
        lines = []
        for entry in recent:
            time_str = entry.timestamp.strftime("%H:%M:%S")
            prefix = f"[{time_str}] {entry.speaker}:"
            lines.append(f"{prefix} {entry.content}")
        
        return "\n".join(lines)
    
    def get_conversation_summary(self) -> str:
        """
        Gets summary of conversation so far.
        
        Returns:
            Summary string
        """
        if self._summaries:
            return "\n\n".join(self._summaries)
        
        # Generate quick summary from recent entries
        return self._generate_quick_summary()
    
    def _generate_quick_summary(self) -> str:
        """Generate quick summary from recent entries."""
        if not self._entries:
            return "No conversation yet."
        
        # Count speakers
        speakers = {}
        for entry in self._entries:
            if entry.entry_type == "transcript":
                speakers[entry.speaker] = speakers.get(entry.speaker, 0) + 1
        
        # Get topics mentioned
        summary_parts = [
            f"Participants: {', '.join(speakers.keys())}",
            f"Total exchanges: {len(self._entries)}"
        ]
        
        if self._current_topic:
            summary_parts.append(f"Current topic: {self._current_topic}")
        
        return "\n".join(summary_parts)
    
    def _maybe_summarize(self) -> None:
        """Summarize older entries if needed."""
        # Simple implementation - just note that summarization happened
        # In production, use LLM to generate actual summary
        if len(self._entries) >= self.max_context_entries * 0.9:
            logger.debug("Context approaching limit, older entries will be discarded")
    
    def _update_participant(self, speaker: str) -> None:
        """Update participant info."""
        if speaker not in self._participants:
            self._participants[speaker] = {
                'first_seen': datetime.now(),
                'message_count': 0
            }
        
        self._participants[speaker]['message_count'] += 1
        self._participants[speaker]['last_seen'] = datetime.now()
    
    def set_meeting_info(self, info: Dict) -> None:
        """Set meeting metadata."""
        self._meeting_info = info
        logger.debug(f"Meeting info set: {info}")
    
    def get_meeting_info(self) -> Dict:
        """Get meeting metadata."""
        return self._meeting_info
    
    def get_participants(self) -> Dict[str, Dict]:
        """Get all participants."""
        return self._participants.copy()
    
    def set_topic(self, topic: str) -> None:
        """Set current discussion topic."""
        if self._current_topic:
            self._topic_history.append(self._current_topic)
        self._current_topic = topic
    
    def get_topic(self) -> Optional[str]:
        """Get current topic."""
        return self._current_topic
    
    def clear(self) -> None:
        """Clear all context."""
        self._entries.clear()
        self._summaries.clear()
        self._participants.clear()
        self._current_topic = None
        self._topic_history.clear()
        logger.info("Context cleared")
    
    def export(self) -> Dict:
        """Export full context for persistence."""
        return {
            'entries': [
                {
                    'timestamp': e.timestamp.isoformat(),
                    'speaker': e.speaker,
                    'content': e.content,
                    'entry_type': e.entry_type,
                    'metadata': e.metadata
                }
                for e in self._entries
            ],
            'summaries': self._summaries,
            'meeting_info': self._meeting_info,
            'participants': self._participants,
            'current_topic': self._current_topic,
            'topic_history': self._topic_history
        }

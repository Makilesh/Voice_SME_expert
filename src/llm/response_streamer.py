"""Streams and processes LLM responses for TTS."""
import logging
from typing import List, Optional
import re

logger = logging.getLogger(__name__)


class ResponseStreamer:
    """
    Streams and processes LLM responses for TTS.
    """
    
    def __init__(
        self,
        sentence_end_tokens: Optional[List[str]] = None,
        min_chunk_length: int = 20
    ):
        """
        Initializes response streamer.
        
        Parameters:
            sentence_end_tokens: Tokens that end sentences
            min_chunk_length: Minimum characters before yielding
        """
        self.sentence_end_tokens = sentence_end_tokens or ['.', '!', '?', ':', ';']
        self.min_chunk_length = min_chunk_length
        
        self._buffer = ""
        self._complete_sentences: List[str] = []
        
        logger.debug("ResponseStreamer initialized")
    
    def process_chunk(self, chunk: str) -> Optional[str]:
        """
        Buffers chunks and returns complete sentences.
        
        Parameters:
            chunk: Text chunk from LLM
        
        Returns:
            string: Complete sentence or None
        """
        self._buffer += chunk
        
        # Look for sentence boundaries
        for token in self.sentence_end_tokens:
            while token in self._buffer:
                # Find the position of the sentence end
                pos = self._buffer.index(token) + len(token)
                
                # Check for common abbreviations that shouldn't end sentences
                before = self._buffer[:pos].strip()
                
                # Skip if it looks like an abbreviation
                if self._is_abbreviation(before):
                    # Move past this token and continue
                    break
                
                # Extract complete sentence
                sentence = self._buffer[:pos].strip()
                self._buffer = self._buffer[pos:].lstrip()
                
                # Only return if long enough
                if len(sentence) >= self.min_chunk_length:
                    self._complete_sentences.append(sentence)
                    return sentence
                elif sentence:
                    # Short sentence - buffer it
                    self._buffer = sentence + " " + self._buffer
        
        return None
    
    def _is_abbreviation(self, text: str) -> bool:
        """Check if text ends with a common abbreviation."""
        abbreviations = [
            'Mr.', 'Mrs.', 'Ms.', 'Dr.', 'Prof.', 'Sr.', 'Jr.',
            'vs.', 'etc.', 'i.e.', 'e.g.', 'a.m.', 'p.m.',
            'Inc.', 'Ltd.', 'Corp.', 'Co.'
        ]
        
        for abbr in abbreviations:
            if text.endswith(abbr):
                return True
        
        # Check for single letter abbreviations (initials)
        if re.search(r'\b[A-Z]\.$', text):
            return True
        
        return False
    
    def flush(self) -> str:
        """
        Flushes any remaining buffered text.
        
        Returns:
            string: Remaining text
        """
        remaining = self._buffer.strip()
        self._buffer = ""
        return remaining
    
    def get_complete_sentences(self) -> List[str]:
        """Get all complete sentences processed so far."""
        return self._complete_sentences.copy()
    
    def reset(self) -> None:
        """Reset streamer state."""
        self._buffer = ""
        self._complete_sentences = []
    
    def has_pending(self) -> bool:
        """Check if there's pending text in buffer."""
        return len(self._buffer.strip()) > 0

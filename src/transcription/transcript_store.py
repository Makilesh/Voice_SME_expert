"""Stores and manages meeting transcripts with speaker labels."""
import logging
import json
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime
import time

logger = logging.getLogger(__name__)


@dataclass
class TranscriptEntry:
    """A single transcript entry."""
    entry_id: int
    text: str
    speaker_id: str
    speaker_name: str
    timestamp: float
    end_timestamp: float
    confidence: float
    created_at: str


class TranscriptStore:
    """
    Stores and manages meeting transcripts with speaker labels.
    """
    
    def __init__(
        self,
        max_entries: int = 10000,
        persistence_path: Optional[str] = None
    ):
        """
        Initializes transcript storage.
        
        Parameters:
            max_entries: Maximum entries to store
            persistence_path: Path to save transcripts (optional)
        """
        self.max_entries = max_entries
        self.persistence_path = Path(persistence_path) if persistence_path else None
        
        self._entries: List[TranscriptEntry] = []
        self._entry_counter = 0
        self._start_time = time.time()
        
        if self.persistence_path:
            self.persistence_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"TranscriptStore initialized: max_entries={max_entries}")
    
    def add_entry(
        self,
        text: str,
        speaker_id: str,
        timestamp: float,
        confidence: float = 1.0,
        end_timestamp: Optional[float] = None,
        speaker_name: Optional[str] = None
    ) -> int:
        """
        Adds transcription entry to store.
        
        Parameters:
            text: Transcribed text
            speaker_id: Speaker identifier
            timestamp: Start timestamp
            confidence: Transcription confidence
            end_timestamp: End timestamp (optional)
            speaker_name: Speaker display name (optional)
        
        Returns:
            int: Entry ID
        """
        if not text.strip():
            return -1
        
        self._entry_counter += 1
        
        entry = TranscriptEntry(
            entry_id=self._entry_counter,
            text=text.strip(),
            speaker_id=speaker_id,
            speaker_name=speaker_name or speaker_id,
            timestamp=timestamp,
            end_timestamp=end_timestamp or timestamp,
            confidence=confidence,
            created_at=datetime.now().isoformat()
        )
        
        self._entries.append(entry)
        
        # Enforce max entries
        if len(self._entries) > self.max_entries:
            self._entries = self._entries[-self.max_entries:]
        
        logger.debug(f"Added entry {entry.entry_id}: [{speaker_id}] {text[:50]}...")
        
        return entry.entry_id
    
    def get_recent(
        self,
        duration_seconds: Optional[float] = None,
        speaker_id: Optional[str] = None,
        max_entries: Optional[int] = None
    ) -> List[Dict]:
        """
        Gets recent transcripts.
        
        Parameters:
            duration_seconds: Time window in seconds (optional)
            speaker_id: Filter by speaker (optional)
            max_entries: Maximum entries to return (optional)
        
        Returns:
            list: Transcript entries
        """
        entries = self._entries.copy()
        
        # Filter by time
        if duration_seconds is not None:
            cutoff = time.time() - duration_seconds
            entries = [e for e in entries if e.timestamp >= cutoff]
        
        # Filter by speaker
        if speaker_id is not None:
            entries = [e for e in entries if e.speaker_id == speaker_id]
        
        # Limit entries
        if max_entries is not None:
            entries = entries[-max_entries:]
        
        return [asdict(e) for e in entries]
    
    def get_by_speaker(self, speaker_id: str) -> List[Dict]:
        """
        Gets all transcripts from specified speaker.
        
        Parameters:
            speaker_id: Speaker identifier
        
        Returns:
            list: Transcript entries for speaker
        """
        entries = [e for e in self._entries if e.speaker_id == speaker_id]
        return [asdict(e) for e in entries]
    
    def get_context_window(
        self,
        num_entries: int = 20,
        max_chars: int = 4000
    ) -> str:
        """
        Gets recent entries formatted for LLM context.
        
        Parameters:
            num_entries: Number of recent entries
            max_chars: Maximum character limit
        
        Returns:
            string: Formatted context
        """
        entries = self._entries[-num_entries:] if self._entries else []
        
        lines = []
        total_chars = 0
        
        for entry in reversed(entries):
            # Format timestamp
            rel_time = entry.timestamp - self._start_time
            minutes = int(rel_time // 60)
            seconds = int(rel_time % 60)
            time_str = f"[{minutes:02d}:{seconds:02d}]"
            
            # Format line
            line = f"{time_str} {entry.speaker_name}: {entry.text}"
            
            if total_chars + len(line) > max_chars:
                break
            
            lines.insert(0, line)
            total_chars += len(line) + 1
        
        return "\n".join(lines)
    
    def search(self, query: str, max_results: int = 10) -> List[Dict]:
        """
        Searches transcripts for relevant content.
        
        Parameters:
            query: Search query
            max_results: Maximum results to return
        
        Returns:
            list: Matching entries
        """
        query_lower = query.lower()
        query_words = query_lower.split()
        
        results = []
        
        for entry in self._entries:
            text_lower = entry.text.lower()
            
            # Simple word matching
            matches = sum(1 for word in query_words if word in text_lower)
            
            if matches > 0:
                results.append({
                    **asdict(entry),
                    'relevance': matches / len(query_words)
                })
        
        # Sort by relevance
        results.sort(key=lambda x: x['relevance'], reverse=True)
        
        return results[:max_results]
    
    def export(
        self,
        format: str = "json",
        output_path: Optional[str] = None
    ) -> bool:
        """
        Exports transcript to file.
        
        Parameters:
            format: Output format (json, txt, srt)
            output_path: Output file path
        
        Returns:
            bool: Success
        """
        if not output_path:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            output_path = f"./outputs/transcript_{timestamp}.{format}"
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            if format == "json":
                data = {
                    'metadata': {
                        'start_time': self._start_time,
                        'entry_count': len(self._entries),
                        'exported_at': datetime.now().isoformat()
                    },
                    'entries': [asdict(e) for e in self._entries]
                }
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
            
            elif format == "txt":
                lines = []
                for entry in self._entries:
                    rel_time = entry.timestamp - self._start_time
                    minutes = int(rel_time // 60)
                    seconds = int(rel_time % 60)
                    lines.append(f"[{minutes:02d}:{seconds:02d}] {entry.speaker_name}: {entry.text}")
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write("\n".join(lines))
            
            elif format == "srt":
                lines = []
                for i, entry in enumerate(self._entries, 1):
                    start_time = self._format_srt_time(entry.timestamp - self._start_time)
                    end_time = self._format_srt_time(entry.end_timestamp - self._start_time)
                    
                    lines.append(str(i))
                    lines.append(f"{start_time} --> {end_time}")
                    lines.append(f"[{entry.speaker_name}] {entry.text}")
                    lines.append("")
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write("\n".join(lines))
            
            else:
                logger.error(f"Unsupported format: {format}")
                return False
            
            logger.info(f"Exported transcript to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Export failed: {e}")
            return False
    
    def _format_srt_time(self, seconds: float) -> str:
        """Format time for SRT files."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
    
    def get_stats(self) -> Dict:
        """Get transcript statistics."""
        if not self._entries:
            return {
                'total_entries': 0,
                'speakers': [],
                'duration_seconds': 0
            }
        
        speakers = {}
        for entry in self._entries:
            if entry.speaker_id not in speakers:
                speakers[entry.speaker_id] = {
                    'name': entry.speaker_name,
                    'entries': 0,
                    'words': 0
                }
            speakers[entry.speaker_id]['entries'] += 1
            speakers[entry.speaker_id]['words'] += len(entry.text.split())
        
        duration = self._entries[-1].timestamp - self._start_time if self._entries else 0
        
        return {
            'total_entries': len(self._entries),
            'speakers': list(speakers.values()),
            'duration_seconds': duration,
            'average_confidence': sum(e.confidence for e in self._entries) / len(self._entries)
        }
    
    def clear(self) -> None:
        """Clear all entries."""
        self._entries.clear()
        self._entry_counter = 0
        logger.info("Transcript store cleared")

"""Tracks and maintains speaker identities across meeting duration."""
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import time

logger = logging.getLogger(__name__)


@dataclass
class SpeakerInfo:
    """Information about a tracked speaker."""
    speaker_id: str
    name: Optional[str] = None
    embedding: Optional[np.ndarray] = None
    embeddings_history: List[np.ndarray] = field(default_factory=list)
    speaking_time: float = 0.0
    segment_count: int = 0
    first_seen: float = 0.0
    last_active: float = 0.0


class SpeakerTracker:
    """
    Tracks and maintains speaker identities across meeting duration.
    """
    
    def __init__(
        self,
        similarity_threshold: float = 0.75,
        max_speakers: int = 10,
        embedding_dim: int = 192
    ):
        """
        Initializes tracker with clustering parameters.
        
        Parameters:
            similarity_threshold: Minimum similarity to match speaker
            max_speakers: Maximum number of speakers to track
            embedding_dim: Dimension of speaker embeddings
        """
        self.similarity_threshold = similarity_threshold
        self.max_speakers = max_speakers
        self.embedding_dim = embedding_dim
        
        self._speakers: Dict[str, SpeakerInfo] = {}
        self._speaker_count = 0
        
        logger.info(f"SpeakerTracker initialized: threshold={similarity_threshold}, max={max_speakers}")
    
    def update(
        self,
        embedding: np.ndarray,
        timestamp: float,
        duration: float = 0.0
    ) -> Tuple[str, bool]:
        """
        Updates tracker with new embedding, returns speaker ID.
        
        Parameters:
            embedding: Speaker embedding vector
            timestamp: Current timestamp
            duration: Duration of the speech segment
        
        Returns:
            tuple: (speaker_id, is_new_speaker)
        """
        if len(embedding) != self.embedding_dim:
            logger.warning(f"Embedding dimension mismatch: {len(embedding)} vs {self.embedding_dim}")
            embedding = np.resize(embedding, self.embedding_dim)
        
        # Find best matching speaker
        best_match_id = None
        best_similarity = 0.0
        
        for speaker_id, speaker_info in self._speakers.items():
            if speaker_info.embedding is not None:
                similarity = self._compute_similarity(embedding, speaker_info.embedding)
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match_id = speaker_id
        
        # Check if match exceeds threshold
        if best_similarity >= self.similarity_threshold and best_match_id:
            # Update existing speaker
            speaker = self._speakers[best_match_id]
            speaker.embeddings_history.append(embedding)
            speaker.last_active = timestamp
            speaker.segment_count += 1
            speaker.speaking_time += duration
            
            # Update centroid embedding (running average)
            if len(speaker.embeddings_history) > 1:
                speaker.embedding = np.mean(speaker.embeddings_history[-10:], axis=0)
            
            logger.debug(f"Matched speaker {best_match_id} (similarity={best_similarity:.3f})")
            return best_match_id, False
        
        # Create new speaker if under limit
        if len(self._speakers) < self.max_speakers:
            self._speaker_count += 1
            new_speaker_id = f"Speaker-{self._speaker_count}"
            
            self._speakers[new_speaker_id] = SpeakerInfo(
                speaker_id=new_speaker_id,
                embedding=embedding,
                embeddings_history=[embedding],
                first_seen=timestamp,
                last_active=timestamp,
                segment_count=1,
                speaking_time=duration
            )
            
            logger.info(f"[New Speaker Detected] {new_speaker_id}")
            return new_speaker_id, True
        
        # Max speakers reached, assign to closest match
        if best_match_id:
            speaker = self._speakers[best_match_id]
            speaker.last_active = timestamp
            speaker.segment_count += 1
            speaker.speaking_time += duration
            
            logger.warning(f"Max speakers reached, assigned to {best_match_id}")
            return best_match_id, False
        
        # Fallback
        return "Speaker-Unknown", False
    
    def _compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity between embeddings."""
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 < 1e-8 or norm2 < 1e-8:
            return 0.0
        
        similarity = np.dot(emb1 / norm1, emb2 / norm2)
        return float((similarity + 1) / 2)  # Scale to 0-1
    
    def get_speaker_stats(self, speaker_id: str) -> Optional[Dict]:
        """
        Returns statistics for specified speaker.
        
        Parameters:
            speaker_id: Speaker identifier
        
        Returns:
            dict: Speaker statistics or None
        """
        if speaker_id not in self._speakers:
            return None
        
        speaker = self._speakers[speaker_id]
        
        return {
            'speaker_id': speaker.speaker_id,
            'name': speaker.name,
            'speaking_time': speaker.speaking_time,
            'segment_count': speaker.segment_count,
            'first_seen': speaker.first_seen,
            'last_active': speaker.last_active
        }
    
    def assign_name(self, speaker_id: str, name: str) -> None:
        """
        Assigns human-readable name to speaker ID.
        
        Parameters:
            speaker_id: Speaker identifier
            name: Human-readable name
        """
        if speaker_id in self._speakers:
            self._speakers[speaker_id].name = name
            logger.info(f"Assigned name '{name}' to {speaker_id}")
        else:
            logger.warning(f"Speaker {speaker_id} not found")
    
    def get_all_speakers(self) -> List[Dict]:
        """
        Returns all tracked speakers.
        
        Returns:
            list: List of speaker information dicts
        """
        speakers = []
        
        for speaker_id, speaker in self._speakers.items():
            speakers.append({
                'speaker_id': speaker.speaker_id,
                'name': speaker.name or "Unknown",
                'speaking_time': speaker.speaking_time,
                'segment_count': speaker.segment_count,
                'first_seen': speaker.first_seen,
                'last_active': speaker.last_active
            })
        
        # Sort by speaking time (descending)
        speakers.sort(key=lambda x: x['speaking_time'], reverse=True)
        
        return speakers
    
    def get_speaker_name(self, speaker_id: str) -> str:
        """
        Get display name for speaker.
        
        Parameters:
            speaker_id: Speaker identifier
        
        Returns:
            str: Display name
        """
        if speaker_id in self._speakers:
            speaker = self._speakers[speaker_id]
            return speaker.name if speaker.name else speaker_id
        return speaker_id
    
    def get_speaker_embedding(self, speaker_id: str) -> Optional[np.ndarray]:
        """
        Get embedding for a speaker.
        
        Parameters:
            speaker_id: Speaker identifier
        
        Returns:
            numpy array or None
        """
        if speaker_id in self._speakers:
            return self._speakers[speaker_id].embedding
        return None
    
    def merge_speakers(self, speaker_id1: str, speaker_id2: str) -> bool:
        """
        Merge two speakers (if determined to be same person).
        
        Parameters:
            speaker_id1: First speaker ID (will be kept)
            speaker_id2: Second speaker ID (will be merged into first)
        
        Returns:
            bool: Success
        """
        if speaker_id1 not in self._speakers or speaker_id2 not in self._speakers:
            return False
        
        speaker1 = self._speakers[speaker_id1]
        speaker2 = self._speakers[speaker_id2]
        
        # Merge embeddings
        speaker1.embeddings_history.extend(speaker2.embeddings_history)
        speaker1.embedding = np.mean(speaker1.embeddings_history[-10:], axis=0)
        
        # Merge stats
        speaker1.speaking_time += speaker2.speaking_time
        speaker1.segment_count += speaker2.segment_count
        speaker1.first_seen = min(speaker1.first_seen, speaker2.first_seen)
        speaker1.last_active = max(speaker1.last_active, speaker2.last_active)
        
        # Remove second speaker
        del self._speakers[speaker_id2]
        
        logger.info(f"Merged {speaker_id2} into {speaker_id1}")
        return True
    
    def reset(self) -> None:
        """Reset all tracked speakers."""
        self._speakers.clear()
        self._speaker_count = 0
        logger.info("Speaker tracker reset")

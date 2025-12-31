"""Real-time speaker diarization to separate and identify speakers."""
import logging
import numpy as np
from typing import List, Dict, Optional, Tuple
import time

from .speaker_embeddings import SpeakerEmbeddingExtractor
from .speaker_tracker import SpeakerTracker

logger = logging.getLogger(__name__)


class SpeakerDiarizer:
    """
    Real-time speaker diarization to separate and identify speakers.
    """
    
    def __init__(
        self,
        embedding_model: str = "speechbrain/spkrec-ecapa-voxceleb",
        similarity_threshold: float = 0.75,
        max_speakers: int = 10,
        min_segment_duration: float = 0.5,
        sample_rate: int = 16000
    ):
        """
        Initializes diarizer with speaker embedding model.
        
        Parameters:
            embedding_model: Model name or path for embeddings
            similarity_threshold: Threshold for speaker matching
            max_speakers: Maximum speakers to track
            min_segment_duration: Minimum segment length in seconds
            sample_rate: Audio sample rate
        """
        self.sample_rate = sample_rate
        self.min_segment_duration = min_segment_duration
        self.min_samples = int(min_segment_duration * sample_rate)
        
        # Initialize components
        self.embedding_extractor = SpeakerEmbeddingExtractor(
            model_path=embedding_model,
            embedding_dim=192
        )
        
        self.speaker_tracker = SpeakerTracker(
            similarity_threshold=similarity_threshold,
            max_speakers=max_speakers,
            embedding_dim=192
        )
        
        # Buffer for accumulating audio
        self._audio_buffer = []
        self._buffer_start_time = 0.0
        
        logger.info(f"SpeakerDiarizer initialized: threshold={similarity_threshold}")
    
    def process_audio(
        self,
        audio_chunk: np.ndarray,
        timestamp: float
    ) -> List[Dict]:
        """
        Processes audio and returns speaker segments.
        
        Parameters:
            audio_chunk: Audio data as numpy array
            timestamp: Current timestamp
        
        Returns:
            list: List of dicts with speaker_id, start_time, end_time, embedding
        """
        segments = []
        
        # Add to buffer
        if len(self._audio_buffer) == 0:
            self._buffer_start_time = timestamp
        
        self._audio_buffer.extend(audio_chunk.tolist())
        
        # Check if buffer has enough audio
        if len(self._audio_buffer) >= self.min_samples:
            # Process buffered audio
            audio_segment = np.array(self._audio_buffer)
            
            # Extract embedding
            start_time = time.perf_counter()
            embedding = self.embedding_extractor.extract_embedding(
                audio_segment, 
                self.sample_rate
            )
            extraction_time = (time.perf_counter() - start_time) * 1000
            
            # Get speaker ID
            duration = len(audio_segment) / self.sample_rate
            speaker_id, is_new = self.speaker_tracker.update(
                embedding,
                timestamp,
                duration
            )
            
            # Create segment
            segment = {
                'speaker_id': speaker_id,
                'start_time': self._buffer_start_time,
                'end_time': timestamp,
                'duration': duration,
                'embedding': embedding,
                'is_new_speaker': is_new,
                'extraction_time_ms': extraction_time
            }
            
            segments.append(segment)
            
            # Clear buffer
            self._audio_buffer = []
            
            logger.debug(f"Segment: {speaker_id} ({duration:.2f}s)")
        
        return segments
    
    def get_speaker_for_segment(
        self,
        audio_segment: np.ndarray
    ) -> Tuple[str, float]:
        """
        Identifies speaker for given audio segment.
        
        Parameters:
            audio_segment: Audio data as numpy array
        
        Returns:
            tuple: (speaker_id, confidence)
        """
        if len(audio_segment) < self.min_samples:
            return "Unknown", 0.0
        
        # Extract embedding
        embedding = self.embedding_extractor.extract_embedding(
            audio_segment,
            self.sample_rate
        )
        
        # Find best match
        best_match = None
        best_similarity = 0.0
        
        for speaker in self.speaker_tracker.get_all_speakers():
            speaker_emb = self.speaker_tracker.get_speaker_embedding(speaker['speaker_id'])
            if speaker_emb is not None:
                similarity = self.embedding_extractor.compare_embeddings(
                    embedding, speaker_emb
                )
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = speaker['speaker_id']
        
        if best_match and best_similarity > 0.5:
            return best_match, best_similarity
        
        return "Unknown", 0.0
    
    def merge_short_segments(
        self,
        segments: List[Dict],
        min_duration: float = 0.3
    ) -> List[Dict]:
        """
        Merges short speaker segments to reduce fragmentation.
        
        Parameters:
            segments: List of segments
            min_duration: Minimum segment duration
        
        Returns:
            list: Merged segments
        """
        if len(segments) <= 1:
            return segments
        
        merged = []
        current = segments[0].copy()
        
        for seg in segments[1:]:
            # Check if same speaker and short gap
            if (seg['speaker_id'] == current['speaker_id'] and
                seg['start_time'] - current['end_time'] < 0.5):
                # Merge
                current['end_time'] = seg['end_time']
                current['duration'] = current['end_time'] - current['start_time']
            else:
                # Only add if long enough
                if current['duration'] >= min_duration:
                    merged.append(current)
                current = seg.copy()
        
        # Add last segment
        if current['duration'] >= min_duration:
            merged.append(current)
        
        return merged
    
    def get_all_speakers(self) -> List[Dict]:
        """Get all tracked speakers."""
        return self.speaker_tracker.get_all_speakers()
    
    def assign_speaker_name(self, speaker_id: str, name: str) -> None:
        """Assign a name to a speaker."""
        self.speaker_tracker.assign_name(speaker_id, name)
    
    def get_speaker_name(self, speaker_id: str) -> str:
        """Get display name for speaker."""
        return self.speaker_tracker.get_speaker_name(speaker_id)
    
    def reset(self) -> None:
        """Reset diarizer state."""
        self._audio_buffer = []
        self._buffer_start_time = 0.0
        self.speaker_tracker.reset()
        logger.info("SpeakerDiarizer reset")

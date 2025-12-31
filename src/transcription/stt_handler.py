"""Main STT handler coordinating transcription with speaker info."""
import logging
import numpy as np
from typing import Dict, List, Optional, AsyncGenerator, Callable
import asyncio
import time

from .streaming_transcriber import StreamingTranscriber
from .transcript_store import TranscriptStore

logger = logging.getLogger(__name__)


class STTHandler:
    """
    Main STT handler coordinating transcription with speaker info.
    """
    
    def __init__(
        self,
        model: str = "base.en",
        language: str = "en",
        compute_type: str = "float32",
        sample_rate: int = 16000
    ):
        """
        Initializes STT with specified model.
        
        Parameters:
            model: Whisper model size
            language: Language code
            compute_type: Compute precision
            sample_rate: Audio sample rate
        """
        self.sample_rate = sample_rate
        
        # Initialize transcriber
        self.transcriber = StreamingTranscriber(
            model=model,
            language=language,
            compute_type=compute_type
        )
        
        # Initialize transcript store
        self.transcript_store = TranscriptStore(max_entries=10000)
        
        # Current segment tracking
        self._current_speaker: Optional[str] = None
        self._segment_start_time: float = 0.0
        self._is_streaming = False
        
        logger.info(f"STTHandler initialized: model={model}")
    
    def transcribe_segment(
        self,
        audio_segment: np.ndarray,
        speaker_id: str,
        speaker_name: Optional[str] = None,
        timestamp: Optional[float] = None
    ) -> Dict:
        """
        Transcribes audio segment with speaker attribution.
        
        Parameters:
            audio_segment: Audio data
            speaker_id: Speaker identifier
            speaker_name: Speaker display name
            timestamp: Segment timestamp
        
        Returns:
            dict: Transcription result
        """
        if timestamp is None:
            timestamp = time.time()
        
        start_time = time.perf_counter()
        
        # Feed audio to transcriber
        self.transcriber.reset()
        self.transcriber.feed_audio(audio_segment)
        text = self.transcriber.get_final_transcript()
        
        processing_time = (time.perf_counter() - start_time) * 1000
        
        # Calculate confidence (simplified)
        confidence = 0.95 if text else 0.0
        
        duration = len(audio_segment) / self.sample_rate
        end_timestamp = timestamp + duration
        
        result = {
            'text': text,
            'speaker_id': speaker_id,
            'speaker_name': speaker_name or speaker_id,
            'start_time': timestamp,
            'end_time': end_timestamp,
            'duration': duration,
            'confidence': confidence,
            'processing_time_ms': processing_time
        }
        
        # Store in transcript
        if text:
            self.transcript_store.add_entry(
                text=text,
                speaker_id=speaker_id,
                timestamp=timestamp,
                confidence=confidence,
                end_timestamp=end_timestamp,
                speaker_name=speaker_name
            )
        
        logger.debug(f"Transcribed [{speaker_id}]: {text[:50]}...")
        
        return result
    
    async def start_streaming(
        self,
        audio_stream: AsyncGenerator[np.ndarray, None],
        diarization_callback: Optional[Callable] = None
    ) -> AsyncGenerator[Dict, None]:
        """
        Starts streaming transcription with speaker diarization.
        
        Parameters:
            audio_stream: Async generator yielding audio chunks
            diarization_callback: Callback to get speaker for segment
        
        Yields:
            dict: Transcription results
        """
        self._is_streaming = True
        audio_buffer = []
        buffer_start_time = time.time()
        min_buffer_duration = 1.0  # seconds
        
        logger.info("Started streaming transcription")
        
        try:
            async for audio_chunk in audio_stream:
                if not self._is_streaming:
                    break
                
                # Accumulate audio
                audio_buffer.extend(audio_chunk.tolist())
                buffer_duration = len(audio_buffer) / self.sample_rate
                
                # Process when enough audio accumulated
                if buffer_duration >= min_buffer_duration:
                    audio_segment = np.array(audio_buffer)
                    
                    # Get speaker from diarization
                    speaker_id = "Speaker-1"
                    speaker_name = None
                    
                    if diarization_callback:
                        try:
                            speaker_id, speaker_name = diarization_callback(audio_segment)
                        except Exception as e:
                            logger.error(f"Diarization callback error: {e}")
                    
                    # Transcribe
                    result = self.transcribe_segment(
                        audio_segment,
                        speaker_id,
                        speaker_name,
                        buffer_start_time
                    )
                    
                    if result['text']:
                        yield result
                    
                    # Reset buffer
                    audio_buffer = []
                    buffer_start_time = time.time()
        
        finally:
            self._is_streaming = False
            logger.info("Stopped streaming transcription")
    
    def stop_streaming(self) -> None:
        """Stop streaming transcription."""
        self._is_streaming = False
    
    def get_full_transcript(self) -> List[Dict]:
        """
        Returns complete meeting transcript.
        
        Returns:
            list: All transcription entries ordered by time
        """
        return self.transcript_store.get_recent()
    
    def get_transcript_text(self) -> str:
        """Get formatted transcript text."""
        return self.transcript_store.get_context_window(num_entries=1000, max_chars=50000)
    
    def search_transcript(self, query: str, max_results: int = 10) -> List[Dict]:
        """Search transcript for relevant content."""
        return self.transcript_store.search(query, max_results)
    
    def export_transcript(self, format: str = "json", output_path: Optional[str] = None) -> bool:
        """Export transcript to file."""
        return self.transcript_store.export(format, output_path)
    
    def get_stats(self) -> Dict:
        """Get transcription statistics."""
        return self.transcript_store.get_stats()
    
    def reset(self) -> None:
        """Reset STT handler."""
        self.transcriber.reset()
        self.transcript_store.clear()
        self._current_speaker = None
        self._is_streaming = False
        logger.info("STTHandler reset")

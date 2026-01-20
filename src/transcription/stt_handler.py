"""Main STT handler coordinating transcription with speaker info."""
import logging
import re
import threading
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
    Enhanced with text corrections and real-time support from voice_engine_MVP.
    """
    
    # Pre-compiled regex patterns for text corrections (from voice_engine_MVP)
    CORRECTIONS = {
        re.compile(r'\b(Shambla Tech|Shambla|Shamlataq|Shamlaq|Shamlata|Samba|Sharma Tech)\b', re.IGNORECASE): 'Shamla Tech',
        re.compile(r'\b(eye services?|I services?|A I services?)\b', re.IGNORECASE): 'AI services',
        re.compile(r'\b(A P I|ay pee eye|a p eye)\b', re.IGNORECASE): 'API',
        re.compile(r'\b(block ?chain)\b', re.IGNORECASE): 'blockchain',
        re.compile(r'\b(crypto ?currency|cripto)\b', re.IGNORECASE): 'cryptocurrency',
        re.compile(r'\bwanna\b', re.IGNORECASE): 'want to',
        re.compile(r'\bgonna\b', re.IGNORECASE): 'going to',
        re.compile(r'\bgotta\b', re.IGNORECASE): 'got to',
        re.compile(r'\blemme\b', re.IGNORECASE): 'let me',
    }
    
    def __init__(
        self,
        model: str = "base.en",
        language: str = "en",
        compute_type: str = "float32",
        sample_rate: int = 16000,
        mode: str = "accurate",
        transcription_timeout: float = 30.0
    ):
        """
        Initializes STT with specified model.
        
        Parameters:
            model: Whisper model size
            language: Language code
            compute_type: Compute precision
            sample_rate: Audio sample rate
            mode: STT mode (fast, balanced, accurate)
            transcription_timeout: Maximum transcription wait time
        """
        self.sample_rate = sample_rate
        self.mode = mode
        self.transcription_timeout = transcription_timeout
        
        # Select model based on mode (from voice_engine_MVP)
        self.model_name = self._select_model(mode, model)
        
        # Initialize transcriber
        self.transcriber = StreamingTranscriber(
            model=self.model_name,
            language=language,
            compute_type=compute_type
        )
        
        # Initialize transcript store
        self.transcript_store = TranscriptStore(max_entries=10000)
        
        # Current segment tracking
        self._current_speaker: Optional[str] = None
        self._segment_start_time: float = 0.0
        self._is_streaming = False
        
        # Real-time transcription state (from voice_engine_MVP)
        self.realtime_text = ""
        self._realtime_lock = threading.Lock()
        
        # TTS stop callback for barge-in
        self.tts_stop_callback: Optional[Callable] = None
        
        # Performance tracking
        self._transcription_count = 0
        self._avg_latency = 0.0
        
        logger.info(f"🎤 STTHandler initialized: model={self.model_name}, mode={mode}")
    
    def _select_model(self, mode: str, default_model: str) -> str:
        """Select model based on mode (from voice_engine_MVP)."""
        models = {
            "fast": "tiny.en",
            "balanced": "small.en",
            "accurate": "base.en"
        }
        return models.get(mode, default_model)
    
    def _apply_corrections(self, text: str) -> str:
        """Apply text corrections using pre-compiled patterns (from voice_engine_MVP)."""
        if not text:
            return text
        
        original = text
        for pattern, replacement in self.CORRECTIONS.items():
            text = pattern.sub(replacement, text)
        
        if original != text:
            logger.debug(f"🔧 Corrected: '{original}' → '{text}'")
        
        return text.strip()
    
    def get_realtime_text(self) -> str:
        """Get current real-time transcription (thread-safe)."""
        with self._realtime_lock:
            return self.realtime_text
    
    def clear_realtime_text(self) -> None:
        """Clear real-time transcription buffer."""
        with self._realtime_lock:
            self.realtime_text = ""
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics (from voice_engine_MVP)."""
        return {
            "model": self.model_name,
            "mode": self.mode,
            "transcription_count": self._transcription_count,
            "avg_latency_ms": round(self._avg_latency, 1),
            "is_streaming": self._is_streaming
        }
    
    def transcribe_segment(
        self,
        audio_segment: np.ndarray,
        speaker_id: str,
        speaker_name: Optional[str] = None,
        timestamp: Optional[float] = None
    ) -> Dict:
        """
        Transcribes audio segment with speaker attribution.
        Enhanced with text corrections from voice_engine_MVP.
        
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
        
        # Apply text corrections (from voice_engine_MVP)
        text = self._apply_corrections(text)
        
        processing_time = (time.perf_counter() - start_time) * 1000
        
        # Update performance stats
        self._transcription_count += 1
        self._avg_latency = ((self._avg_latency * (self._transcription_count - 1)) + processing_time) / self._transcription_count
        
        # Update real-time text
        if text:
            with self._realtime_lock:
                self.realtime_text = text
        
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

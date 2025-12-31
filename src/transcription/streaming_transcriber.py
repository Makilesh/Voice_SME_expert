"""Real-time streaming transcription with partial results."""
import logging
import numpy as np
from typing import Optional, Generator, AsyncGenerator
import asyncio

logger = logging.getLogger(__name__)


class StreamingTranscriber:
    """
    Real-time streaming transcription with partial results.
    """
    
    def __init__(
        self,
        model: str = "base.en",
        language: str = "en",
        compute_type: str = "float32",
        device: str = "cpu"
    ):
        """
        Initializes streaming transcriber.
        
        Parameters:
            model: Whisper model size (tiny, base, small, medium, large)
            language: Language code
            compute_type: Compute precision (float32, float16, int8)
            device: Device to run on (cpu, cuda)
        """
        self.model_name = model
        self.language = language
        self.compute_type = compute_type
        self.device = device
        
        self._model = None
        self._initialized = False
        
        # Audio buffer for streaming
        self._audio_buffer = []
        self._partial_transcript = ""
        self._final_transcript = ""
        
        # Configuration
        self.sample_rate = 16000
        self.min_audio_length = 0.5  # seconds
        self.max_audio_length = 30.0  # seconds
        
        logger.info(f"StreamingTranscriber initialized: model={model}, lang={language}")
    
    def _load_model(self) -> None:
        """Load the transcription model lazily."""
        if self._initialized:
            return
        
        try:
            from faster_whisper import WhisperModel
            
            self._model = WhisperModel(
                self.model_name,
                device=self.device,
                compute_type=self.compute_type
            )
            self._initialized = True
            logger.info(f"Loaded Whisper model: {self.model_name}")
            
        except ImportError:
            logger.warning("faster-whisper not available")
            self._initialized = True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self._initialized = True
    
    def feed_audio(self, audio_chunk: np.ndarray) -> Optional[str]:
        """
        Feeds audio and returns partial transcript if available.
        
        Parameters:
            audio_chunk: Audio data as numpy array
        
        Returns:
            string: Partial transcript or None
        """
        self._load_model()
        
        # Add to buffer
        self._audio_buffer.extend(audio_chunk.tolist())
        
        # Check if enough audio accumulated
        buffer_duration = len(self._audio_buffer) / self.sample_rate
        
        if buffer_duration >= self.min_audio_length:
            # Transcribe current buffer
            audio = np.array(self._audio_buffer)
            
            try:
                if self._model is not None:
                    segments, _ = self._model.transcribe(
                        audio,
                        language=self.language,
                        task="transcribe",
                        beam_size=5,
                        vad_filter=True
                    )
                    
                    # Collect text from segments
                    texts = [seg.text for seg in segments]
                    self._partial_transcript = " ".join(texts).strip()
                else:
                    # Fallback - no transcription
                    self._partial_transcript = ""
                
                return self._partial_transcript if self._partial_transcript else None
                
            except Exception as e:
                logger.error(f"Transcription error: {e}")
                return None
        
        return None
    
    def get_final_transcript(self) -> str:
        """
        Returns finalized transcript when speech ends.
        
        Returns:
            string: Final transcript for current segment
        """
        if not self._audio_buffer:
            return self._final_transcript
        
        self._load_model()
        
        # Transcribe remaining audio
        audio = np.array(self._audio_buffer)
        
        try:
            if self._model is not None and len(audio) > self.sample_rate * 0.3:
                segments, _ = self._model.transcribe(
                    audio,
                    language=self.language,
                    task="transcribe",
                    beam_size=5,
                    vad_filter=True
                )
                
                texts = [seg.text for seg in segments]
                self._final_transcript = " ".join(texts).strip()
            else:
                self._final_transcript = self._partial_transcript
        
        except Exception as e:
            logger.error(f"Final transcription error: {e}")
            self._final_transcript = self._partial_transcript
        
        # Clear buffer
        self._audio_buffer = []
        self._partial_transcript = ""
        
        return self._final_transcript
    
    def reset(self) -> None:
        """
        Resets transcriber state for new segment.
        """
        self._audio_buffer = []
        self._partial_transcript = ""
        self._final_transcript = ""
        logger.debug("Transcriber reset")
    
    def get_buffer_duration(self) -> float:
        """Get current buffer duration in seconds."""
        return len(self._audio_buffer) / self.sample_rate
    
    async def transcribe_stream(
        self,
        audio_stream: AsyncGenerator[np.ndarray, None]
    ) -> AsyncGenerator[str, None]:
        """
        Async generator for streaming transcription.
        
        Parameters:
            audio_stream: Async generator yielding audio chunks
        
        Yields:
            Transcription results
        """
        async for chunk in audio_stream:
            result = self.feed_audio(chunk)
            if result:
                yield result
        
        # Get final result
        final = self.get_final_transcript()
        if final:
            yield final

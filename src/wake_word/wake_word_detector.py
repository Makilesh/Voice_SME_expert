"""Always-on wake word detection with low latency."""
import logging
import numpy as np
from typing import List, Callable, Optional, Tuple, AsyncGenerator
import asyncio
import time

from .keyword_spotter import KeywordSpotter

logger = logging.getLogger(__name__)


class WakeWordDetector:
    """
    Always-on wake word detection with low latency.
    """
    
    def __init__(
        self,
        wake_phrases: List[str],
        sensitivity: float = 0.5,
        model_path: Optional[str] = None,
        sample_rate: int = 16000
    ):
        """
        Initializes wake word detection.
        
        Parameters:
            wake_phrases: List of wake phrases to detect
            sensitivity: Detection sensitivity (0-1)
            model_path: Path to custom model
            sample_rate: Audio sample rate
        """
        self.wake_phrases = wake_phrases
        self.sensitivity = sensitivity
        self.model_path = model_path
        self.sample_rate = sample_rate
        
        # Initialize keyword spotter
        self.spotter = KeywordSpotter(
            keywords=wake_phrases,
            model_type="openwakeword",
            sample_rate=sample_rate
        )
        
        # Detection state
        self._is_detecting = False
        self._callback: Optional[Callable] = None
        self._last_detection_time = 0.0
        self._cooldown_seconds = 2.0  # Minimum time between detections
        
        # Audio buffer for context after wake word
        self._post_wake_buffer: List[np.ndarray] = []
        self._post_wake_duration = 0.5  # seconds to capture after wake word
        
        logger.info(f"WakeWordDetector initialized: {wake_phrases}")
    
    def start_detection(
        self,
        audio_stream: AsyncGenerator[np.ndarray, None],
        callback: Callable[[str, np.ndarray], None]
    ) -> asyncio.Task:
        """
        Starts continuous wake word detection.
        
        Parameters:
            audio_stream: Async generator yielding audio chunks
            callback: Function called when wake word detected
                     (wake_phrase, audio_after_wake)
        
        Returns:
            asyncio.Task: The detection task
        """
        self._is_detecting = True
        self._callback = callback
        
        return asyncio.create_task(self._detection_loop(audio_stream))
    
    async def _detection_loop(
        self,
        audio_stream: AsyncGenerator[np.ndarray, None]
    ) -> None:
        """Main detection loop."""
        logger.info("Wake word detection started")
        
        try:
            async for audio_chunk in audio_stream:
                if not self._is_detecting:
                    break
                
                # Check for wake word
                detected, confidence, phrase = self.is_wake_word(audio_chunk)
                
                if detected:
                    current_time = time.time()
                    
                    # Check cooldown
                    if current_time - self._last_detection_time >= self._cooldown_seconds:
                        self._last_detection_time = current_time
                        
                        logger.info(f"[Wake Word Detected] '{phrase}' (confidence={confidence:.2f})")
                        
                        # Capture post-wake audio
                        post_wake_audio = await self._capture_post_wake_audio(audio_stream)
                        
                        # Call callback
                        if self._callback:
                            try:
                                self._callback(phrase, post_wake_audio)
                            except Exception as e:
                                logger.error(f"Callback error: {e}")
        
        except asyncio.CancelledError:
            logger.info("Wake word detection cancelled")
        except Exception as e:
            logger.error(f"Detection loop error: {e}")
        finally:
            self._is_detecting = False
            logger.info("Wake word detection stopped")
    
    async def _capture_post_wake_audio(
        self,
        audio_stream: AsyncGenerator[np.ndarray, None]
    ) -> np.ndarray:
        """Capture audio after wake word for query."""
        chunks = []
        target_samples = int(self._post_wake_duration * self.sample_rate)
        collected_samples = 0
        
        try:
            async for chunk in audio_stream:
                chunks.append(chunk)
                collected_samples += len(chunk)
                
                if collected_samples >= target_samples:
                    break
        except Exception as e:
            logger.error(f"Error capturing post-wake audio: {e}")
        
        if chunks:
            return np.concatenate(chunks)
        return np.array([])
    
    def stop_detection(self) -> None:
        """
        Stops wake word detection.
        """
        self._is_detecting = False
        self._callback = None
        logger.info("Wake word detection stopping...")
    
    def is_wake_word(
        self,
        audio_chunk: np.ndarray
    ) -> Tuple[bool, float, str]:
        """
        Checks if audio contains wake word.
        
        Parameters:
            audio_chunk: Audio data
        
        Returns:
            tuple: (detected, confidence, matched_phrase)
        """
        timestamp = time.time()
        
        # Process through keyword spotter
        detections = self.spotter.process_frame(audio_chunk, timestamp)
        
        # Check for valid detections
        for detection in detections:
            confidence = detection['confidence']
            
            # Apply sensitivity
            threshold = 1.0 - self.sensitivity
            
            if confidence >= threshold:
                return True, confidence, detection['keyword']
        
        return False, 0.0, ""
    
    def set_sensitivity(self, sensitivity: float) -> None:
        """
        Adjusts detection sensitivity.
        
        Parameters:
            sensitivity: Sensitivity value 0-1 (higher = more sensitive)
        """
        if not 0 <= sensitivity <= 1:
            raise ValueError("Sensitivity must be between 0 and 1")
        
        self.sensitivity = sensitivity
        logger.info(f"Wake word sensitivity set to {sensitivity}")
    
    def add_wake_phrase(self, phrase: str) -> bool:
        """Add a wake phrase."""
        if phrase not in self.wake_phrases:
            self.wake_phrases.append(phrase)
            self.spotter.add_keyword(phrase)
            return True
        return False
    
    def remove_wake_phrase(self, phrase: str) -> bool:
        """Remove a wake phrase."""
        if phrase in self.wake_phrases:
            self.wake_phrases.remove(phrase)
            self.spotter.remove_keyword(phrase)
            return True
        return False
    
    def get_wake_phrases(self) -> List[str]:
        """Get current wake phrases."""
        return self.wake_phrases.copy()
    
    def is_detecting(self) -> bool:
        """Check if currently detecting."""
        return self._is_detecting

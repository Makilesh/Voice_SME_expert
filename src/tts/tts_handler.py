"""Main TTS handler coordinating synthesis and playback with barge-in support."""
import logging
import threading
import time
from typing import Optional, Callable, AsyncGenerator
import asyncio
import numpy as np

from .cartesia_engine import CartesiaEngine
from .audio_player import AudioPlayer

logger = logging.getLogger(__name__)


class TTSHandler:
    """
    Main TTS handler coordinating synthesis and playback.
    Enhanced with barge-in detection from voice_engine_MVP.
    """
    
    def __init__(
        self,
        cartesia_api_key: str = "",
        cartesia_voice_id: str = "e07c00bc-4134-4eae-9ea4-1a55fb45746b",  # Brooke voice
        cartesia_model: str = "sonic-3",  # Upgraded to sonic-3
        sample_rate: int = 22050,  # Cartesia optimal sample rate
        output_device_id: Optional[int] = None,
        stt_handler=None,  # Reference to STT for barge-in detection
        barge_in_enabled: bool = True,
        barge_in_startup_buffer: float = 0.15,
        barge_in_check_interval: float = 0.02,
        barge_in_min_chars: int = 2
    ):
        """
        Initializes TTS handler.
        
        Parameters:
            cartesia_api_key: Cartesia API key
            cartesia_voice_id: Voice to use
            cartesia_model: TTS model
            sample_rate: Audio sample rate
            output_device_id: Output device
            stt_handler: STT handler for barge-in detection
            barge_in_enabled: Enable barge-in detection
            barge_in_startup_buffer: Ignore input for first N seconds
            barge_in_check_interval: How often to check for barge-in
            barge_in_min_chars: Minimum chars to trigger barge-in
        """
        self.sample_rate = sample_rate
        
        # Barge-in configuration (from voice_engine_MVP)
        self._stt_handler = stt_handler
        self._barge_in_enabled = barge_in_enabled
        self._barge_in_startup_buffer = barge_in_startup_buffer
        self._barge_in_check_interval = barge_in_check_interval
        self._barge_in_min_chars = barge_in_min_chars
        self._barge_in_detected = False
        self._last_seen_realtime_text = ""
        
        # Thread-safe state management
        self._state_lock = threading.Lock()
        self._stop_event = threading.Event()
        
        # Initialize Cartesia engine
        self._cartesia = None
        if cartesia_api_key:
            self._cartesia = CartesiaEngine(
                api_key=cartesia_api_key,
                voice_id=cartesia_voice_id,
                model=cartesia_model,
                sample_rate=sample_rate
            )
        
        # Initialize audio player
        self._player = AudioPlayer(
            device_id=output_device_id,
            sample_rate=sample_rate
        )
        
        # State
        self._is_speaking = False
        self._current_text: Optional[str] = None
        self._playback_start_time: float = 0.0
        
        # Callbacks
        self._on_start: Optional[Callable] = None
        self._on_complete: Optional[Callable] = None
        
        # Fallback TTS
        self._use_fallback = not cartesia_api_key
        
        logger.info(f"TTSHandler initialized (barge-in: {'enabled' if barge_in_enabled else 'disabled'})")
    
    async def speak(self, text: str, blocking: bool = False) -> None:
        """
        Synthesizes and plays text with optional barge-in detection.
        
        Parameters:
            text: Text to speak
            blocking: Whether to block until complete
            enable_barge_in: Whether to enable barge-in detection
        """
        if not text.strip():
            return
        
        with self._state_lock:
            self._is_speaking = True
            self._current_text = text
            self._barge_in_detected = False
            self._stop_event.clear()
            self._playback_start_time = time.time()
        
        if self._on_start:
            self._on_start(text)
        
        try:
            if self._cartesia and not self._use_fallback:
                # Use Cartesia
                audio = await self._cartesia.synthesize(text)
                self._player.play(audio)
            else:
                # Use fallback TTS
                audio = self._fallback_synthesize(text)
                if audio is not None:
                    self._player.play(audio)
            
            if blocking:
                # Wait for playback to complete
                while self._player.is_playing():
                    await asyncio.sleep(0.1)
                    
        except Exception as e:
            logger.error(f"TTS error: {e}")
            raise
        finally:
            with self._state_lock:
                self._is_speaking = False
                self._current_text = None
            
            if self._on_complete:
                self._on_complete()
    
    def _check_barge_in(self) -> bool:
        """
        Check for barge-in using STT real-time text (from voice_engine_MVP).
        Returns True if user is speaking during TTS playback.
        """
        if not self._barge_in_enabled or not self._stt_handler:
            return False
        
        # Skip check during startup buffer
        elapsed = time.time() - self._playback_start_time
        if elapsed < self._barge_in_startup_buffer:
            return False
        
        try:
            # Get current real-time text from STT
            current_text = ""
            if hasattr(self._stt_handler, 'get_realtime_text'):
                current_text = self._stt_handler.get_realtime_text()
            elif hasattr(self._stt_handler, 'realtime_text'):
                current_text = self._stt_handler.realtime_text
            
            if not current_text:
                self._last_seen_realtime_text = ""
                return False
            
            # Check if new text appeared
            if current_text != self._last_seen_realtime_text:
                new_chars = len(current_text) - len(self._last_seen_realtime_text)
                if new_chars >= self._barge_in_min_chars:
                    logger.info(f"🎤 Barge-in detected: '{current_text[:30]}...'")
                    self._barge_in_detected = True
                    self._last_seen_realtime_text = current_text
                    return True
                self._last_seen_realtime_text = current_text
            
            return False
            
        except Exception as e:
            logger.debug(f"Barge-in check error: {e}")
            return False
    
    def wait_for_completion(self, timeout: float = 30.0) -> bool:
        """
        Wait for TTS playback to complete or barge-in (from voice_engine_MVP).
        
        Parameters:
            timeout: Maximum wait time in seconds
        
        Returns:
            True if completed normally, False if interrupted by barge-in
        """
        start_time = time.time()
        
        while self._player.is_playing():
            # Check for barge-in
            if self._check_barge_in():
                self.stop_playback()
                return False
            
            # Check for stop event
            if self._stop_event.is_set():
                return False
            
            # Check timeout
            if time.time() - start_time > timeout:
                logger.warning(f"⏱️ TTS playback timeout after {timeout}s")
                self.stop_playback()
                return False
            
            time.sleep(self._barge_in_check_interval)
        
        return True
    
    def stop_playback(self) -> None:
        """Immediately stop audio playback."""
        self._stop_event.set()
        self._player.stop()
        with self._state_lock:
            self._is_speaking = False
        logger.debug("🛑 Playback stopped")
    
    def was_barge_in(self) -> bool:
        """Check if last playback was interrupted by barge-in."""
        return self._barge_in_detected
    
    async def speak_streaming(
        self,
        text_generator: AsyncGenerator[str, None]
    ) -> None:
        """
        Streams text-to-speech for low latency.
        
        Parameters:
            text_generator: Async generator yielding text chunks
        """
        self._is_speaking = True
        
        if self._on_start:
            self._on_start("")
        
        try:
            buffer = ""
            
            async for chunk in text_generator:
                buffer += chunk
                
                # Check for sentence boundaries
                for delimiter in ['.', '!', '?']:
                    if delimiter in buffer:
                        idx = buffer.rfind(delimiter) + 1
                        sentence = buffer[:idx].strip()
                        buffer = buffer[idx:]
                        
                        if sentence:
                            if self._cartesia and not self._use_fallback:
                                audio = await self._cartesia.synthesize(sentence)
                                self._player.queue_audio(audio)
                            else:
                                audio = self._fallback_synthesize(sentence)
                                if audio is not None:
                                    self._player.queue_audio(audio)
                        break
            
            # Handle remaining text
            if buffer.strip():
                if self._cartesia and not self._use_fallback:
                    audio = await self._cartesia.synthesize(buffer)
                    self._player.queue_audio(audio)
                else:
                    audio = self._fallback_synthesize(buffer)
                    if audio is not None:
                        self._player.queue_audio(audio)
                        
        except Exception as e:
            logger.error(f"Streaming TTS error: {e}")
            raise
        finally:
            self._is_speaking = False
            
            if self._on_complete:
                self._on_complete()
    
    def _fallback_synthesize(self, text: str) -> Optional[np.ndarray]:
        """Fallback TTS using pyttsx3."""
        try:
            import pyttsx3
            import io
            import wave
            
            engine = pyttsx3.init()
            
            # This is a simplified fallback - pyttsx3 doesn't easily
            # return audio data, so we just log for now
            logger.warning("Fallback TTS: pyttsx3 playback")
            engine.say(text)
            engine.runAndWait()
            
            return None  # pyttsx3 plays directly
            
        except ImportError:
            logger.error("No TTS engine available (pyttsx3 not installed)")
            return None
        except Exception as e:
            logger.error(f"Fallback TTS error: {e}")
            return None
    
    def stop(self) -> None:
        """Stops current speech."""
        self._player.stop()
        self._is_speaking = False
        self._current_text = None
        logger.debug("Speech stopped")
    
    def pause(self) -> None:
        """Pauses speech."""
        self._player.pause()
    
    def resume(self) -> None:
        """Resumes speech."""
        self._player.resume()
    
    def is_speaking(self) -> bool:
        """Check if currently speaking."""
        return self._is_speaking or self._player.is_playing()
    
    def set_voice(self, voice_id: str) -> None:
        """Change voice."""
        if self._cartesia:
            self._cartesia.set_voice(voice_id)
    
    async def list_voices(self) -> list:
        """List available voices."""
        if self._cartesia:
            return await self._cartesia.list_voices()
        return []
    
    def set_output_device(self, device_id: int) -> None:
        """Change output device."""
        self._player.set_device(device_id)
    
    def set_on_start(self, callback: Callable) -> None:
        """Set speech start callback."""
        self._on_start = callback
    
    def set_on_complete(self, callback: Callable) -> None:
        """Set speech complete callback."""
        self._on_complete = callback
        self._player.set_on_complete(callback)

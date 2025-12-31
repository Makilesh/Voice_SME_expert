"""Main TTS handler coordinating synthesis and playback."""
import logging
from typing import Optional, Callable, AsyncGenerator
import asyncio
import numpy as np

from .cartesia_engine import CartesiaEngine
from .audio_player import AudioPlayer

logger = logging.getLogger(__name__)


class TTSHandler:
    """
    Main TTS handler coordinating synthesis and playback.
    """
    
    def __init__(
        self,
        cartesia_api_key: str = "",
        cartesia_voice_id: str = "default",
        cartesia_model: str = "sonic-english",
        sample_rate: int = 24000,
        output_device_id: Optional[int] = None
    ):
        """
        Initializes TTS handler.
        
        Parameters:
            cartesia_api_key: Cartesia API key
            cartesia_voice_id: Voice to use
            cartesia_model: TTS model
            sample_rate: Audio sample rate
            output_device_id: Output device
        """
        self.sample_rate = sample_rate
        
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
        
        # Callbacks
        self._on_start: Optional[Callable] = None
        self._on_complete: Optional[Callable] = None
        
        # Fallback TTS
        self._use_fallback = not cartesia_api_key
        
        logger.info("TTSHandler initialized")
    
    async def speak(self, text: str, blocking: bool = False) -> None:
        """
        Synthesizes and plays text.
        
        Parameters:
            text: Text to speak
            blocking: Whether to block until complete
        """
        if not text.strip():
            return
        
        self._is_speaking = True
        self._current_text = text
        
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
            self._is_speaking = False
            self._current_text = None
            
            if self._on_complete:
                self._on_complete()
    
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

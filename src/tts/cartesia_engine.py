"""Cartesia TTS engine integration with optimizations from voice_engine_MVP."""
import logging
import time
from typing import Optional, AsyncGenerator, Callable
import asyncio
import numpy as np
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class PlaybackState(Enum):
    """TTS playback states for thread-safe management."""
    IDLE = "idle"
    STREAMING = "streaming"
    PLAYING = "playing"
    STOPPED = "stopped"


@dataclass
class VoiceConfig:
    """Voice configuration for Cartesia TTS."""
    voice_id: str = "e07c00bc-4134-4eae-9ea4-1a55fb45746b"  # Brooke
    model: str = "sonic-3"  # sonic-3 for best quality
    language: str = "en"
    
    def to_cartesia_voice(self) -> dict:
        """Convert to Cartesia API voice format."""
        return {
            "mode": "id",
            "id": self.voice_id
        }


@dataclass
class AudioOutputConfig:
    """Audio configuration for Cartesia TTS."""
    sample_rate: int = 22050  # Cartesia optimal: 22050, 24000
    channels: int = 1
    encoding: str = "pcm_f32le"
    container: str = "raw"
    
    def to_cartesia_format(self) -> dict:
        """Convert to Cartesia API output format."""
        return {
            "container": self.container,
            "encoding": self.encoding,
            "sample_rate": self.sample_rate
        }


class CartesiaEngine:
    """
    Cartesia TTS engine for ultra-low-latency speech synthesis.
    Enhanced with optimizations from voice_engine_MVP.
    """
    
    def __init__(
        self,
        api_key: str,
        voice_id: str = "e07c00bc-4134-4eae-9ea4-1a55fb45746b",  # Brooke
        model: str = "sonic-3",  # Use sonic-3 for best quality
        sample_rate: int = 22050,  # Cartesia optimal sample rate
        voice_config: Optional[VoiceConfig] = None,
        audio_config: Optional[AudioOutputConfig] = None
    ):
        """
        Initializes Cartesia TTS engine.
        
        Parameters:
            api_key: Cartesia API key
            voice_id: Voice identifier
            model: TTS model name
            sample_rate: Output sample rate
            voice_config: Optional voice configuration
            audio_config: Optional audio output configuration
        """
        self.api_key = api_key
        self.voice_id = voice_id
        self.model = model
        self.sample_rate = sample_rate
        
        # Configuration (from voice_engine_MVP)
        self.voice_config = voice_config or VoiceConfig(voice_id=voice_id, model=model)
        self.audio_config = audio_config or AudioOutputConfig(sample_rate=sample_rate)
        
        self._client = None
        self._async_client = None
        self._state = PlaybackState.IDLE
        
        # Performance monitoring
        self._stats = {
            "synthesis_count": 0,
            "total_latency_ms": 0,
            "avg_latency_ms": 0
        }
        
        # Barge-in callback (set by TTS handler)
        self.barge_in_callback: Optional[Callable[[], bool]] = None
        
        logger.info(f"🎤 CartesiaEngine initialized: model={model}, voice={voice_id}, sample_rate={sample_rate}Hz")
    
    def _get_client(self):
        """Get or create Cartesia client."""
        if self._client is None:
            try:
                from cartesia import Cartesia
                self._client = Cartesia(api_key=self.api_key)
            except ImportError:
                logger.error("cartesia package not installed")
                raise
        return self._client
    
    async def _get_async_client(self):
        """Get or create async Cartesia client."""
        if self._async_client is None:
            try:
                from cartesia import AsyncCartesia
                self._async_client = AsyncCartesia(api_key=self.api_key)
            except ImportError:
                # Fall back to sync client wrapper
                return self._get_client()
        return self._async_client
    
    async def synthesize(self, text: str) -> np.ndarray:
        """
        Synthesizes speech from text with performance monitoring.
        
        Parameters:
            text: Text to synthesize
        
        Returns:
            Audio samples as numpy array
        """
        start_time = time.time()
        self._state = PlaybackState.STREAMING
        
        try:
            client = self._get_client()
            
            # Generate speech
            audio_data = await asyncio.to_thread(
                self._synthesize_sync,
                text
            )
            
            # Update stats
            latency_ms = (time.time() - start_time) * 1000
            self._stats["synthesis_count"] += 1
            self._stats["total_latency_ms"] += latency_ms
            self._stats["avg_latency_ms"] = self._stats["total_latency_ms"] / self._stats["synthesis_count"]
            
            logger.debug(f"✅ Synthesized in {latency_ms:.0f}ms: '{text[:30]}...'")
            
            self._state = PlaybackState.IDLE
            return audio_data
            
        except Exception as e:
            self._state = PlaybackState.IDLE
            logger.error(f"❌ Synthesis error: {e}")
            raise
    
    def _synthesize_sync(self, text: str) -> np.ndarray:
        """Synchronous synthesis with voice_engine_MVP format."""
        client = self._get_client()
        
        response = client.tts.sse(
            model_id=self.voice_config.model,
            transcript=text,
            voice=self.voice_config.to_cartesia_voice(),
            output_format=self.audio_config.to_cartesia_format()
        )
        
        # Collect audio chunks
        audio_chunks = []
        first_chunk = True
        first_chunk_time = None
        
        for chunk in response:
            # Check for barge-in
            if self.barge_in_callback and self.barge_in_callback():
                logger.info("🎤 Barge-in detected during synthesis")
                break
                
            if hasattr(chunk, 'audio') and chunk.audio:
                if first_chunk:
                    first_chunk_time = time.time()
                    first_chunk = False
                audio_chunks.append(chunk.audio)
        
        # Combine and convert to numpy
        audio_bytes = b''.join(audio_chunks)
        audio_array = np.frombuffer(audio_bytes, dtype=np.float32)
        
        return audio_array
    
    async def synthesize_streaming(
        self,
        text: str
    ) -> AsyncGenerator[np.ndarray, None]:
        """
        Streams synthesized audio chunks with barge-in support.
        
        Parameters:
            text: Text to synthesize
        
        Yields:
            Audio chunks as numpy arrays
        """
        self._state = PlaybackState.STREAMING
        start_time = time.time()
        first_chunk = True
        
        try:
            client = self._get_client()
            
            response = client.tts.sse(
                model_id=self.voice_config.model,
                transcript=text,
                voice=self.voice_config.to_cartesia_voice(),
                output_format=self.audio_config.to_cartesia_format()
            )
            
            for chunk in response:
                # Check for barge-in
                if self.barge_in_callback and self.barge_in_callback():
                    logger.info("🎤 Barge-in detected during streaming")
                    break
                    
                if hasattr(chunk, 'audio') and chunk.audio:
                    if first_chunk:
                        latency_ms = (time.time() - start_time) * 1000
                        logger.debug(f"🎵 First chunk in {latency_ms:.0f}ms")
                        first_chunk = False
                    audio_array = np.frombuffer(chunk.audio, dtype=np.float32)
                    yield audio_array
                    
        except Exception as e:
            logger.error(f"❌ Streaming synthesis error: {e}")
            raise
        finally:
            self._state = PlaybackState.IDLE
    
    def set_barge_in_callback(self, callback: Callable[[], bool]) -> None:
        """Set callback for barge-in detection."""
        self.barge_in_callback = callback
    
    def get_stats(self) -> dict:
        """Get synthesis performance statistics."""
        return self._stats.copy()
    
    def get_state(self) -> PlaybackState:
        """Get current playback state."""
        return self._state
    
    def set_voice(self, voice_id: str) -> None:
        """Change voice."""
        self.voice_id = voice_id
        logger.info(f"Voice changed to: {voice_id}")
    
    async def list_voices(self) -> list:
        """List available voices."""
        try:
            client = self._get_client()
            voices = await asyncio.to_thread(client.voices.list)
            return [
                {'id': v.id, 'name': v.name, 'language': v.language}
                for v in voices
            ]
        except Exception as e:
            logger.error(f"Error listing voices: {e}")
            return []
    
    def is_available(self) -> bool:
        """Check if Cartesia is available."""
        return bool(self.api_key)

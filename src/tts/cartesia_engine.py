"""Cartesia TTS engine integration."""
import logging
from typing import Optional, AsyncGenerator
import asyncio
import numpy as np

logger = logging.getLogger(__name__)


class CartesiaEngine:
    """
    Cartesia TTS engine for low-latency speech synthesis.
    """
    
    def __init__(
        self,
        api_key: str,
        voice_id: str = "default",
        model: str = "sonic-english",
        sample_rate: int = 24000
    ):
        """
        Initializes Cartesia TTS engine.
        
        Parameters:
            api_key: Cartesia API key
            voice_id: Voice identifier
            model: TTS model name
            sample_rate: Output sample rate
        """
        self.api_key = api_key
        self.voice_id = voice_id
        self.model = model
        self.sample_rate = sample_rate
        
        self._client = None
        
        logger.info(f"CartesiaEngine initialized: model={model}, voice={voice_id}")
    
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
    
    async def synthesize(self, text: str) -> np.ndarray:
        """
        Synthesizes speech from text.
        
        Parameters:
            text: Text to synthesize
        
        Returns:
            Audio samples as numpy array
        """
        try:
            client = self._get_client()
            
            # Generate speech
            audio_data = await asyncio.to_thread(
                self._synthesize_sync,
                text
            )
            
            return audio_data
            
        except Exception as e:
            logger.error(f"Synthesis error: {e}")
            raise
    
    def _synthesize_sync(self, text: str) -> np.ndarray:
        """Synchronous synthesis."""
        client = self._get_client()
        
        response = client.tts.sse(
            model_id=self.model,
            transcript=text,
            voice={
                "mode": "id",
                "id": self.voice_id
            },
            output_format={
                "container": "raw",
                "encoding": "pcm_f32le",
                "sample_rate": self.sample_rate
            }
        )
        
        # Collect audio chunks
        audio_chunks = []
        for chunk in response:
            if hasattr(chunk, 'audio') and chunk.audio:
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
        Streams synthesized audio chunks.
        
        Parameters:
            text: Text to synthesize
        
        Yields:
            Audio chunks as numpy arrays
        """
        try:
            client = self._get_client()
            
            response = client.tts.sse(
                model_id=self.model,
                transcript=text,
                voice={
                    "mode": "id",
                    "id": self.voice_id
                },
                output_format={
                    "container": "raw",
                    "encoding": "pcm_f32le",
                    "sample_rate": self.sample_rate
                }
            )
            
            for chunk in response:
                if hasattr(chunk, 'audio') and chunk.audio:
                    audio_array = np.frombuffer(chunk.audio, dtype=np.float32)
                    yield audio_array
                    
        except Exception as e:
            logger.error(f"Streaming synthesis error: {e}")
            raise
    
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

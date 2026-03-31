"""
RealtimeSTT-backed STT handler adapter.
Wraps voice_MVP STTHandler to conform to Voice_SME_expert's interface.
"""
import sys
import asyncio
import logging
import threading
from pathlib import Path
from typing import Optional, Dict, Callable

# Import MVP STT handler
_MVP_PATH = Path(__file__).parent.parent.parent.parent / "voice_engine_MVP" / "src"
if str(_MVP_PATH) not in sys.path:
    sys.path.insert(0, str(_MVP_PATH))

try:
    from stt_handler import STTHandler as _MVPSTTHandler
except ImportError as e:
    logging.error(f"Failed to import MVP STTHandler: {e}")
    _MVPSTTHandler = None

logger = logging.getLogger(__name__)


class RealtimeSTTHandler:
    """
    Adapter wrapping voice_MVP STTHandler for use in Voice_SME_expert.
    Provides continuous listening, barge-in support, and per-speaker transcription.
    """

    def __init__(self, mode: str = "accurate", sample_rate: int = 16000):
        """
        Initialize the RealtimeSTT adapter.
        
        Args:
            mode: STT mode - 'fast', 'balanced', or 'accurate'
            sample_rate: Audio sample rate in Hz
        """
        if _MVPSTTHandler is None:
            raise ImportError("voice_MVP STTHandler not available")
        
        self._handler = _MVPSTTHandler(mode=mode)
        self.sample_rate = sample_rate
        self._is_streaming = False
        self._mode = mode
        
        logger.info(f"🎤 RealtimeSTTHandler initialized (mode={mode}, sample_rate={sample_rate})")

    async def start(self) -> None:
        """Start continuous listening."""
        try:
            await self._handler.start_listening()
            self._is_streaming = True
            logger.info("✅ RealtimeSTT: Continuous listening active")
        except Exception as e:
            logger.error(f"❌ Failed to start RealtimeSTT: {e}")
            raise

    async def stop(self) -> None:
        """Stop listening and release microphone."""
        try:
            await self._handler.stop_listening()
            self._is_streaming = False
            logger.info("🎤 RealtimeSTT: Stopped")
        except Exception as e:
            logger.error(f"❌ Error stopping RealtimeSTT: {e}")

    async def get_transcription(self, timeout: float = 30.0) -> str:
        """
        Block until a complete utterance is transcribed.
        
        Args:
            timeout: Maximum time to wait for transcription
            
        Returns:
            Transcribed text or empty string on timeout
        """
        try:
            # The MVP handler's get_transcription is already async
            text = await self._handler.get_transcription()
            return text or ""
        except asyncio.TimeoutError:
            logger.warning(f"⚠️ STT timeout after {timeout}s")
            return ""
        except Exception as e:
            logger.error(f"❌ Transcription error: {e}")
            return ""

    def get_realtime_text(self) -> str:
        """
        Non-blocking: returns partial transcription during speech.
        Used for barge-in detection.
        """
        return self._handler.get_realtime_text()

    def clear_realtime_text(self) -> None:
        """Clear the real-time transcription buffer."""
        self._handler.clear_realtime_text()

    def set_tts_active(self, active: bool) -> None:
        """
        Tell STT whether TTS is playing (suppresses echo).
        
        Args:
            active: True when TTS is speaking, False otherwise
        """
        self._handler.tts_is_active = active

    def set_tts_stop_callback(self, callback: Optional[Callable]) -> None:
        """
        Register callback called when barge-in is detected.
        
        Args:
            callback: Function to call when user interrupts TTS
        """
        self._handler.tts_stop_callback = callback

    def set_current_tts_text(self, text: str) -> None:
        """
        Set the text currently being spoken by TTS (for echo detection).
        
        Args:
            text: The text being spoken
        """
        if hasattr(self._handler, 'set_current_tts_text'):
            self._handler.set_current_tts_text(text)

    def clear_current_tts_text(self) -> None:
        """Clear the current TTS text."""
        if hasattr(self._handler, 'clear_current_tts_text'):
            self._handler.clear_current_tts_text()

    def flush_recorder(self) -> None:
        """Reset STT state after TTS playback."""
        if hasattr(self._handler, 'flush_recorder'):
            self._handler.flush_recorder()
        else:
            self.clear_realtime_text()

    def get_audio_rms(self) -> float:
        """Get current audio RMS level for voice activity detection."""
        if hasattr(self._handler, 'get_audio_rms'):
            return self._handler.get_audio_rms()
        return 0.0

    def get_performance_stats(self) -> Dict:
        """
        Get performance statistics.
        
        Returns:
            Dictionary with transcription stats
        """
        if hasattr(self._handler, 'get_performance_stats'):
            return self._handler.get_performance_stats()
        return {
            "model": self._mode,
            "transcription_count": 0,
            "avg_latency_ms": 0.0,
            "is_listening": self._is_streaming
        }

    @property
    def is_streaming(self) -> bool:
        """Check if STT is currently streaming."""
        return self._is_streaming

    @property
    def is_listening(self) -> bool:
        """Check if STT is currently listening (alias for is_streaming)."""
        return self._is_streaming

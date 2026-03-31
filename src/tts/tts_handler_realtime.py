"""
Cartesia-backed TTS handler adapter (wraps voice_MVP TTSHandler).
Provides streaming TTS with <150ms barge-in detection via STT monitoring.
"""
import sys
import logging
from pathlib import Path
from typing import Optional

# Import MVP TTS handler
_MVP_PATH = Path(__file__).parent.parent.parent.parent / "voice_engine_MVP" / "src"
if str(_MVP_PATH) not in sys.path:
    sys.path.insert(0, str(_MVP_PATH))

try:
    from tts_handler_optimized import TTSHandler as _MVPTTSHandler
except ImportError as e:
    logging.error(f"Failed to import MVP TTSHandler: {e}")
    _MVPTTSHandler = None

logger = logging.getLogger(__name__)


class MeetingTTSHandler:
    """
    TTS handler for Voice_SME_expert.
    Wraps voice_MVP TTSHandler — provides Cartesia streaming TTS
    with <150ms barge-in detection via STT monitoring.
    """

    def __init__(self, stt_handler, config):
        """
        Initialize the MeetingTTSHandler.
        
        Args:
            stt_handler: RealtimeSTTHandler instance (for barge-in detection)
            config: AppConfig instance with TTS settings
        """
        if _MVPTTSHandler is None:
            raise ImportError("voice_MVP TTSHandler not available")
        
        # Get the inner MVP STT handler for barge-in integration
        inner_stt = None
        if hasattr(stt_handler, '_handler'):
            inner_stt = stt_handler._handler
        else:
            inner_stt = stt_handler
        
        # Determine TTS engine from config
        use_cartesia = bool(getattr(config, 'cartesia_api_key', ''))
        use_kokoro = getattr(config, 'use_kokoro_tts', False)
        
        try:
            self._handler = _MVPTTSHandler(
                stt_handler=inner_stt,
                use_kokoro=use_kokoro,
                use_cartesia=use_cartesia
            )
            self._config = config
            self._last_text = ""
            
            engine_name = "Cartesia" if use_cartesia else ("Kokoro" if use_kokoro else "System")
            logger.info(f"✅ MeetingTTSHandler initialized ({engine_name} engine)")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize TTS handler: {e}")
            raise

    def speak(self, text: str, enable_barge_in: bool = True) -> str:
        """
        Start speaking. Non-blocking - returns immediately.
        
        Args:
            text: Text to synthesize and speak
            enable_barge_in: Whether to allow interruption
            
        Returns:
            The text being spoken
        """
        if not text or not text.strip():
            logger.warning("⚠️ Empty text passed to speak()")
            return ""
        
        self._last_text = text
        
        try:
            return self._handler.speak(text, enable_barge_in=enable_barge_in)
        except Exception as e:
            logger.error(f"❌ TTS speak error: {e}")
            return ""

    def wait_for_completion(self, timeout: float = 30.0) -> bool:
        """
        Block until TTS finishes or is interrupted.
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Returns:
            True if completed normally, False if interrupted or timed out
        """
        try:
            return self._handler.wait_for_completion(timeout=timeout)
        except Exception as e:
            logger.error(f"❌ TTS wait error: {e}")
            return False

    def stop_playback(self) -> None:
        """Stop current playback immediately."""
        try:
            self._handler.stop_playback()
        except Exception as e:
            logger.error(f"❌ TTS stop error: {e}")

    def was_barge_in(self) -> bool:
        """
        Check if barge-in was detected during last playback.
        
        Returns:
            True if user interrupted the TTS
        """
        try:
            if hasattr(self._handler, 'is_barge_in_detected'):
                return self._handler.is_barge_in_detected()
            elif hasattr(self._handler, 'barge_in_detected'):
                return self._handler.barge_in_detected
            return False
        except Exception:
            return False

    def shutdown(self) -> None:
        """Shutdown the TTS handler and release resources."""
        try:
            if hasattr(self._handler, 'shutdown'):
                self._handler.shutdown()
            logger.info("🔌 MeetingTTSHandler shutdown complete")
        except Exception as e:
            logger.error(f"❌ TTS shutdown error: {e}")

    @property
    def is_speaking(self) -> bool:
        """Check if TTS is currently playing."""
        try:
            if hasattr(self._handler, 'is_playing'):
                return self._handler.is_playing
            return False
        except Exception:
            return False

    @property
    def last_text(self) -> str:
        """Get the last text that was spoken."""
        return self._last_text

    def get_stats(self) -> dict:
        """Get TTS performance statistics."""
        try:
            if hasattr(self._handler, 'get_stats'):
                return self._handler.get_stats()
            return {}
        except Exception:
            return {}

"""Audio playback for TTS output."""
import logging
from typing import Optional, Callable
import threading
import queue
import numpy as np

logger = logging.getLogger(__name__)


class AudioPlayer:
    """
    Audio playback for TTS output with queue management.
    """
    
    def __init__(
        self,
        device_id: Optional[int] = None,
        sample_rate: int = 24000,
        buffer_size: int = 2048
    ):
        """
        Initializes audio player.
        
        Parameters:
            device_id: Output device ID
            sample_rate: Audio sample rate
            buffer_size: Playback buffer size
        """
        self.device_id = device_id
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        
        # Audio queue
        self._audio_queue: queue.Queue = queue.Queue()
        
        # Playback state
        self._is_playing = False
        self._is_paused = False
        self._playback_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        # Callback for playback completion
        self._on_complete: Optional[Callable] = None
        
        # Stream reference
        self._stream = None
        
        logger.info(f"AudioPlayer initialized: sample_rate={sample_rate}")
    
    def play(self, audio_data: np.ndarray) -> None:
        """
        Plays audio data immediately.
        
        Parameters:
            audio_data: Audio samples as numpy array
        """
        self._audio_queue.put(audio_data)
        
        if not self._is_playing:
            self._start_playback()
    
    def queue_audio(self, audio_data: np.ndarray) -> None:
        """
        Adds audio to playback queue.
        
        Parameters:
            audio_data: Audio samples
        """
        self._audio_queue.put(audio_data)
    
    def _start_playback(self) -> None:
        """Start playback thread."""
        if self._playback_thread and self._playback_thread.is_alive():
            return
        
        self._stop_event.clear()
        self._is_playing = True
        
        self._playback_thread = threading.Thread(
            target=self._playback_loop,
            daemon=True
        )
        self._playback_thread.start()
    
    def _playback_loop(self) -> None:
        """Main playback loop."""
        try:
            import sounddevice as sd
            
            while not self._stop_event.is_set():
                try:
                    # Get audio from queue
                    audio_data = self._audio_queue.get(timeout=0.1)
                    
                    if self._is_paused:
                        # Re-queue if paused
                        self._audio_queue.put(audio_data)
                        continue
                    
                    # Play audio
                    sd.play(audio_data, samplerate=self.sample_rate, device=self.device_id)
                    sd.wait()  # Wait for playback to finish
                    
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"Playback error: {e}")
            
        except ImportError:
            logger.error("sounddevice not installed")
        finally:
            self._is_playing = False
            
            if self._on_complete:
                self._on_complete()
    
    def stop(self) -> None:
        """Stops playback and clears queue."""
        self._stop_event.set()
        
        # Clear queue
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                break
        
        # Stop current playback
        try:
            import sounddevice as sd
            sd.stop()
        except:
            pass
        
        self._is_playing = False
        logger.debug("Playback stopped")
    
    def pause(self) -> None:
        """Pauses playback."""
        self._is_paused = True
        logger.debug("Playback paused")
    
    def resume(self) -> None:
        """Resumes playback."""
        self._is_paused = False
        
        if not self._is_playing and not self._audio_queue.empty():
            self._start_playback()
        
        logger.debug("Playback resumed")
    
    def is_playing(self) -> bool:
        """Check if currently playing."""
        return self._is_playing and not self._is_paused
    
    def get_queue_size(self) -> int:
        """Get number of items in queue."""
        return self._audio_queue.qsize()
    
    def set_on_complete(self, callback: Callable) -> None:
        """Set playback completion callback."""
        self._on_complete = callback
    
    def set_device(self, device_id: int) -> None:
        """Change output device."""
        self.device_id = device_id
        logger.info(f"Output device changed to: {device_id}")
    
    @staticmethod
    def list_devices() -> list:
        """List available output devices."""
        try:
            import sounddevice as sd
            devices = sd.query_devices()
            return [
                {'id': i, 'name': d['name'], 'channels': d['max_output_channels']}
                for i, d in enumerate(devices)
                if d['max_output_channels'] > 0
            ]
        except ImportError:
            return []

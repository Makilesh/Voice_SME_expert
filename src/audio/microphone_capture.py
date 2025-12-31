"""Captures audio from physical microphone for in-person meetings."""
import asyncio
import logging
from typing import AsyncGenerator, Callable, List, Dict, Optional
import numpy as np
import sounddevice as sd

from .audio_capture import AudioCaptureBase

logger = logging.getLogger(__name__)


class MicrophoneCapture(AudioCaptureBase):
    """
    Captures audio from physical microphone.
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_size: int = 1024,
        channels: int = 1,
        device_index: Optional[int] = None
    ):
        """
        Initialize microphone capture.
        
        Parameters:
            sample_rate: Audio sample rate in Hz
            chunk_size: Number of frames per buffer
            channels: Number of audio channels
            device_index: Microphone device index (None for default)
        """
        super().__init__(sample_rate, chunk_size, channels)
        self.device_index = device_index
        self._stream: Optional[sd.InputStream] = None
        self._audio_queue: Optional[asyncio.Queue] = None
        self._noise_reduction_enabled = False
        self._noise_reduction_strength = 1
        
        logger.info(f"MicrophoneCapture initialized: {sample_rate}Hz, chunk={chunk_size}")
    
    def start_capture(self, callback: Callable[[np.ndarray], None]) -> None:
        """
        Starts audio capture and calls callback with each chunk.
        
        Parameters:
            callback: Function to call with each audio chunk
        """
        if self._is_capturing:
            logger.warning("Already capturing audio")
            return
        
        self._callback = callback
        
        def audio_callback(indata, frames, time, status):
            if status:
                logger.warning(f"Audio input status: {status}")
            
            # Convert to numpy array and normalize
            audio_data = indata.copy().flatten()
            
            if self._callback:
                self._callback(audio_data)
        
        try:
            self._stream = sd.InputStream(
                device=self.device_index,
                channels=self.channels,
                samplerate=self.sample_rate,
                blocksize=self.chunk_size,
                callback=audio_callback,
                dtype=np.float32
            )
            
            self._stream.start()
            self._is_capturing = True
            logger.info(f"Started microphone capture on device {self.device_index}")
            
        except Exception as e:
            logger.error(f"Failed to start microphone capture: {e}")
            raise
    
    def stop_capture(self) -> None:
        """
        Stops audio capture and releases resources.
        """
        if not self._is_capturing:
            return
        
        try:
            if self._stream:
                self._stream.stop()
                self._stream.close()
                self._stream = None
            
            self._is_capturing = False
            self._callback = None
            logger.info("Stopped microphone capture")
            
        except Exception as e:
            logger.error(f"Error stopping microphone capture: {e}")
    
    async def get_audio_stream(self) -> AsyncGenerator[np.ndarray, None]:
        """
        Async generator for streaming audio data.
        
        Yields:
            numpy array: Audio chunks
        """
        self._audio_queue = asyncio.Queue(maxsize=100)
        
        def queue_callback(audio_chunk: np.ndarray):
            try:
                self._audio_queue.put_nowait(audio_chunk)
            except asyncio.QueueFull:
                logger.warning("Audio queue full, dropping frame")
        
        self.start_capture(queue_callback)
        
        try:
            while self._is_capturing:
                try:
                    audio_chunk = await asyncio.wait_for(
                        self._audio_queue.get(),
                        timeout=1.0
                    )
                    yield audio_chunk
                except asyncio.TimeoutError:
                    continue
        finally:
            self.stop_capture()
    
    def set_device(self, device_id: Optional[int] = None) -> bool:
        """
        Sets the audio input device.
        
        Parameters:
            device_id: Device ID or index
        
        Returns:
            bool: True if successful
        """
        was_capturing = self._is_capturing
        
        if was_capturing:
            self.stop_capture()
        
        self.device_index = device_id
        
        # Test if device is valid
        try:
            test_stream = sd.InputStream(
                device=self.device_index,
                channels=self.channels,
                samplerate=self.sample_rate,
                blocksize=self.chunk_size
            )
            test_stream.close()
            logger.info(f"Set audio device to {device_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to set device {device_id}: {e}")
            return False
    
    @staticmethod
    def list_devices() -> List[Dict]:
        """
        Returns available microphone devices.
        
        Returns:
            List of dicts with device info
        """
        devices = []
        
        try:
            device_list = sd.query_devices()
            for idx, device in enumerate(device_list):
                if device['max_input_channels'] > 0:
                    devices.append({
                        'index': idx,
                        'name': device['name'],
                        'channels': device['max_input_channels'],
                        'sample_rate': device['default_samplerate'],
                        'hostapi': sd.query_hostapis(device['hostapi'])['name']
                    })
        except Exception as e:
            logger.error(f"Error listing devices: {e}")
        
        return devices
    
    def set_noise_reduction(self, enabled: bool = True, aggressiveness: int = 1) -> None:
        """
        Enables/configures noise reduction.
        
        Parameters:
            enabled: Enable noise reduction
            aggressiveness: Strength level (0-3)
        """
        if not 0 <= aggressiveness <= 3:
            raise ValueError("aggressiveness must be 0-3")
        
        self._noise_reduction_enabled = enabled
        self._noise_reduction_strength = aggressiveness
        logger.info(f"Noise reduction: {enabled}, strength: {aggressiveness}")
    
    def get_device_info(self) -> Optional[Dict]:
        """
        Get information about current device.
        
        Returns:
            Dict with device information or None
        """
        try:
            if self.device_index is not None:
                device = sd.query_devices(self.device_index)
            else:
                device = sd.query_devices(kind='input')
            
            return {
                'name': device['name'],
                'channels': device['max_input_channels'],
                'sample_rate': device['default_samplerate'],
                'hostapi': sd.query_hostapis(device['hostapi'])['name']
            }
        except Exception as e:
            logger.error(f"Error getting device info: {e}")
            return None

"""Captures system audio from virtual meetings (Zoom, Meet, Teams)."""
import asyncio
import logging
from typing import AsyncGenerator, Callable, Optional, List
import numpy as np
import sounddevice as sd
import platform

from .audio_capture import AudioCaptureBase

logger = logging.getLogger(__name__)


class VirtualAudioCapture(AudioCaptureBase):
    """
    Captures system audio from virtual meetings using loopback.
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_size: int = 1024,
        channels: int = 1,
        loopback_device: Optional[str] = None
    ):
        """
        Initialize virtual audio capture using system loopback.
        
        Parameters:
            sample_rate: Audio sample rate in Hz
            chunk_size: Number of frames per buffer
            channels: Number of audio channels
            loopback_device: Name or index of loopback device
        """
        super().__init__(sample_rate, chunk_size, channels)
        self.loopback_device = loopback_device
        self._stream: Optional[sd.InputStream] = None
        self._audio_queue: Optional[asyncio.Queue] = None
        
        logger.info(f"VirtualAudioCapture initialized: {sample_rate}Hz")
    
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
        
        # Find loopback device if not specified
        if self.loopback_device is None:
            detected_device = self.detect_meeting_audio()
            if detected_device:
                self.loopback_device = detected_device
                logger.info(f"Auto-detected loopback device: {detected_device}")
            else:
                logger.warning("No loopback device detected, using default")
        
        def audio_callback(indata, frames, time, status):
            if status:
                logger.warning(f"Audio input status: {status}")
            
            audio_data = indata.copy().flatten()
            
            if self._callback:
                self._callback(audio_data)
        
        try:
            device_index = self._find_device_index(self.loopback_device)
            
            self._stream = sd.InputStream(
                device=device_index,
                channels=self.channels,
                samplerate=self.sample_rate,
                blocksize=self.chunk_size,
                callback=audio_callback,
                dtype=np.float32
            )
            
            self._stream.start()
            self._is_capturing = True
            logger.info(f"Started virtual audio capture on device {self.loopback_device}")
            
        except Exception as e:
            logger.error(f"Failed to start virtual audio capture: {e}")
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
            logger.info("Stopped virtual audio capture")
            
        except Exception as e:
            logger.error(f"Error stopping virtual audio capture: {e}")
    
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
            device_id: Device ID or name
        
        Returns:
            bool: True if successful
        """
        was_capturing = self._is_capturing
        
        if was_capturing:
            self.stop_capture()
        
        self.loopback_device = device_id
        return True
    
    def detect_meeting_audio(self) -> Optional[str]:
        """
        Auto-detects audio output from meeting applications.
        
        Returns:
            Device name or None
        """
        try:
            devices = sd.query_devices()
            system = platform.system()
            
            # Look for loopback or stereo mix devices
            loopback_keywords = [
                'loopback', 'stereo mix', 'wave out', 'what u hear',
                'zoom', 'meet', 'teams', 'virtual'
            ]
            
            for idx, device in enumerate(devices):
                if device['max_input_channels'] > 0:
                    device_name = device['name'].lower()
                    
                    for keyword in loopback_keywords:
                        if keyword in device_name:
                            logger.info(f"Found potential loopback device: {device['name']}")
                            return idx
            
            # Platform-specific defaults
            if system == 'Windows':
                # Look for Stereo Mix or VB-Audio Virtual Cable
                for idx, device in enumerate(devices):
                    name = device['name'].lower()
                    if 'stereo mix' in name or 'cable' in name:
                        return idx
            elif system == 'Darwin':  # macOS
                # Look for BlackHole or Soundflower
                for idx, device in enumerate(devices):
                    name = device['name'].lower()
                    if 'blackhole' in name or 'soundflower' in name:
                        return idx
            
            logger.warning("No loopback device detected")
            return None
            
        except Exception as e:
            logger.error(f"Error detecting meeting audio: {e}")
            return None
    
    async def capture_application_audio(
        self,
        application_name: str
    ) -> AsyncGenerator[np.ndarray, None]:
        """
        Captures audio specifically from named application.
        
        Parameters:
            application_name: Name of application (zoom, meet, teams)
        
        Yields:
            numpy array: Audio chunks from application
        """
        logger.info(f"Capturing audio from application: {application_name}")
        
        # This would require platform-specific implementation
        # For now, use general loopback
        async for chunk in self.get_audio_stream():
            yield chunk
    
    def _find_device_index(self, device: Optional[str]) -> Optional[int]:
        """
        Find device index from name or return as-is if integer.
        
        Parameters:
            device: Device name or index
        
        Returns:
            Device index or None
        """
        if device is None:
            return None
        
        if isinstance(device, int):
            return device
        
        # Search by name
        try:
            devices = sd.query_devices()
            device_lower = str(device).lower()
            
            for idx, dev in enumerate(devices):
                if device_lower in dev['name'].lower():
                    return idx
            
            logger.warning(f"Device '{device}' not found, using default")
            return None
            
        except Exception as e:
            logger.error(f"Error finding device: {e}")
            return None
    
    @staticmethod
    def list_loopback_devices() -> List[dict]:
        """
        List available loopback/virtual audio devices.
        
        Returns:
            List of loopback device information
        """
        devices = []
        
        try:
            device_list = sd.query_devices()
            loopback_keywords = [
                'loopback', 'stereo mix', 'wave out', 'cable',
                'virtual', 'blackhole', 'soundflower'
            ]
            
            for idx, device in enumerate(device_list):
                if device['max_input_channels'] > 0:
                    device_name = device['name'].lower()
                    
                    # Check if it's a loopback device
                    is_loopback = any(kw in device_name for kw in loopback_keywords)
                    
                    if is_loopback:
                        devices.append({
                            'index': idx,
                            'name': device['name'],
                            'channels': device['max_input_channels'],
                            'sample_rate': device['default_samplerate']
                        })
        
        except Exception as e:
            logger.error(f"Error listing loopback devices: {e}")
        
        return devices

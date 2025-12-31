"""Abstract base class for audio capture."""
from abc import ABC, abstractmethod
from typing import AsyncGenerator, Callable, Optional
import numpy as np


class AudioCaptureBase(ABC):
    """
    Abstract base class for audio capture with common interface.
    """
    
    def __init__(self, sample_rate: int = 16000, chunk_size: int = 1024, channels: int = 1):
        """
        Initialize audio capture base.
        
        Parameters:
            sample_rate: Audio sample rate in Hz
            chunk_size: Number of frames per buffer
            channels: Number of audio channels (1 for mono, 2 for stereo)
        """
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.channels = channels
        self._is_capturing = False
        self._callback: Optional[Callable] = None
    
    @abstractmethod
    def start_capture(self, callback: Callable[[np.ndarray], None]) -> None:
        """
        Starts audio capture and calls callback with each chunk.
        
        Parameters:
            callback: Function to call with each audio chunk (numpy array)
        """
        pass
    
    @abstractmethod
    def stop_capture(self) -> None:
        """
        Stops audio capture and releases resources.
        """
        pass
    
    @abstractmethod
    async def get_audio_stream(self) -> AsyncGenerator[np.ndarray, None]:
        """
        Async generator for streaming audio data.
        
        Yields:
            numpy array: Audio chunks as numpy arrays
        """
        pass
    
    @abstractmethod
    def set_device(self, device_id: Optional[int] = None) -> bool:
        """
        Sets the audio input device.
        
        Parameters:
            device_id: Device ID or index, None for default
        
        Returns:
            bool: True if successful
        """
        pass
    
    def is_capturing(self) -> bool:
        """
        Check if currently capturing audio.
        
        Returns:
            bool: True if capturing
        """
        return self._is_capturing
    
    def get_config(self) -> dict:
        """
        Get current audio configuration.
        
        Returns:
            dict: Configuration parameters
        """
        return {
            "sample_rate": self.sample_rate,
            "chunk_size": self.chunk_size,
            "channels": self.channels,
            "is_capturing": self._is_capturing
        }

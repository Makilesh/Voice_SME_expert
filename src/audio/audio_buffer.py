"""Thread-safe circular buffer for audio streaming."""
import logging
import threading
from collections import deque
from typing import Optional
import numpy as np
import time

logger = logging.getLogger(__name__)


class AudioBuffer:
    """
    Thread-safe circular buffer for audio data.
    """
    
    def __init__(self, max_duration_seconds: float = 30.0, sample_rate: int = 16000):
        """
        Creates circular buffer with specified capacity.
        
        Parameters:
            max_duration_seconds: Maximum buffer duration in seconds
            sample_rate: Audio sample rate in Hz
        """
        self.max_duration = max_duration_seconds
        self.sample_rate = sample_rate
        self.max_samples = int(max_duration_seconds * sample_rate)
        
        self._buffer = deque(maxlen=self.max_samples)
        self._timestamps = deque(maxlen=self.max_samples)
        self._lock = threading.RLock()
        self._total_samples_written = 0
        
        logger.info(f"AudioBuffer initialized: {max_duration_seconds}s capacity @ {sample_rate}Hz")
    
    def write(self, audio_chunk: np.ndarray, timestamp: Optional[float] = None) -> bool:
        """
        Writes audio chunk to buffer.
        
        Parameters:
            audio_chunk: Audio data as numpy array
            timestamp: Optional timestamp (uses current time if None)
        
        Returns:
            bool: True if successful
        """
        if timestamp is None:
            timestamp = time.time()
        
        try:
            with self._lock:
                # Add each sample with timestamp
                for sample in audio_chunk:
                    self._buffer.append(sample)
                    self._timestamps.append(timestamp)
                
                self._total_samples_written += len(audio_chunk)
                
                return True
                
        except Exception as e:
            logger.error(f"Error writing to buffer: {e}")
            return False
    
    def read(self, duration_seconds: float) -> np.ndarray:
        """
        Reads specified duration from buffer (removes data).
        
        Parameters:
            duration_seconds: Duration to read in seconds
        
        Returns:
            numpy array: Audio data
        """
        num_samples = int(duration_seconds * self.sample_rate)
        
        with self._lock:
            samples_to_read = min(num_samples, len(self._buffer))
            
            if samples_to_read == 0:
                return np.array([])
            
            # Extract samples
            audio_data = np.array([self._buffer.popleft() for _ in range(samples_to_read)])
            
            # Remove corresponding timestamps
            for _ in range(samples_to_read):
                self._timestamps.popleft()
            
            return audio_data
    
    def get_latest(self, duration_seconds: float) -> np.ndarray:
        """
        Gets most recent audio without removing from buffer.
        
        Parameters:
            duration_seconds: Duration to get in seconds
        
        Returns:
            numpy array: Most recent audio
        """
        num_samples = int(duration_seconds * self.sample_rate)
        
        with self._lock:
            samples_to_get = min(num_samples, len(self._buffer))
            
            if samples_to_get == 0:
                return np.array([])
            
            # Get last N samples without removing
            buffer_list = list(self._buffer)
            audio_data = np.array(buffer_list[-samples_to_get:])
            
            return audio_data
    
    def get_all(self) -> np.ndarray:
        """
        Gets all audio in buffer without removing.
        
        Returns:
            numpy array: All buffered audio
        """
        with self._lock:
            if len(self._buffer) == 0:
                return np.array([])
            
            return np.array(list(self._buffer))
    
    def clear(self) -> None:
        """
        Clears all data from buffer.
        """
        with self._lock:
            self._buffer.clear()
            self._timestamps.clear()
            logger.debug("Buffer cleared")
    
    def get_duration(self) -> float:
        """
        Get current duration of audio in buffer.
        
        Returns:
            float: Duration in seconds
        """
        with self._lock:
            return len(self._buffer) / self.sample_rate
    
    def get_size(self) -> int:
        """
        Get number of samples in buffer.
        
        Returns:
            int: Number of samples
        """
        with self._lock:
            return len(self._buffer)
    
    def is_full(self) -> bool:
        """
        Check if buffer is full.
        
        Returns:
            bool: True if buffer is at capacity
        """
        with self._lock:
            return len(self._buffer) >= self.max_samples
    
    def is_empty(self) -> bool:
        """
        Check if buffer is empty.
        
        Returns:
            bool: True if buffer is empty
        """
        with self._lock:
            return len(self._buffer) == 0
    
    def get_stats(self) -> dict:
        """
        Get buffer statistics.
        
        Returns:
            dict: Buffer statistics
        """
        with self._lock:
            current_duration = len(self._buffer) / self.sample_rate
            fill_percentage = (len(self._buffer) / self.max_samples) * 100
            
            return {
                'current_samples': len(self._buffer),
                'max_samples': self.max_samples,
                'current_duration_sec': current_duration,
                'max_duration_sec': self.max_duration,
                'fill_percentage': fill_percentage,
                'total_samples_written': self._total_samples_written,
                'sample_rate': self.sample_rate
            }
    
    def get_time_range(self) -> tuple:
        """
        Get timestamp range of buffered audio.
        
        Returns:
            tuple: (earliest_timestamp, latest_timestamp) or (None, None) if empty
        """
        with self._lock:
            if len(self._timestamps) == 0:
                return (None, None)
            
            timestamps_list = list(self._timestamps)
            return (timestamps_list[0], timestamps_list[-1])

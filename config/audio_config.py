"""Audio-specific configuration."""
from dataclasses import dataclass
from typing import Optional


@dataclass
class AudioCaptureConfig:
    """Configuration for audio input devices."""
    
    sample_rate: int = 16000
    channels: int = 1
    chunk_size: int = 1024
    buffer_duration: float = 5.0
    device_index: int = -1  # -1 for default device
    
    def __post_init__(self):
        """Validate audio capture configuration."""
        if self.sample_rate not in [8000, 16000, 22050, 44100, 48000]:
            raise ValueError(f"Unsupported sample_rate: {self.sample_rate}")
        
        if self.channels not in [1, 2]:
            raise ValueError(f"Only mono (1) or stereo (2) channels supported")
        
        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive")
        
        if self.buffer_duration <= 0:
            raise ValueError(f"buffer_duration must be positive")


@dataclass
class AudioOutputConfig:
    """Configuration for audio output/playback."""
    
    sample_rate: int = 24000
    channels: int = 1
    output_device: Optional[str] = None
    buffer_size: int = 2048
    
    def __post_init__(self):
        """Validate audio output configuration."""
        if self.sample_rate not in [16000, 22050, 24000, 44100, 48000]:
            raise ValueError(f"Unsupported output sample_rate: {self.sample_rate}")
        
        if self.channels not in [1, 2]:
            raise ValueError(f"Only mono (1) or stereo (2) channels supported")
        
        if self.buffer_size <= 0:
            raise ValueError(f"buffer_size must be positive")


@dataclass
class AudioPreprocessingConfig:
    """Configuration for audio preprocessing."""
    
    enable_noise_reduction: bool = True
    noise_reduction_strength: int = 1  # 0-3
    enable_vad: bool = True
    vad_aggressiveness: int = 2  # 0-3
    target_db: float = -20.0
    normalize_audio: bool = True
    
    def __post_init__(self):
        """Validate preprocessing configuration."""
        if not 0 <= self.noise_reduction_strength <= 3:
            raise ValueError(f"noise_reduction_strength must be 0-3")
        
        if not 0 <= self.vad_aggressiveness <= 3:
            raise ValueError(f"vad_aggressiveness must be 0-3")

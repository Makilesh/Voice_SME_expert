"""Wake word detection configuration."""
from dataclasses import dataclass
from typing import List
from pathlib import Path


@dataclass
class WakeWordConfig:
    """Configuration for wake word detection system."""
    
    wake_phrases: List[str]
    sensitivity: float = 0.5
    model_path: str = "./models/wake_word/"
    detection_mode: str = "openwakeword"  # openwakeword, porcupine
    buffer_duration: float = 1.0
    min_confidence: float = 0.5
    
    def __post_init__(self):
        """Validate wake word configuration."""
        if not self.wake_phrases:
            raise ValueError("At least one wake phrase required")
        
        if not 0 <= self.sensitivity <= 1:
            raise ValueError("sensitivity must be between 0 and 1")
        
        if not 0 <= self.min_confidence <= 1:
            raise ValueError("min_confidence must be between 0 and 1")
        
        if self.detection_mode not in ["openwakeword", "porcupine"]:
            raise ValueError(f"Unsupported detection_mode: {self.detection_mode}")
        
        if self.buffer_duration <= 0:
            raise ValueError("buffer_duration must be positive")
        
        # Create model directory if it doesn't exist
        Path(self.model_path).mkdir(parents=True, exist_ok=True)
    
    def get_model_files(self) -> List[Path]:
        """
        Get list of wake word model files.
        
        Returns:
            List of Path objects for model files
        """
        model_dir = Path(self.model_path)
        if not model_dir.exists():
            return []
        
        # Look for .onnx files (OpenWakeWord) or .ppn files (Porcupine)
        model_files = []
        model_files.extend(model_dir.glob("*.onnx"))
        model_files.extend(model_dir.glob("*.ppn"))
        
        return model_files

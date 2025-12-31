"""Low-level keyword spotting implementation."""
import logging
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class KeywordMatch:
    """Keyword detection result."""
    keyword: str
    confidence: float
    timestamp: float


class KeywordSpotter:
    """
    Low-level keyword spotting implementation.
    """
    
    def __init__(
        self,
        keywords: List[str],
        model_type: str = "openwakeword",
        sample_rate: int = 16000
    ):
        """
        Initializes keyword spotter with target phrases.
        
        Parameters:
            keywords: List of keywords to detect
            model_type: Detection model type
            sample_rate: Audio sample rate
        """
        self.keywords = [kw.lower() for kw in keywords]
        self.model_type = model_type
        self.sample_rate = sample_rate
        
        self._model = None
        self._initialized = False
        self._frame_buffer = []
        self._frame_size = int(sample_rate * 0.08)  # 80ms frames
        
        logger.info(f"KeywordSpotter initialized: {keywords}")
    
    def _load_model(self) -> None:
        """Load the keyword spotting model lazily."""
        if self._initialized:
            return
        
        try:
            if self.model_type == "openwakeword":
                import openwakeword
                from openwakeword.model import Model
                
                # Load pre-trained models
                openwakeword.utils.download_models()
                
                self._model = Model(
                    wakeword_models=["hey_jarvis_v0.1"],  # Use available model
                    inference_framework="onnx"
                )
                logger.info("Loaded OpenWakeWord model")
            
            self._initialized = True
            
        except ImportError as e:
            logger.warning(f"Wake word library not available: {e}")
            self._initialized = True
        except Exception as e:
            logger.error(f"Error loading keyword spotter: {e}")
            self._initialized = True
    
    def process_frame(
        self,
        audio_frame: np.ndarray,
        timestamp: float = 0.0
    ) -> List[Dict]:
        """
        Processes single audio frame for keywords.
        
        Parameters:
            audio_frame: Audio data
            timestamp: Current timestamp
        
        Returns:
            list: Detected keywords with confidence
        """
        self._load_model()
        
        detections = []
        
        # Add to frame buffer
        self._frame_buffer.extend(audio_frame.tolist())
        
        # Process when we have enough frames
        while len(self._frame_buffer) >= self._frame_size:
            frame = np.array(self._frame_buffer[:self._frame_size])
            self._frame_buffer = self._frame_buffer[self._frame_size:]
            
            try:
                if self._model is not None:
                    # Process with model
                    prediction = self._model.predict(frame)
                    
                    # Check each model's predictions
                    for model_name, scores in prediction.items():
                        if isinstance(scores, (list, np.ndarray)):
                            score = max(scores) if len(scores) > 0 else 0
                        else:
                            score = scores
                        
                        if score > 0.5:  # Detection threshold
                            # Map to our keywords
                            for keyword in self.keywords:
                                detections.append({
                                    'keyword': keyword,
                                    'confidence': float(score),
                                    'timestamp': timestamp,
                                    'model': model_name
                                })
                else:
                    # Fallback: simple energy-based detection (for testing)
                    pass
                    
            except Exception as e:
                logger.error(f"Frame processing error: {e}")
        
        return detections
    
    def add_keyword(self, keyword: str) -> bool:
        """
        Adds new keyword to detection list.
        
        Parameters:
            keyword: Keyword to add
        
        Returns:
            bool: Success
        """
        keyword_lower = keyword.lower()
        
        if keyword_lower not in self.keywords:
            self.keywords.append(keyword_lower)
            logger.info(f"Added keyword: {keyword}")
            return True
        
        return False
    
    def remove_keyword(self, keyword: str) -> bool:
        """
        Removes keyword from detection list.
        
        Parameters:
            keyword: Keyword to remove
        
        Returns:
            bool: Success
        """
        keyword_lower = keyword.lower()
        
        if keyword_lower in self.keywords:
            self.keywords.remove(keyword_lower)
            logger.info(f"Removed keyword: {keyword}")
            return True
        
        return False
    
    def get_keywords(self) -> List[str]:
        """Get current keyword list."""
        return self.keywords.copy()
    
    def reset(self) -> None:
        """Reset spotter state."""
        self._frame_buffer = []

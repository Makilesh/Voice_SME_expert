"""Error recovery configuration (from voice_engine_MVP)."""
from dataclasses import dataclass
from typing import Optional


@dataclass
class ErrorRecoveryConfig:
    """
    Error handling and recovery configuration.
    Based on voice_engine_MVP patterns for graceful degradation.
    """
    
    # Error thresholds
    max_consecutive_errors: int = 3
    error_cooldown_seconds: float = 60.0
    
    # Retry configuration
    retry_base_delay: float = 1.0  # Initial delay in seconds
    retry_max_delay: float = 10.0  # Maximum delay
    retry_jitter: float = 0.1  # Random jitter factor
    
    # Circuit breaker
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: float = 30.0
    
    # Fallback behavior
    enable_graceful_degradation: bool = True
    fallback_response: str = "I'm having a bit of trouble right now. Could you try again?"
    
    def calculate_retry_delay(self, attempt: int) -> float:
        """
        Calculate retry delay with exponential backoff.
        
        Parameters:
            attempt: Current attempt number (0-indexed)
        
        Returns:
            Delay in seconds
        """
        import random
        
        delay = self.retry_base_delay * (2 ** attempt)
        delay = min(delay, self.retry_max_delay)
        
        # Add jitter
        jitter = delay * self.retry_jitter * random.random()
        return delay + jitter
    
    def __post_init__(self):
        """Validate configuration."""
        if self.max_consecutive_errors < 1:
            raise ValueError("max_consecutive_errors must be at least 1")
        
        if self.retry_base_delay < 0:
            raise ValueError("retry_base_delay must be non-negative")
        
        if self.retry_max_delay < self.retry_base_delay:
            raise ValueError("retry_max_delay must be >= retry_base_delay")
        
        if not 0 <= self.retry_jitter <= 1:
            raise ValueError("retry_jitter must be between 0 and 1")

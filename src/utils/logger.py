"""Centralized logging configuration."""
import logging
import sys
from pathlib import Path
from typing import Optional
import colorlog
from datetime import datetime


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    enable_color: bool = True
) -> logging.Logger:
    """
    Configures application logging.
    
    Parameters:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
        enable_color: Enable colored console output
    
    Returns:
        logging.Logger: Configured root logger
    """
    # Create logs directory if needed
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Set logging level
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    # Create formatters
    if enable_color:
        console_formatter = colorlog.ColoredFormatter(
            '%(log_color)s%(asctime)s [%(levelname)-8s]%(reset)s %(name)s - %(message)s',
            datefmt='%H:%M:%S',
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white',
            }
        )
    else:
        console_formatter = logging.Formatter(
            '%(asctime)s [%(levelname)-8s] %(name)s - %(message)s',
            datefmt='%H:%M:%S'
        )
    
    file_formatter = logging.Formatter(
        '%(asctime)s [%(levelname)-8s] %(name)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Remove existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
    
    return root_logger


class PerformanceLogger:
    """Logger for performance metrics."""
    
    def __init__(self, logger_name: str = "performance"):
        """
        Initialize performance logger.
        
        Parameters:
            logger_name: Name for the logger
        """
        self.logger = logging.getLogger(logger_name)
        self._metrics = {}
    
    def log_latency(self, component: str, latency_ms: float) -> None:
        """
        Logs component latency.
        
        Parameters:
            component: Component name
            latency_ms: Latency in milliseconds
        """
        if component not in self._metrics:
            self._metrics[component] = []
        
        self._metrics[component].append(latency_ms)
        self.logger.debug(f"{component}: {latency_ms:.2f}ms")
    
    def get_average_latency(self, component: str) -> Optional[float]:
        """
        Get average latency for component.
        
        Parameters:
            component: Component name
        
        Returns:
            Average latency in ms or None
        """
        if component not in self._metrics or not self._metrics[component]:
            return None
        
        return sum(self._metrics[component]) / len(self._metrics[component])
    
    def get_summary(self) -> dict:
        """
        Get performance summary.
        
        Returns:
            dict: Summary of all metrics
        """
        summary = {}
        
        for component, latencies in self._metrics.items():
            if latencies:
                summary[component] = {
                    'avg_ms': sum(latencies) / len(latencies),
                    'min_ms': min(latencies),
                    'max_ms': max(latencies),
                    'count': len(latencies)
                }
        
        return summary
    
    def reset(self) -> None:
        """Reset all metrics."""
        self._metrics.clear()

"""Monitors and manages memory usage."""
import logging
import psutil
import gc
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class MemoryMonitor:
    """
    Monitors system and process memory usage.
    """
    
    def __init__(self, warning_threshold_mb: float = 1000.0):
        """
        Initialize memory monitor.
        
        Parameters:
            warning_threshold_mb: Memory usage threshold for warnings in MB
        """
        self.warning_threshold = warning_threshold_mb * 1024 * 1024  # Convert to bytes
        self.process = psutil.Process()
        logger.info(f"MemoryMonitor initialized: warning threshold={warning_threshold_mb}MB")
    
    def check_memory(self) -> Dict:
        """
        Checks current memory usage.
        
        Returns:
            dict: Memory usage statistics
        """
        try:
            # Process memory
            process_info = self.process.memory_info()
            process_mb = process_info.rss / (1024 * 1024)
            
            # System memory
            system_mem = psutil.virtual_memory()
            
            stats = {
                'process_mb': process_mb,
                'process_percent': self.process.memory_percent(),
                'system_total_mb': system_mem.total / (1024 * 1024),
                'system_available_mb': system_mem.available / (1024 * 1024),
                'system_used_percent': system_mem.percent
            }
            
            # Check threshold
            if process_info.rss > self.warning_threshold:
                logger.warning(f"Memory usage high: {process_mb:.1f}MB")
            
            return stats
            
        except Exception as e:
            logger.error(f"Error checking memory: {e}")
            return {}
    
    def trigger_cleanup(self) -> None:
        """
        Triggers garbage collection if needed.
        """
        try:
            mem_before = self.process.memory_info().rss / (1024 * 1024)
            
            # Force garbage collection
            collected = gc.collect()
            
            mem_after = self.process.memory_info().rss / (1024 * 1024)
            freed = mem_before - mem_after
            
            logger.info(f"GC: collected {collected} objects, freed {freed:.1f}MB")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    def get_memory_info(self) -> str:
        """
        Get formatted memory information.
        
        Returns:
            str: Formatted memory info
        """
        stats = self.check_memory()
        
        if not stats:
            return "Memory info unavailable"
        
        return (
            f"Process: {stats['process_mb']:.1f}MB ({stats['process_percent']:.1f}%) | "
            f"System: {stats['system_used_percent']:.1f}% used, "
            f"{stats['system_available_mb']:.1f}MB available"
        )

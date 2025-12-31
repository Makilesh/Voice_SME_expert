"""Tracks performance metrics."""
import logging
import time
from typing import Dict, List, Optional
from collections import defaultdict
from datetime import datetime

logger = logging.getLogger(__name__)


class PerformanceTracker:
    """
    Tracks performance metrics for components.
    """
    
    def __init__(self):
        """Initialize performance tracker."""
        self._metrics: Dict[str, List[float]] = defaultdict(list)
        self._timestamps: Dict[str, List[datetime]] = defaultdict(list)
        self._start_times: Dict[str, float] = {}
        logger.info("PerformanceTracker initialized")
    
    def start_timer(self, name: str) -> None:
        """
        Start timing an operation.
        
        Parameters:
            name: Operation name
        """
        self._start_times[name] = time.perf_counter()
    
    def stop_timer(self, name: str) -> Optional[float]:
        """
        Stop timing and record metric.
        
        Parameters:
            name: Operation name
        
        Returns:
            float: Elapsed time in seconds or None if not started
        """
        if name not in self._start_times:
            logger.warning(f"Timer '{name}' was not started")
            return None
        
        elapsed = time.perf_counter() - self._start_times[name]
        self.record_metric(name, elapsed, "seconds")
        del self._start_times[name]
        
        return elapsed
    
    def record_metric(self, name: str, value: float, unit: str = "") -> None:
        """
        Records performance metric.
        
        Parameters:
            name: Metric name
            value: Metric value
            unit: Unit of measurement
        """
        self._metrics[name].append(value)
        self._timestamps[name].append(datetime.now())
        
        logger.debug(f"Metric '{name}': {value:.4f} {unit}")
    
    def get_summary(self) -> Dict:
        """
        Gets performance summary.
        
        Returns:
            dict: Metric summaries
        """
        summary = {}
        
        for name, values in self._metrics.items():
            if values:
                summary[name] = {
                    'count': len(values),
                    'mean': sum(values) / len(values),
                    'min': min(values),
                    'max': max(values),
                    'total': sum(values)
                }
        
        return summary
    
    def get_metric(self, name: str) -> Optional[Dict]:
        """
        Get statistics for specific metric.
        
        Parameters:
            name: Metric name
        
        Returns:
            dict: Metric statistics or None
        """
        if name not in self._metrics or not self._metrics[name]:
            return None
        
        values = self._metrics[name]
        
        return {
            'count': len(values),
            'mean': sum(values) / len(values),
            'min': min(values),
            'max': max(values),
            'latest': values[-1],
            'total': sum(values)
        }
    
    def reset(self) -> None:
        """Reset all metrics."""
        self._metrics.clear()
        self._timestamps.clear()
        self._start_times.clear()
        logger.info("Performance metrics reset")
    
    def format_summary(self) -> str:
        """
        Get formatted summary string.
        
        Returns:
            str: Formatted summary
        """
        summary = self.get_summary()
        
        if not summary:
            return "No metrics recorded"
        
        lines = ["Performance Summary:"]
        lines.append("=" * 50)
        
        for name, stats in summary.items():
            lines.append(f"{name}:")
            lines.append(f"  Count: {stats['count']}")
            lines.append(f"  Mean:  {stats['mean']:.4f}")
            lines.append(f"  Min:   {stats['min']:.4f}")
            lines.append(f"  Max:   {stats['max']:.4f}")
        
        return "\n".join(lines)

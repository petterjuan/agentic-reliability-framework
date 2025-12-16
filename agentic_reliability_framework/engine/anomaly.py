"""
Advanced Anomaly Detection with adaptive thresholds
Extracted from app.py for modularity
"""

import numpy as np
import threading
import logging
from typing import Dict, List
from collections import deque

from ..models import ReliabilityEvent
from ..config import config

logger = logging.getLogger(__name__)


class AdvancedAnomalyDetector:
    """Enhanced anomaly detection with adaptive thresholds"""
    
    def __init__(self) -> None:
        self.historical_data: deque[ReliabilityEvent] = deque(maxlen=100)
        self.adaptive_thresholds: Dict[str, float] = {
            'latency_p99': config.latency_warning,
            'error_rate': config.error_rate_warning
        }
        self._lock = threading.RLock()
        logger.info("Initialized AdvancedAnomalyDetector")
    
    def detect_anomaly(self, event: ReliabilityEvent) -> bool:
        """Detect if event is anomalous using adaptive thresholds"""
        with self._lock:
            latency_anomaly: bool = event.latency_p99 > self.adaptive_thresholds['latency_p99']
            error_anomaly: bool = event.error_rate > self.adaptive_thresholds['error_rate']
            
            resource_anomaly: bool = False
            if event.cpu_util is not None and event.cpu_util > config.cpu_critical:
                resource_anomaly = True
            if event.memory_util is not None and event.memory_util > config.memory_critical:
                resource_anomaly = True
            
            self._update_thresholds(event)
            
            is_anomaly: bool = latency_anomaly or error_anomaly or resource_anomaly
            
            if is_anomaly:
                logger.info(
                    f"Anomaly detected for {event.component}: "
                    f"latency={latency_anomaly}, error={error_anomaly}, "
                    f"resource={resource_anomaly}"
                )
            
            return is_anomaly
    
    def _update_thresholds(self, event: ReliabilityEvent) -> None:
        """Update adaptive thresholds based on historical data"""
        self.historical_data.append(event)
        
        if len(self.historical_data) > 10:
            recent_latencies: List[float] = [e.latency_p99 for e in list(self.historical_data)[-20:]]
            new_threshold: float = float(np.percentile(recent_latencies, 90))
            self.adaptive_thresholds['latency_p99'] = new_threshold
            logger.debug(f"Updated adaptive latency threshold to {new_threshold:.2f}ms")
    
    def get_thresholds(self) -> Dict[str, float]:
        """Get current adaptive thresholds"""
        with self._lock:
            return self.adaptive_thresholds.copy()

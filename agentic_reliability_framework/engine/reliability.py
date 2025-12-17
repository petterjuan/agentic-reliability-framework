"""
V2 Reliability Engine - Core anomaly detection and healing action generation.
Base implementation for ARF v2 functionality.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum

from ..models import ReliabilityEvent, EventSeverity
from ..config import config

logger = logging.getLogger(__name__)

# Default thresholds
DEFAULT_ERROR_THRESHOLD = 0.05
DEFAULT_LATENCY_THRESHOLD = 150.0


@dataclass
class V2HealingAction:
    """V2 healing action data structure"""
    action: str
    component: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "action": self.action,
            "component": self.component,
            "parameters": self.parameters,
            "confidence": self.confidence,
            "description": self.description,
            "metadata": self.metadata
        }


class V2ReliabilityEngine:
    """V2 reliability engine with basic anomaly detection and healing"""

    def __init__(self):
        self._lock = threading.RLock()
        self._start_time = time.time()
        
        # Initialize metrics
        self.metrics: Dict[str, Union[int, float]] = {
            "events_processed": 0,
            "anomalies_detected": 0,
            "healing_actions_generated": 0,
        }
        
        # Initialize event store
        self.event_store = ThreadSafeEventStore()

    async def process_event(self, event: ReliabilityEvent) -> Dict[str, Any]:
        """
        Process event using v2 pipeline
        
        Args:
            event: ReliabilityEvent to process
            
        Returns:
            Dictionary with processing results
        """
        return await self._v2_process(event)

    async def _v2_process(self, event: ReliabilityEvent, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Original v2 processing logic"""
        try:
            # Simulate v2 processing
            await asyncio.sleep(0.01)  # Simulate processing time
            
            # Get thresholds from config or use defaults
            error_threshold = getattr(config, 'error_threshold', DEFAULT_ERROR_THRESHOLD)
            latency_threshold = getattr(config, 'latency_threshold', DEFAULT_LATENCY_THRESHOLD)
            
            # Convert severity value to int if needed
            severity_value = event.severity.value if hasattr(event.severity, 'value') else "low"
            severity_numeric: int
            
            # Proper enum value handling
            if isinstance(severity_value, str):
                # Map string severity to numeric value
                severity_map = {"low": 1, "medium": 2, "high": 3, "critical": 4}
                severity_numeric = severity_map.get(severity_value.lower(), 1)
            elif isinstance(severity_value, Enum):
                # Handle enum members - get their value
                enum_value = severity_value.value
                if isinstance(enum_value, (int, float)):
                    severity_numeric = int(enum_value)
                elif isinstance(enum_value, str):
                    # Map string enum value
                    severity_map = {"low": 1, "medium": 2, "high": 3, "critical": 4}
                    severity_numeric = severity_map.get(enum_value.lower(), 1)
                else:
                    severity_numeric = 1
            else:
                # Handle direct numeric values (for IntEnum or similar)
                try:
                    severity_numeric = int(severity_value)
                except (TypeError, ValueError):
                    severity_numeric = 1
            
            # Basic anomaly detection
            is_anomaly = (
                event.error_rate > error_threshold or
                event.latency_p99 > latency_threshold or
                severity_numeric >= 2
            )
            
            result: Dict[str, Any] = {
                "status": "ANOMALY" if is_anomaly else "NORMAL",
                "incident_id": f"inc_{int(time.time())}_{event.component}",
                "component": event.component,
                "severity": severity_numeric,
                "detected_at": time.time(),
                "confidence": 0.85 if is_anomaly else 0.95,
                "healing_actions": self._generate_healing_actions(event) if is_anomaly else [],
            }
            
            # Update metrics
            with self._lock:
                self.metrics["events_processed"] += 1
                if is_anomaly:
                    self.metrics["anomalies_detected"] += 1
                    self.metrics["healing_actions_generated"] += len(result["healing_actions"])
            
            return result
            
        except Exception as e:
            logger.error(f"Error in v2 processing: {e}", exc_info=True)
            return {
                "status": "ERROR",
                "incident_id": "",
                "error": str(e),
                "healing_actions": []
            }

    def _generate_healing_actions(self, event: ReliabilityEvent) -> List[Dict[str, Any]]:
        """Generate healing actions based on event"""
        actions: List[Dict[str, Any]] = []
        
        # Get thresholds from config or use defaults
        error_threshold = getattr(config, 'error_threshold', DEFAULT_ERROR_THRESHOLD)
        latency_threshold = getattr(config, 'latency_threshold', DEFAULT_LATENCY_THRESHOLD)
        
        if event.error_rate > error_threshold:
            actions.append({
                "action": "restart_service",
                "component": event.component,
                "parameters": {"force": True},
                "confidence": 0.7,
                "description": f"Restart {event.component} due to high error rate",
                "metadata": {"trigger": "error_rate", "threshold": error_threshold}
            })
        
        if event.latency_p99 > latency_threshold:
            actions.append({
                "action": "scale_up",
                "component": event.component,
                "parameters": {"instances": 2},
                "confidence": 0.6,
                "description": f"Scale up {event.component} due to high latency",
                "metadata": {"trigger": "latency", "threshold": latency_threshold}
            })
        
        # Convert severity value to int if needed
        severity_value = event.severity.value if hasattr(event.severity, 'value') else "low"
        severity_numeric: int
        
        # Proper enum value handling (same as above)
        if isinstance(severity_value, str):
            severity_map = {"low": 1, "medium": 2, "high": 3, "critical": 4}
            severity_numeric = severity_map.get(severity_value.lower(), 1)
        elif isinstance(severity_value, Enum):
            # Handle enum members - get their value
            enum_value = severity_value.value
            if isinstance(enum_value, (int, float)):
                severity_numeric = int(enum_value)
            elif isinstance(enum_value, str):
                # Map string enum value
                severity_map = {"low": 1, "medium": 2, "high": 3, "critical": 4}
                severity_numeric = severity_map.get(enum_value.lower(), 1)
            else:
                severity_numeric = 1
        else:
            # Handle direct numeric values (for IntEnum or similar)
            try:
                severity_numeric = int(severity_value)
            except (TypeError, ValueError):
                severity_numeric = 1
        
        if severity_numeric >= 3:
            actions.append({
                "action": "escalate_to_team",
                "component": event.component,
                "parameters": {"team": "sre", "urgency": "high"},
                "confidence": 0.9,
                "description": f"Escalate {event.component} to SRE team",
                "metadata": {"trigger": "severity", "level": severity_numeric}
            })
        
        # Sort by confidence
        actions.sort(key=lambda x: float(x.get("confidence", 0.0)), reverse=True)
        return actions

    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics"""
        return {
            **self.metrics,
            "uptime_seconds": time.time() - self._start_time,
            "engine_version": "v2",
            "event_store_count": self.event_store.count(),
        }

    def get_engine_stats(self) -> Dict[str, Any]:
        """Alias for get_stats"""
        return self.get_stats()


# Thread-safe event store
class ThreadSafeEventStore:
    """Thread-safe event store for v2 engine"""
    def __init__(self) -> None:
        self._events: List[Any] = []
        self._lock = threading.RLock()
    
    def add_event(self, event: Any) -> None:
        """Add event to store"""
        with self._lock:
            self._events.append(event)
    
    def add(self, event: Any) -> None:
        """Alias for add_event"""
        self.add_event(event)
    
    def get_events(self, limit: int = 100) -> List[Any]:
        """Get events from store"""
        with self._lock:
            return self._events[-limit:] if self._events else []
    
    def get_recent(self, limit: int = 100) -> List[Any]:
        """Alias for get_events"""
        return self.get_events(limit)
    
    def clear(self) -> None:
        """Clear all events"""
        with self._lock:
            self._events.clear()
    
    def count(self) -> int:
        """Count events in store"""
        with self._lock:
            return len(self._events)


# Factory function for backward compatibility
def ReliabilityEngine(*args: Any, **kwargs: Any) -> V2ReliabilityEngine:
    """Factory function for v2 engine"""
    return V2ReliabilityEngine(*args, **kwargs)

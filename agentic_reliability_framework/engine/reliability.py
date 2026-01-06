"""
V3 Reliability Engine - Core implementation with backward compatibility.
This file contains the base V3 engine that v3_reliability.py extends.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum

from ..models import ReliabilityEvent, EventSeverity
from ..config import config

logger = logging.getLogger(__name__)

# Default thresholds
DEFAULT_ERROR_THRESHOLD = 0.05
DEFAULT_LATENCY_THRESHOLD = 150.0
DEFAULT_LEARNING_MIN_DATA_POINTS = 5
DEFAULT_CONFIDENCE_ANOMALY = 0.85
DEFAULT_CONFIDENCE_NORMAL = 0.95


@dataclass
class MCPResponse:
    """MCP response data structure for backward compatibility"""
    executed: bool = False
    status: str = "unknown"
    result: Dict[str, Any] = field(default_factory=dict)
    message: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "executed": self.executed,
            "status": self.status,
            "result": self.result,
            "message": self.message
        }

    def __repr__(self) -> str:
        return f"MCPResponse(status={self.status}, executed={self.executed})"


class V3ReliabilityEngine:
    """Base V3 reliability engine with core functionality"""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        Initialize base V3 engine.
        Args:
            *args: Positional arguments for backward compatibility
            **kwargs: Keyword arguments for backward compatibility
        """
        self.rag = kwargs.get('rag_graph')
        self.mcp = kwargs.get('mcp_server')
        self._lock = threading.RLock()
        self._start_time = time.time()

        # Initialize metrics
        self.metrics: Dict[str, Union[int, float]] = {
            "events_processed": 0,
            "anomalies_detected": 0,
            "rag_queries": 0,
            "mcp_executions": 0,
            "successful_outcomes": 0,
            "failed_outcomes": 0,
        }

        # Thread-safe event store
        self.event_store = ThreadSafeEventStore()

        logger.info("Initialized V3ReliabilityEngine (base implementation)")

    async def process_event(self, event: ReliabilityEvent) -> Dict[str, Any]:
        """
        Main entry point for processing an event.
        Automatically stores events and tracks metrics.
        """
        return await self.process_event_enhanced(event=event)

    async def process_event_enhanced(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """
        Enhanced event processing with v2 logic.
        Automatically adds event to store and updates metrics.
        """
        event = kwargs.get("event") or (args[0] if args else None)
        if not event or not isinstance(event, ReliabilityEvent):
            return {
                "status": "ERROR",
                "incident_id": "",
                "error": "Invalid event",
                "healing_actions": []
            }

        try:
            # Store event
            self.event_store.add_event(event)

            # Simulate processing delay
            await asyncio.sleep(0.01)

            # Thresholds per component (fallback to global defaults)
            error_threshold = getattr(config, 'error_threshold', DEFAULT_ERROR_THRESHOLD)
            latency_threshold = getattr(config, 'latency_threshold', DEFAULT_LATENCY_THRESHOLD)

            # Severity numeric mapping
            severity_numeric = self._severity_to_numeric(event.severity)

            # Basic anomaly detection
            is_anomaly = (
                event.error_rate > error_threshold or
                event.latency_p99 > latency_threshold or
                severity_numeric >= 2
            )

            confidence_anomaly = getattr(config, 'confidence_anomaly', DEFAULT_CONFIDENCE_ANOMALY)
            confidence_normal = getattr(config, 'confidence_normal', DEFAULT_CONFIDENCE_NORMAL)

            result: Dict[str, Any] = {
                "status": "ANOMALY" if is_anomaly else "NORMAL",
                "incident_id": f"inc_{int(time.time())}_{event.component}",
                "component": event.component,
                "severity": severity_numeric,
                "detected_at": time.time(),
                "confidence": confidence_anomaly if is_anomaly else confidence_normal,
                "healing_actions": self._generate_healing_actions(event) if is_anomaly else [],
                "processing_version": "v3_base",
            }

            # Update metrics safely
            with self._lock:
                self.metrics["events_processed"] += 1
                if is_anomaly:
                    self.metrics["anomalies_detected"] += 1

            return result

        except Exception as e:
            incident_id = f"inc_{int(time.time())}_{getattr(event, 'component', 'unknown')}"
            logger.error(f"Error in v2 processing (incident {incident_id}): {e}", exc_info=True)
            return {
                "status": "ERROR",
                "incident_id": incident_id,
                "error": str(e),
                "healing_actions": []
            }

    def _severity_to_numeric(self, severity: Union[str, Enum, int, float]) -> int:
        """Convert severity to numeric value"""
        mapping = {"low": 1, "medium": 2, "high": 3, "critical": 4}
        if isinstance(severity, Enum):
            severity = getattr(severity, 'value', 1)
        if isinstance(severity, str):
            return mapping.get(severity.lower(), 1)
        try:
            return int(severity)
        except Exception:
            return 1

    def _generate_healing_actions(self, event: ReliabilityEvent) -> List[Dict[str, Any]]:
        """Generate healing actions based on event"""
        actions: List[Dict[str, Any]] = []

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

        severity_str = str(getattr(event.severity, 'value', 'low')).lower()
        if severity_str in ['high', 'critical', '3', '4']:
            actions.append({
                "action": "escalate_to_team",
                "component": event.component,
                "parameters": {"team": "sre", "urgency": "high"},
                "confidence": 0.9,
                "description": f"Escalate {event.component} to SRE team",
                "metadata": {"trigger": "severity", "level": severity_str}
            })

        # Sort by confidence descending
        actions.sort(key=lambda x: float(x.get("confidence", 0.0)), reverse=True)
        return actions

    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics"""
        stats: Dict[str, Any] = {
            "events_processed": self.metrics["events_processed"],
            "anomalies_detected": self.metrics["anomalies_detected"],
            "anomaly_ratio": round(
                self.metrics["anomalies_detected"] / max(self.metrics["events_processed"], 1), 4
            ),
            "rag_queries": self.metrics["rag_queries"],
            "mcp_executions": self.metrics["mcp_executions"],
            "successful_outcomes": self.metrics["successful_outcomes"],
            "failed_outcomes": self.metrics["failed_outcomes"],
            "uptime_seconds": time.time() - self._start_time,
            "engine_version": "v3_base",
            "event_store_count": self.event_store.count(),
        }
        return stats

    def get_engine_stats(self) -> Dict[str, Any]:
        """Alias for get_stats"""
        return self.get_stats()

    def shutdown(self) -> None:
        """Graceful shutdown"""
        logger.info("Shutting down V3ReliabilityEngine...")


# Backward compatibility aliases
class EnhancedReliabilityEngine(V3ReliabilityEngine):
    """Alias for V3ReliabilityEngine for backward compatibility"""
    
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        logger.warning("EnhancedReliabilityEngine is an alias for V3ReliabilityEngine")
        super().__init__(*args, **kwargs)


# Thread-safe event store
class ThreadSafeEventStore:
    """Thread-safe event store"""

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
def ReliabilityEngine(*args: Any, **kwargs: Any) -> V3ReliabilityEngine:
    """Factory function for backward compatibility"""
    return V3ReliabilityEngine(*args, **kwargs)

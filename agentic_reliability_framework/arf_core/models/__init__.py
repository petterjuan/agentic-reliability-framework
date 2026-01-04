"""
OSS Models Module
Apache 2.0 Licensed
"""

from .healing_intent import (
    HealingIntent,
    HealingIntentSerializer,
    HealingIntentError,
    SerializationError,
    ValidationError,
    IntentSource,
    IntentStatus,
    create_rollback_intent,
    create_restart_intent,
    create_scale_out_intent,
    create_oss_advisory_intent,
)

# Define EventSeverity enum locally to avoid circular imports
from enum import Enum

class EventSeverity(Enum):
    """Severity levels for reliability events"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

# Create a local ReliabilityEvent class to avoid circular imports
from typing import Any, Optional, Dict, List
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional as Opt

@dataclass
class ReliabilityEvent:
    """Local ReliabilityEvent for OSS to avoid circular imports"""
    component: str
    severity: Any
    latency_p99: float = 100.0
    error_rate: float = 0.05
    throughput: float = 1000.0
    cpu_util: float = 0.5
    memory_util: float = 0.5
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Convert severity to string if it's an enum"""
        if hasattr(self.severity, 'value'):
            self.severity = self.severity.value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {
            "component": self.component,
            "severity": self.severity,
            "latency_p99": self.latency_p99,
            "error_rate": self.error_rate,
            "throughput": self.throughput,
            "cpu_util": self.cpu_util,
            "memory_util": self.memory_util,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }

def create_compatible_event(
    component: str,
    severity: Any,
    latency_p99: float = 100.0,
    error_rate: float = 0.05,
    throughput: float = 1000.0,
    cpu_util: float = 0.5,
    memory_util: float = 0.5,
    timestamp: Optional[datetime] = None,
    **extra_kwargs: Any
) -> ReliabilityEvent:
    """
    Create a ReliabilityEvent that's compatible with both Pydantic and dataclass expectations
    
    This is a factory function that returns an object with the right attributes
    regardless of whether the Pydantic model is available.
    """
    event_kwargs = {
        "component": component,
        "severity": severity,
        "latency_p99": latency_p99,
        "error_rate": error_rate,
        "throughput": throughput,
        "cpu_util": cpu_util,
        "memory_util": memory_util,
    }
    
    if timestamp is not None:
        event_kwargs["timestamp"] = timestamp
    
    if extra_kwargs:
        event_kwargs["metadata"] = extra_kwargs
    
    return ReliabilityEvent(**event_kwargs)

# For backward compatibility with existing code
class ReliabilityEventCompat(ReliabilityEvent):
    """Compatibility wrapper that behaves like both dataclass and Pydantic model"""
    
    @classmethod
    def parse_obj(cls, data: Dict[str, Any]) -> 'ReliabilityEventCompat':
        """Mimic Pydantic's parse_obj method"""
        return cls(**data)
    
    def dict(self) -> Dict[str, Any]:
        """Mimic Pydantic's dict method"""
        return self.to_dict()
    
    def json(self) -> str:
        """Mimic Pydantic's json method"""
        import json
        return json.dumps(self.to_dict())

__all__ = [
    "HealingIntent",
    "HealingIntentSerializer",
    "HealingIntentError",
    "SerializationError",
    "ValidationError",
    "IntentSource",
    "IntentStatus",
    "create_rollback_intent",
    "create_restart_intent",
    "create_scale_out_intent",
    "create_oss_advisory_intent",
    "ReliabilityEvent",
    "ReliabilityEventCompat",
    "EventSeverity",
    "create_compatible_event",
]

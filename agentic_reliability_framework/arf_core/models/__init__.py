# arf_core/models/__init__.py - FIXED VERSION
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

# Define EventSeverity enum
from enum import Enum

class EventSeverity(Enum):
    """Severity levels for reliability events"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

# Define ReliabilityEvent compatibility wrapper
from typing import Any, Optional, Dict
from datetime import datetime

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
):
    """
    Create a ReliabilityEvent that's compatible with both Pydantic and dataclass expectations
    
    This is a factory function that returns an object with the right attributes
    regardless of whether the Pydantic model is available.
    """
    # Convert severity if it's an EventSeverity enum
    severity_value = severity.value if hasattr(severity, 'value') else str(severity)
    
    try:
        # Try to use Pydantic model (primary for RAG graph)
        from agentic_reliability_framework.models import ReliabilityEvent as PydanticEvent
        from agentic_reliability_framework.models import EventSeverity as PydanticEventSeverity
        
        # Map severity string to Pydantic EventSeverity enum
        severity_map = {
            "low": PydanticEventSeverity.LOW,
            "medium": PydanticEventSeverity.MEDIUM,
            "high": PydanticEventSeverity.HIGH,
            "critical": PydanticEventSeverity.CRITICAL
        }
        
        pydantic_severity = severity_map.get(severity_value.lower(), PydanticEventSeverity.MEDIUM)
        
        # Create Pydantic event
        event_kwargs = {
            "component": component,
            "severity": pydantic_severity,
            "latency_p99": latency_p99,
            "error_rate": error_rate,
            "throughput": throughput,
            "cpu_util": cpu_util if cpu_util is not None else 0.5,
            "memory_util": memory_util if memory_util is not None else 0.5,
        }
        
        # Add timestamp if provided (Pydantic model has default)
        if timestamp is not None:
            event_kwargs["timestamp"] = timestamp
        
        # Add any extra kwargs
        event_kwargs.update(extra_kwargs)
        
        return PydanticEvent(**event_kwargs)
        
    except ImportError:
        # Fallback to dataclass when Pydantic is not available
        from dataclasses import dataclass
        
        @dataclass
        class FallbackReliabilityEvent:
            component: str
            severity: Any
            latency_p99: float = 100.0
            error_rate: float = 0.05
            throughput: float = 1000.0
            cpu_util: float = 0.5
            memory_util: float = 0.5
            timestamp: Optional[datetime] = None
            
            def __post_init__(self):
                if self.timestamp is None:
                    self.timestamp = datetime.now()
                
                # Ensure severity is a string
                if hasattr(self.severity, 'value'):
                    self.severity = self.severity.value
        
        return FallbackReliabilityEvent(
            component=component,
            severity=severity,
            latency_p99=latency_p99,
            error_rate=error_rate,
            throughput=throughput,
            cpu_util=cpu_util,
            memory_util=memory_util,
            timestamp=timestamp,
        )

# For backward compatibility, create a class-like interface
class ReliabilityEvent:
    """Compatibility wrapper for ReliabilityEvent"""
    
    def __new__(cls, *args, **kwargs):
        return create_compatible_event(*args, **kwargs)

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
    "EventSeverity",
    "create_compatible_event",
]

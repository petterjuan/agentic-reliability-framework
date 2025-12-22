# arf_core/models/__init__.py
"""
OSS Models Module - Contains all OSS data models
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

# Re-export ReliabilityEvent and EventSeverity from main package
# This is the KEY FIX for your import issue
try:
    from agentic_reliability_framework.models import ReliabilityEvent, EventSeverity
except ImportError:
    # Fallback implementation if main models not available
    from enum import Enum
    from dataclasses import dataclass
    from datetime import datetime
    
    class EventSeverity(Enum):
        LOW = "low"
        MEDIUM = "medium"
        HIGH = "high"
        CRITICAL = "critical"
    
    @dataclass
    class ReliabilityEvent:
        """Fallback ReliabilityEvent for OSS edition"""
        component: str
        severity: EventSeverity
        latency_p99: float = 100.0
        error_rate: float = 0.05
        throughput: float = 1000.0
        cpu_util: float = 0.5
        memory_util: float = 0.5
        timestamp: datetime = None
        
        def __post_init__(self):
            if self.timestamp is None:
                from datetime import datetime
                self.timestamp = datetime.now()

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
]

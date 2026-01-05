"""
OSS Models Module - ARF Core
Apache 2.0 Licensed

Exports OSS-compatible models from the main models module.
Uses lazy imports to avoid circular dependencies.
"""

import sys
from typing import TYPE_CHECKING

# Use TYPE_CHECKING for static type analysis only
if TYPE_CHECKING:
    from agentic_reliability_framework.models import (
        ReliabilityEvent,
        EventSeverity,
        create_compatible_event as _create_compatible_event,
    )

# Export healing intent components (these are in arf_core, no circular issue)
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

# ============================================================================
# LAZY IMPORT SYSTEM FOR MAIN MODELS
# ============================================================================

# Singleton cache for main models
_MAIN_MODELS_CACHE = {
    "ReliabilityEvent": None,
    "EventSeverity": None,
    "create_compatible_event": None,
}

def _lazy_import_main_models():
    """Lazily import main models to avoid circular dependencies"""
    try:
        # Try to import from main models module
        from agentic_reliability_framework.models import (
            ReliabilityEvent,
            EventSeverity,
            create_compatible_event,
        )
        
        _MAIN_MODELS_CACHE["ReliabilityEvent"] = ReliabilityEvent
        _MAIN_MODELS_CACHE["EventSeverity"] = EventSeverity
        _MAIN_MODELS_CACHE["create_compatible_event"] = create_compatible_event
        
    except ImportError as e:
        # If main models aren't available, create minimal compatibility shims
        # These should only be used as fallback in OSS edition
        
        from enum import Enum
        from dataclasses import dataclass, field
        from datetime import datetime
        from typing import Any, Dict, Optional
        
        # Minimal EventSeverity enum
        class EventSeverity(Enum):
            LOW = "low"
            MEDIUM = "medium"
            HIGH = "high"
            CRITICAL = "critical"
        
        # Minimal ReliabilityEvent dataclass
        @dataclass
        class ReliabilityEvent:
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
            """Create compatibility event (fallback version)"""
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
        
        _MAIN_MODELS_CACHE["ReliabilityEvent"] = ReliabilityEvent
        _MAIN_MODELS_CACHE["EventSeverity"] = EventSeverity
        _MAIN_MODELS_CACHE["create_compatible_event"] = create_compatible_event

# ============================================================================
# PUBLIC API - LAZY LOADED ATTRIBUTES
# ============================================================================

class _ModelsModule:
    """Module proxy that provides lazy-loaded main models"""
    
    @property
    def ReliabilityEvent(self):
        """Get ReliabilityEvent class (lazy loaded)"""
        if _MAIN_MODELS_CACHE["ReliabilityEvent"] is None:
            _lazy_import_main_models()
        return _MAIN_MODELS_CACHE["ReliabilityEvent"]
    
    @property
    def EventSeverity(self):
        """Get EventSeverity enum (lazy loaded)"""
        if _MAIN_MODELS_CACHE["EventSeverity"] is None:
            _lazy_import_main_models()
        return _MAIN_MODELS_CACHE["EventSeverity"]
    
    @property
    def create_compatible_event(self):
        """Get create_compatible_event function (lazy loaded)"""
        if _MAIN_MODELS_CACHE["create_compatible_event"] is None:
            _lazy_import_main_models()
        return _MAIN_MODELS_CACHE["create_compatible_event"]

# Create module proxy instance
_models_proxy = _ModelsModule()

# Export attributes through the proxy
ReliabilityEvent = _models_proxy.ReliabilityEvent
EventSeverity = _models_proxy.EventSeverity
create_compatible_event = _models_proxy.create_compatible_event

# ============================================================================
# COMPATIBILITY WRAPPERS
# ============================================================================

class ReliabilityEventCompat:
    """
    Compatibility wrapper that works with both Pydantic and dataclass versions
    
    This provides a consistent API regardless of which implementation is loaded.
    """
    
    @staticmethod
    def create(event_data: Dict[str, Any]) -> 'ReliabilityEvent':
        """Create a ReliabilityEvent from data"""
        if _MAIN_MODELS_CACHE["ReliabilityEvent"] is None:
            _lazy_import_main_models()
        
        ReliabilityEventClass = _MAIN_MODELS_CACHE["ReliabilityEvent"]
        
        # Check if it's a Pydantic model
        if hasattr(ReliabilityEventClass, 'parse_obj'):
            return ReliabilityEventClass.parse_obj(event_data)
        else:
            # Assume it's a dataclass
            return ReliabilityEventClass(**event_data)
    
    @staticmethod
    def to_dict(event) -> Dict[str, Any]:
        """Convert event to dictionary"""
        if hasattr(event, 'dict'):
            return event.dict()
        elif hasattr(event, 'to_dict'):
            return event.to_dict()
        else:
            # Fallback: use __dict__
            return {k: v for k, v in event.__dict__.items() if not k.startswith('_')}
    
    @staticmethod
    def json(event) -> str:
        """Convert event to JSON"""
        import json
        event_dict = ReliabilityEventCompat.to_dict(event)
        return json.dumps(event_dict, default=str)

# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Healing Intent (from arf_core)
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
    
    # Main models (lazy loaded)
    "ReliabilityEvent",
    "EventSeverity",
    "create_compatible_event",
    
    # Compatibility wrapper
    "ReliabilityEventCompat",
]

# ============================================================================
# MODULE INITIALIZATION
# ============================================================================

# Pre-load main models if we're not in a test environment
# This ensures they're available when needed
if "pytest" not in sys.modules and "test" not in sys.argv[0]:
    try:
        _lazy_import_main_models()
    except Exception:
        # Silently fail - models will be loaded on demand
        pass

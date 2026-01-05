# agentic_reliability_framework/__init__.py
"""
Agentic Reliability Framework (ARF) - OSS Edition
Production-grade multi-agent AI for reliability monitoring (Advisory only)
Apache 2.0 Licensed

Main package entry point with OSS boundary enforcement.
"""

# ============================================================================
# VERSION - IMPORT FIRST
# ============================================================================

from .__version__ import __version__

# ============================================================================
# DIRECT ABSOLUTE IMPORTS - NO CIRCULAR DEPENDENCIES
# ============================================================================

# IMPORTANT: Import DIRECTLY from source modules, not through intermediate modules
# This breaks the circular dependency chain

try:
    # 1. Import HealingIntent directly from its module
    from .arf_core.models.healing_intent import (
        HealingIntent,
        HealingIntentSerializer,
        create_rollback_intent,
        create_restart_intent,
        create_scale_out_intent,
        create_oss_advisory_intent,
        IntentSource,
        IntentStatus,
    )
    
    # 2. Import OSSMCPClient from the COMPREHENSIVE implementation
    # FIXED: Use oss_mcp_client.py (1000+ lines) not simple_mcp_client.py (84 lines)
    from .arf_core.engine.oss_mcp_client import (
        OSSMCPClient,
        create_oss_mcp_client,
        OSSMCPResponse,
        OSSAnalysisResult,
    )
    
    # 3. Import constants directly
    from .arf_core.constants import (
        OSS_EDITION,
        OSS_LICENSE,
        EXECUTION_ALLOWED,
        MCP_MODES_ALLOWED,
        MAX_INCIDENT_NODES,
        MAX_OUTCOME_NODES,
        validate_oss_config,
        get_oss_capabilities,
        check_oss_compliance,
        OSSBoundaryError,
    )
    
    # 4. Import core models
    from .arf_core.models import (
        ReliabilityEvent,
        EventSeverity,
        create_compatible_event,
    )
    
    # 5. Import engine factory from main engine module
    from .engine.engine_factory import (
        EngineFactory,
        create_engine,
        get_engine,
        get_oss_engine_capabilities,
    )
    
    OSS_AVAILABLE = True
    
except ImportError as e:
    OSS_AVAILABLE = False
    # Don't print in production - use logging if needed
    import logging
    logger = logging.getLogger(__name__)
    logger.debug(f"OSS components not available during import: {e}")
    
    # Create minimal stubs for emergency fallback
    class HealingIntent:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
        
        def to_enterprise_request(self):
            return {"error": "oss_not_available"}
    
    class HealingIntentSerializer:
        @staticmethod
        def serialize(intent):
            return {"error": "oss_not_available"}
    
    def create_rollback_intent(*args, **kwargs):
        return HealingIntent(action="rollback", component="unknown")
    
    def create_restart_intent(*args, **kwargs):
        return HealingIntent(action="restart_container", component="unknown")
    
    def create_scale_out_intent(*args, **kwargs):
        return HealingIntent(action="scale_out", component="unknown")
    
    def create_oss_advisory_intent(*args, **kwargs):
        return HealingIntent(action="unknown", component="unknown")
    
    class OSSMCPClient:
        def __init__(self, config=None):
            self.mode = "advisory"
            self.config = config or {}
        
        async def execute_tool(self, request_dict):
            return {"error": "oss_not_available", "executed": False}
    
    def create_oss_mcp_client(config=None):
        return OSSMCPClient(config)
    
    class OSSMCPResponse:
        pass
    
    class OSSAnalysisResult:
        pass
    
    class ReliabilityEvent:
        pass
    
    class EventSeverity:
        LOW = "low"
        MEDIUM = "medium"
        HIGH = "high"
        CRITICAL = "critical"
    
    def create_compatible_event(*args, **kwargs):
        return ReliabilityEvent()
    
    class EngineFactory:
        def create_engine(self, config=None):
            return OSSMCPClient(config)
    
    def create_engine(config=None):
        return OSSMCPClient(config)
    
    def get_engine(config=None):
        return OSSMCPClient(config)
    
    def get_oss_engine_capabilities():
        return {"available": False}
    
    def validate_oss_config(config=None):
        return {"status": "oss_not_available"}
    
    def get_oss_capabilities():
        return {"available": False}
    
    def check_oss_compliance():
        return False
    
    class OSSBoundaryError(Exception):
        pass
    
    class IntentSource:
        OSS_ANALYSIS = "oss_analysis"
        RAG_SIMILARITY = "rag_similarity"
    
    class IntentStatus:
        CREATED = "created"
        OSS_ADVISORY_ONLY = "oss_advisory_only"
    
    OSS_EDITION = "open-source"
    OSS_LICENSE = "Apache 2.0"
    EXECUTION_ALLOWED = False
    MCP_MODES_ALLOWED = ("advisory",)
    MAX_INCIDENT_NODES = 1000
    MAX_OUTCOME_NODES = 5000

# ============================================================================
# PUBLIC API - MINIMAL & CLEAN
# ============================================================================

__all__ = [
    # Version
    "__version__",
    
    # OSS Constants
    "OSS_EDITION",
    "OSS_LICENSE", 
    "EXECUTION_ALLOWED",
    "MCP_MODES_ALLOWED",
    "MAX_INCIDENT_NODES",
    "MAX_OUTCOME_NODES",
    "validate_oss_config",
    "get_oss_capabilities",
    "check_oss_compliance",
    "OSSBoundaryError",
    
    # OSS Models
    "HealingIntent",
    "HealingIntentSerializer",
    "IntentSource",
    "IntentStatus",
    "create_rollback_intent", 
    "create_restart_intent",
    "create_scale_out_intent",
    "create_oss_advisory_intent",
    
    # Core Models
    "ReliabilityEvent",
    "EventSeverity",
    "create_compatible_event",
    
    # OSS Engine
    "OSSMCPClient",
    "OSSMCPResponse",
    "OSSAnalysisResult",
    "create_oss_mcp_client",
    
    # Engine Factory
    "EngineFactory",
    "create_engine",
    "get_engine",
    "get_oss_engine_capabilities",
    
    # Availability
    "OSS_AVAILABLE",
]

# ============================================================================
# LAZY LOADING FOR HEAVY MODULES ONLY
# ============================================================================

from importlib import import_module
from typing import Any

# Map for lazy loading of non-core components
_map_module_attr: dict[str, tuple[str, str]] = {
    # App components (not part of OSS core)
    "SimplePredictiveEngine": (".engine.predictive", "SimplePredictiveEngine"),
    "BusinessImpactCalculator": (".engine.business", "BusinessImpactCalculator"),
    "AdvancedAnomalyDetector": (".engine.anomaly", "AdvancedAnomalyDetector"),
    "BusinessMetricsTracker": (".engine.business", "BusinessMetricsTracker"),
    "EnhancedReliabilityEngine": (".engine.reliability", "EnhancedReliabilityEngine"),
    "ThreadSafeEventStore": (".engine.reliability", "ThreadSafeEventStore"),
    "V3ReliabilityEngine": (".engine.v3_reliability", "V3ReliabilityEngine"),
    "MCPServer": (".engine.mcp_server", "MCPServer"),
    "MCPMode": (".engine.mcp_server", "MCPMode"),
    "MCPRequest": (".engine.mcp_server", "MCPRequest"),
    "MCPResponse": (".engine.mcp_server", "MCPResponse"),
}

def __getattr__(name: str) -> Any:
    """
    Lazy-load heavy modules on attribute access.
    OSS core components are already imported above.
    """
    if name in globals():
        return globals()[name]
    
    entry = _map_module_attr.get(name)
    if entry is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    
    module_name, attr_name = entry
    
    try:
        module = import_module(module_name, package=__package__)
        return getattr(module, attr_name)
    except ImportError as exc:
        raise AttributeError(
            f"Could not lazy-load {name} from {module_name}: {exc}"
        ) from exc


def __dir__() -> list[str]:
    """Expose the declared public symbols for tab-completion."""
    std = set(globals().keys())
    return sorted(std.union(__all__))

# ============================================================================
# NO PRINT STATEMENTS ON IMPORT - Use logging if needed
# ============================================================================

# Removed all print statements - they pollute output on import
# If you need startup messages, use proper logging

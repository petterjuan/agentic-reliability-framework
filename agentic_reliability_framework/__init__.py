"""
Agentic Reliability Framework (ARF)
Production-grade multi-agent AI for reliability monitoring
"""

from importlib import import_module
from typing import Any, TYPE_CHECKING

from .__version__ import __version__  # runtime import for __version__

# ============================================================================
# DIRECT OSS IMPORTS - NO LAZY LOADING FOR OSS COMPONENTS
# ============================================================================

# Import OSS components DIRECTLY to avoid circular dependencies
try:
    from agentic_reliability_framework.arf_core import (
        # OSS Models
        HealingIntent,
        HealingIntentSerializer,
        create_rollback_intent,
        create_restart_intent,
        create_scale_out_intent,
        
        # OSS Engine
        OSSMCPClient,
        create_mcp_client,
        
        # OSS Constants & Validation
        validate_oss_constants,
        get_oss_capabilities,
        OSSBoundaryError,
        
        # OSS Metadata
        OSS_EDITION,
        OSS_LICENSE,
    )
    OSS_AVAILABLE = True
except ImportError as e:
    OSS_AVAILABLE = False
    print(f"⚠️  OSS components not available: {e}")
    
    # Create minimal stubs for OSS components
    class HealingIntent:
        pass
    
    class HealingIntentSerializer:
        pass
    
    class OSSMCPClient:
        def __init__(self):
            self.mode = "advisory"
    
    def create_mcp_client():
        return OSSMCPClient()
    
    def validate_oss_constants():
        return {"status": "oss_not_available"}
    
    def get_oss_capabilities():
        return {"available": False}
    
    class OSSBoundaryError(Exception):
        pass
    
    def create_rollback_intent():
        return None
    
    def create_restart_intent():
        return None
    
    def create_scale_out_intent():
        return None
    
    OSS_EDITION = True
    OSS_LICENSE = "Apache 2.0"

__all__ = [
    # Version
    "__version__",
    
    # === HEALING INTENT & OSS EXPORTS (DIRECT) ===
    "HealingIntent",
    "HealingIntentSerializer",
    "create_rollback_intent",
    "create_restart_intent",
    "create_scale_out_intent",
    "OSSMCPClient",
    "create_mcp_client",
    "validate_oss_constants",
    "get_oss_capabilities",
    "OSSBoundaryError",
    "OSS_EDITION",
    "OSS_LICENSE",
    
    # === CORE ENGINES (LAZY LOADED) ===
    "V3ReliabilityEngine",
    "EnhancedReliabilityEngine",
    "ReliabilityEngine",
    "EnhancedV3ReliabilityEngine",
    
    # === OTHER ENGINES ===
    "SimplePredictiveEngine",
    "BusinessImpactCalculator",
    "AdvancedAnomalyDetector",
    "create_enhanced_ui",
    
    # === LAZY LOADERS ===
    "get_engine",
    "get_agents",
    "get_faiss_index",
    "get_business_metrics",
    "enhanced_engine",
]

# Inform static analyzers/types about the exported names
if TYPE_CHECKING:  # pragma: no cover - static-analysis only
    # === HEALING INTENT & OSS ===
    HealingIntent: Any
    HealingIntentSerializer: Any
    create_rollback_intent: Any
    create_restart_intent: Any
    create_scale_out_intent: Any
    OSSMCPClient: Any
    create_mcp_client: Any
    validate_oss_constants: Any
    get_oss_capabilities: Any
    OSSBoundaryError: Any
    
    # === CORE ENGINES ===
    V3ReliabilityEngine: Any
    EnhancedReliabilityEngine: Any
    ReliabilityEngine: Any
    EnhancedV3ReliabilityEngine: Any
    
    # === OTHER ENGINES ===
    SimplePredictiveEngine: Any
    BusinessImpactCalculator: Any
    AdvancedAnomalyDetector: Any
    create_enhanced_ui: Any
    
    # === LAZY LOADERS ===
    get_engine: Any
    get_agents: Any
    get_faiss_index: Any
    get_business_metrics: Any
    enhanced_engine: Any

# ============================================================================
# LAZY LOADING FOR NON-OSS COMPONENTS ONLY
# ============================================================================

_map_module_attr: dict[str, tuple[str, str]] = {
    # === CORE ENGINES ===
    "V3ReliabilityEngine": (".engine.reliability", "V3ReliabilityEngine"),
    "EnhancedReliabilityEngine": (".engine.reliability", "EnhancedReliabilityEngine"),
    "ReliabilityEngine": (".engine.reliability", "ReliabilityEngine"),
    
    # === ENHANCED V3 ENGINE ===
    "EnhancedV3ReliabilityEngine": (".engine.v3_reliability", "V3ReliabilityEngine"),
    
    # === OTHER ENGINES ===
    "SimplePredictiveEngine": (".app", "SimplePredictiveEngine"),
    "BusinessImpactCalculator": (".app", "BusinessImpactCalculator"),
    "AdvancedAnomalyDetector": (".app", "AdvancedAnomalyDetector"),
    "create_enhanced_ui": (".app", "create_enhanced_ui"),
    
    # === LAZY LOADERS ===
    "get_engine": (".lazy", "get_engine"),
    "get_agents": (".lazy", "get_agents"),
    "get_faiss_index": (".lazy", "get_faiss_index"),
    "get_business_metrics": (".lazy", "get_business_metrics"),
    "enhanced_engine": (".lazy", "get_enhanced_reliability_engine"),
}

def __getattr__(name: str) -> Any:
    """
    Lazy-load heavy modules on attribute access.
    OSS components are imported directly above.
    """
    entry = _map_module_attr.get(name)
    if entry is None:
        # Check if it's an OSS component (should be in globals already)
        oss_components = [
            "HealingIntent", "HealingIntentSerializer", "OSSMCPClient", 
            "create_mcp_client", "validate_oss_constants", "get_oss_capabilities",
            "OSSBoundaryError", "OSS_EDITION", "OSS_LICENSE",
            "create_rollback_intent", "create_restart_intent", "create_scale_out_intent"
        ]
        if name in oss_components:
            if not OSS_AVAILABLE:
                raise AttributeError(
                    f"OSS component '{name}' not available. "
                    f"The arf_core module failed to import."
                )
            return globals().get(name)
        
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = entry
    
    try:
        # Handle relative imports
        module: Any
        if module_name.startswith("."):
            module = import_module(module_name, package=__package__)
        else:
            module = import_module(module_name)
            
        return getattr(module, attr_name)
    except ImportError as exc:
        raise AttributeError(
            f"module {module_name!r} not found: {exc}"
        ) from exc
    except AttributeError as exc:
        raise AttributeError(
            f"module {module.__name__!r} has no attribute {attr_name!r}"
        ) from exc


def __dir__() -> list[str]:
    """Expose the declared public symbols for tab-completion."""
    std = set(globals().keys())
    return sorted(std.union(__all__))


# Print helpful info on import
if __name__ != "__main__":
    import sys
    if "pytest" not in sys.modules and "test" not in sys.argv[0]:
        print(f"✅ Agentic Reliability Framework v{__version__}")
        if OSS_AVAILABLE:
            print(f"📦 OSS Edition: HealingIntent, OSSMCPClient (advisory-only)")
        else:
            print(f"⚠️  OSS components not available")
        print(f"🔗 EnhancedV3ReliabilityEngine ready")

"""
Agentic Reliability Framework (ARF)
Production-grade multi-agent AI for reliability monitoring
"""

from importlib import import_module
from typing import Any, TYPE_CHECKING

from .__version__ import __version__  # runtime import for __version__

__all__ = [
    # Version
    "__version__",
    
    # === HEALING INTENT & OSS EXPORTS ===
    "HealingIntent",                     # OSS->Enterprise boundary
    "HealingIntentSerializer",           # Serialization utilities
    
    # === CORE ENGINES ===
    "V3ReliabilityEngine",
    "EnhancedReliabilityEngine",         # Backward compatibility alias
    "ReliabilityEngine",                 # Backward compatibility alias
    "EnhancedV3ReliabilityEngine",       # The actual enhanced v3 engine
    
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
    
    # === OSS-SPECIFIC EXPORTS ===
    "OSSMCPClient",                      # OSS-only MCP client (advisory)
    "create_mcp_client",                 # Factory for OSS client (NOTE: renamed from create_oss_mcp_client)
    "validate_oss_constants",            # OSS configuration validator
    "get_oss_capabilities",              # Get OSS edition capabilities
    "OSSBoundaryError",                  # OSS boundary violation error
    
    # === FACTORY FUNCTIONS ===
    "create_rollback_intent",            # Common intent creators
    "create_restart_intent",
    "create_scale_out_intent",
]

# Inform static analyzers/types about the exported names without importing modules.
if TYPE_CHECKING:  # pragma: no cover - static-analysis only
    # === HEALING INTENT ===
    HealingIntent: Any
    HealingIntentSerializer: Any
    
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
    
    # === OSS-SPECIFIC ===
    OSSMCPClient: Any
    create_mcp_client: Any  # NOTE: renamed
    validate_oss_constants: Any
    get_oss_capabilities: Any
    OSSBoundaryError: Any
    
    # === FACTORY FUNCTIONS ===
    create_rollback_intent: Any
    create_restart_intent: Any
    create_scale_out_intent: Any


def __getattr__(name: str) -> Any:
    """
    Lazy-load heavy modules on attribute access using importlib + getattr.
    """
    map_module_attr: dict[str, tuple[str, str]] = {
        # === HEALING INTENT EXPORTS ===
        "HealingIntent": ("agentic_reliability_framework.arf_core", "HealingIntent"),
        "HealingIntentSerializer": ("agentic_reliability_framework.arf_core", "HealingIntentSerializer"),
        
        # === OSS-SPECIFIC COMPONENTS ===
        "OSSMCPClient": ("agentic_reliability_framework.arf_core", "OSSMCPClient"),
        "create_mcp_client": ("agentic_reliability_framework.arf_core", "create_mcp_client"),
        "validate_oss_constants": ("agentic_reliability_framework.arf_core", "validate_oss_constants"),
        "get_oss_capabilities": ("agentic_reliability_framework.arf_core", "get_oss_capabilities"),
        "OSSBoundaryError": ("agentic_reliability_framework.arf_core", "OSSBoundaryError"),
        
        # === FACTORY FUNCTIONS ===
        "create_rollback_intent": ("agentic_reliability_framework.arf_core.models.healing_intent", "create_rollback_intent"),
        "create_restart_intent": ("agentic_reliability_framework.arf_core.models.healing_intent", "create_restart_intent"),
        "create_scale_out_intent": ("agentic_reliability_framework.arf_core.models.healing_intent", "create_scale_out_intent"),
        
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

    entry = map_module_attr.get(name)
    if entry is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = entry
    
    try:
        # Handle relative imports for core modules
        module: Any
        if module_name.startswith("."):
            module = import_module(module_name, package=__package__)
        else:
            # For arf_core imports (separate module)
            module = import_module(module_name)
            
        return getattr(module, attr_name)
    except ImportError as exc:
        # Provide helpful error message for missing OSS components
        if "arf_core" in module_name:
            raise AttributeError(
                f"OSS component '{name}' not available. "
                f"The arf_core module may not be installed or imported correctly. "
                f"Expected module: {module_name}"
            ) from exc
        raise AttributeError(
            f"module {module_name!r} not found: {exc}"
        ) from exc
    except AttributeError as exc:
        raise AttributeError(
            f"module {module.__name__!r} has no attribute {attr_name!r}"
        ) from exc


def __dir__() -> list[str]:
    """Expose the declared public symbols for tab-completion and tooling."""
    std = set(globals().keys())
    return sorted(std.union(__all__))


# Print helpful info on import (development only)
if __name__ != "__main__":
    import sys
    if "pytest" not in sys.modules and "test" not in sys.argv[0]:
        print(f"✅ Agentic Reliability Framework v{__version__}")
        print(f"📦 Includes: HealingIntent, OSSMCPClient, EnhancedV3ReliabilityEngine")
        print(f"🔗 OSS→Enterprise handoff ready")

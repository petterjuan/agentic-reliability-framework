"""
Agentic Reliability Framework - OSS Core Package
Apache 2.0 Licensed - Production-grade multi-agent AI for reliability monitoring
OSS Edition: Advisory mode only, no execution capability

This package contains the Open Source core components:
- HealingIntent model (OSS -> Enterprise handoff)
- OSS MCP Client (advisory only)
- OSS configuration with hard limits
- Backward compatibility with existing framework

Upgrade to Enterprise for execution capabilities: https://arf.dev/enterprise
"""

from importlib import import_module
from typing import Any, TYPE_CHECKING

__all__ = [
    # === OSS-SPECIFIC EXPORTS ===
    "__version__",
    "OSS_EDITION",
    "OSS_LICENSE",
    "ENTERPRISE_UPGRADE_URL",
    
    # OSS Core Components
    "HealingIntent",
    "HealingIntentSerializer",
    "OSSMCPClient",
    "create_oss_mcp_client",
    "validate_oss_config",
    "get_oss_capabilities",
    "check_oss_compliance",
    "OSSBoundaryError",
    
    # OSS Factories
    "create_rollback_intent",
    "create_restart_intent",
    "create_scale_out_intent",
    
    # === BACKWARD COMPATIBILITY EXPORTS ===
    # V3 Reliability Engines
    "V3ReliabilityEngine",
    "EnhancedReliabilityEngine",
    "ReliabilityEngine",
    "EnhancedV3ReliabilityEngine",
    
    # Other Engines
    "SimplePredictiveEngine",
    "BusinessImpactCalculator",
    "AdvancedAnomalyDetector",
    "create_enhanced_ui",
    
    # Lazy Loaders
    "get_engine",
    "get_agents",
    "get_faiss_index",
    "get_business_metrics",
    "enhanced_engine",
    
    # Existing Framework Components
    "MCPServer",
    "MCPMode",
    "MCPRequest",
    "MCPResponse",
    "MCPRequestStatus",
    "config",
]

# === OSS EDITION METADATA ===
OSS_EDITION: str = "open-source"
OSS_LICENSE: str = "Apache 2.0"
ENTERPRISE_UPGRADE_URL: str = "https://arf.dev/enterprise"

# Import version from parent package
try:
    from ..__version__ import __version__
except ImportError:
    __version__ = "3.3.0-oss"

# Inform static analyzers/types about the exported names without importing modules.
if TYPE_CHECKING:  # pragma: no cover - static-analysis only
    # === OSS-SPECIFIC TYPE HINTS ===
    HealingIntent: Any
    HealingIntentSerializer: Any
    OSSMCPClient: Any
    create_oss_mcp_client: Any
    validate_oss_config: Any
    get_oss_capabilities: Any
    check_oss_compliance: Any
    OSSBoundaryError: Any
    
    # OSS Factory Functions
    create_rollback_intent: Any
    create_restart_intent: Any
    create_scale_out_intent: Any
    
    # === BACKWARD COMPATIBILITY TYPE HINTS ===
    # V3 Reliability Engines
    V3ReliabilityEngine: Any
    EnhancedReliabilityEngine: Any
    ReliabilityEngine: Any
    EnhancedV3ReliabilityEngine: Any
    
    # Other Engines
    SimplePredictiveEngine: Any
    BusinessImpactCalculator: Any
    AdvancedAnomalyDetector: Any
    create_enhanced_ui: Any
    
    # Lazy Loaders
    get_engine: Any
    get_agents: Any
    get_faiss_index: Any
    get_business_metrics: Any
    enhanced_engine: Any
    
    # Existing Framework Components
    MCPServer: Any
    MCPMode: Any
    MCPRequest: Any
    MCPResponse: Any
    MCPRequestStatus: Any
    config: Any

# REMOVED: EventSeverity and ReliabilityEvent definitions from here
# They should ONLY be in arf_core/models/__init__.py

def __getattr__(name: str) -> Any:
    """
    Lazy-load heavy modules on attribute access using importlib + getattr.
    
    This provides backward compatibility while enabling OSS/Enterprise separation.
    """
    map_module_attr: dict[str, tuple[str, str]] = {
        # === OSS-SPECIFIC MODULES ===
        "HealingIntent": (".models.healing_intent", "HealingIntent"),
        "HealingIntentSerializer": (".models.healing_intent", "HealingIntentSerializer"),
        "OSSMCPClient": (".engine.oss_mcp_client", "OSSMCPClient"),
        "create_oss_mcp_client": (".engine.oss_mcp_client", "create_oss_mcp_client"),
        "validate_oss_config": (".constants", "validate_oss_config"),
        "get_oss_capabilities": (".constants", "get_oss_capabilities"),
        "check_oss_compliance": (".constants", "check_oss_compliance"),
        "OSSBoundaryError": (".constants", "OSSBoundaryError"),
        
        # OSS Factory Functions
        "create_rollback_intent": (".models.healing_intent", "create_rollback_intent"),
        "create_restart_intent": (".models.healing_intent", "create_restart_intent"),
        "create_scale_out_intent": (".models.healing_intent", "create_scale_out_intent"),
        
        # === BACKWARD COMPATIBILITY MODULES ===
        # Base V3 engine (v2 functionality)
        "V3ReliabilityEngine": ("agentic_reliability_framework.engine.reliability", "V3ReliabilityEngine"),
        "EnhancedReliabilityEngine": ("agentic_reliability_framework.engine.reliability", "EnhancedReliabilityEngine"),
        "ReliabilityEngine": ("agentic_reliability_framework.engine.reliability", "ReliabilityEngine"),
        
        # Enhanced V3 engine (with RAG+MCP)
        "EnhancedV3ReliabilityEngine": ("agentic_reliability_framework.engine.v3_reliability", "V3ReliabilityEngine"),
        
        # Other engines
        "SimplePredictiveEngine": ("agentic_reliability_framework.app", "SimplePredictiveEngine"),
        "BusinessImpactCalculator": ("agentic_reliability_framework.app", "BusinessImpactCalculator"),
        "AdvancedAnomalyDetector": ("agentic_reliability_framework.app", "AdvancedAnomalyDetector"),
        "create_enhanced_ui": ("agentic_reliability_framework.app", "create_enhanced_ui"),
        
        # Lazy loaders
        "get_engine": ("agentic_reliability_framework.lazy", "get_engine"),
        "get_agents": ("agentic_reliability_framework.lazy", "get_agents"),
        "get_faiss_index": ("agentic_reliability_framework.lazy", "get_faiss_index"),
        "get_business_metrics": ("agentic_reliability_framework.lazy", "get_business_metrics"),
        "enhanced_engine": ("agentic_reliability_framework.lazy", "get_enhanced_reliability_engine"),
        
        # Existing Framework Components
        "MCPServer": ("agentic_reliability_framework.engine.mcp_server", "MCPServer"),
        "MCPMode": ("agentic_reliability_framework.engine.mcp_server", "MCPMode"),
        "MCPRequest": ("agentic_reliability_framework.engine.mcp_server", "MCPRequest"),
        "MCPResponse": ("agentic_reliability_framework.engine.mcp_server", "MCPResponse"),
        "MCPRequestStatus": ("agentic_reliability_framework.engine.mcp_server", "MCPRequestStatus"),
        "config": ("agentic_reliability_framework.config", "config"),
    }

    entry = map_module_attr.get(name)
    if entry is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = entry
    
    try:
        # Handle relative imports for OSS modules
        imported_module: Any
        if module_name.startswith("."):
            imported_module = import_module(module_name, package=__package__)
        else:
            imported_module = import_module(module_name)
            
        return getattr(imported_module, attr_name)
    except ImportError as exc:
        # If OSS module not found, provide helpful error message
        if module_name.startswith("."):
            raise AttributeError(
                f"OSS component '{name}' not available. "
                f"This may be an Enterprise installation or the OSS package is incomplete. "
                f"Expected module: {module_name}"
            ) from exc
        raise AttributeError(
            f"module not found: {module_name}"
        ) from exc
    except AttributeError as exc:
        raise AttributeError(
            f"module has no attribute {attr_name!r}"
        ) from exc


def __dir__() -> list[str]:
    """Expose the declared public symbols for tab-completion and tooling."""
    std = set(globals().keys())
    return sorted(std.union(__all__))


# Print OSS edition info on import (development only)
if __name__ != "__main__":
    print(f"✅ ARF OSS Core Package v{__version__} ({OSS_LICENSE})")
    print(f"⚠️  OSS Edition: Advisory mode only")
    print(f"💼 Upgrade to Enterprise for execution: {ENTERPRISE_UPGRADE_URL}")

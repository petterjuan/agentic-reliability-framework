# agentic_reliability_framework/__init__.py (URGENT FIX)
"""
Agentic Reliability Framework (ARF) - OSS Edition
Production-grade multi-agent AI for reliability monitoring (Advisory only)
Apache 2.0 Licensed
"""

# ============================================================================
# VERSION - IMPORT FIRST
# ============================================================================

from .__version__ import __version__

# ============================================================================
# DIRECT ABSOLUTE IMPORTS - NO CIRCULAR DEPENDENCIES
# ============================================================================

# IMPORTANT: Import DIRECTLY from source modules, not through arf_core
# This breaks the circular dependency chain

try:
    # 1. Import HealingIntent directly from its module
    from agentic_reliability_framework.arf_core.models.healing_intent import (
        HealingIntent,
        HealingIntentSerializer,
        create_rollback_intent,
        create_restart_intent,
        create_scale_out_intent
    )
    
    # 2. Import OSSMCPClient directly
    from agentic_reliability_framework.arf_core.engine.simple_mcp_client import (
        OSSMCPClient,
        create_mcp_client
    )
    
    # 3. Import constants directly
    from agentic_reliability_framework.arf_core.constants import (
        OSS_EDITION,
        OSS_LICENSE,
        EXECUTION_ALLOWED,
        MCP_MODES_ALLOWED,
        validate_oss_config,
        get_oss_capabilities,
        OSSBoundaryError
    )
    
    OSS_AVAILABLE = True
    
except ImportError as e:
    OSS_AVAILABLE = False
    print(f"⚠️  OSS components not available: {e}")
    
    # Create minimal stubs for emergency fallback
    class HealingIntent:
        pass
    
    class HealingIntentSerializer:
        pass
    
    def create_rollback_intent():
        return None
    
    def create_restart_intent():
        return None
    
    def create_scale_out_intent():
        return None
    
    class OSSMCPClient:
        def __init__(self):
            self.mode = "advisory"
    
    def create_mcp_client():
        return OSSMCPClient()
    
    def validate_oss_config():
        return {"status": "oss_not_available"}
    
    def get_oss_capabilities():
        return {"available": False}
    
    class OSSBoundaryError(Exception):
        pass
    
    OSS_EDITION = True
    OSS_LICENSE = "Apache 2.0"
    EXECUTION_ALLOWED = False
    MCP_MODES_ALLOWED = ("advisory",)

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
    "validate_oss_config",
    "get_oss_capabilities",
    "OSSBoundaryError",
    
    # OSS Models
    "HealingIntent",
    "HealingIntentSerializer",
    "create_rollback_intent", 
    "create_restart_intent",
    "create_scale_out_intent",
    
    # OSS Engine
    "OSSMCPClient",
    "create_mcp_client",
    
    # Availability
    "OSS_AVAILABLE",
]

# ============================================================================
# LAZY LOADING FOR HEAVY MODULES ONLY
# ============================================================================

from importlib import import_module
from typing import Any

_map_module_attr: dict[str, tuple[str, str]] = {
    # App components (not part of OSS core)
    "SimplePredictiveEngine": (".app", "SimplePredictiveEngine"),
    "BusinessImpactCalculator": (".app", "BusinessImpactCalculator"),
    "AdvancedAnomalyDetector": (".app", "AdvancedAnomalyDetector"),
    "create_enhanced_ui": (".app", "create_enhanced_ui"),
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
        raise AttributeError(f"module {module_name!r} not found: {exc}") from exc


def __dir__() -> list[str]:
    """Expose the declared public symbols for tab-completion."""
    std = set(globals().keys())
    return sorted(std.union(__all__))

# ============================================================================
# IMPORT VERIFICATION
# ============================================================================

if __name__ != "__main__":
    import sys
    if "pytest" not in sys.modules and "test" not in sys.argv[0]:
        print(f"✅ Agentic Reliability Framework v{__version__} (OSS Edition)")
        if OSS_AVAILABLE:
            print(f"📦 HealingIntent & OSSMCPClient available (advisory-only)")
            # Quick sanity check
            try:
                hi = HealingIntent(action="test", component="test")
                print(f"✓ HealingIntent instantiation successful")
            except Exception as e:
                print(f"⚠️  HealingIntent instantiation failed: {e}")
        else:
            print(f"❌ OSS core components not available - package is BROKEN")

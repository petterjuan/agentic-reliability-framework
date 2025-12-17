"""
Agentic Reliability Framework (ARF)
Production-grade multi-agent AI for reliability monitoring
"""

from importlib import import_module
from typing import Any, TYPE_CHECKING

from .__version__ import __version__  # runtime import for __version__

__all__ = [
    "__version__",
    "V2ReliabilityEngine",
    "V3ReliabilityEngine",
    "EnhancedReliabilityEngine",  # Backward compatibility alias
    "ReliabilityEngine",  # Backward compatibility alias
    "SimplePredictiveEngine",
    "BusinessImpactCalculator",
    "AdvancedAnomalyDetector",
    "create_enhanced_ui",
    "get_engine",
    "get_agents",
    "get_faiss_index",
    "get_business_metrics",
    "enhanced_engine",
]

# Inform static analyzers/types about the exported names without importing modules.
# We intentionally *don't* import real symbols here (no `from .app import ...`) so we
# preserve lazy runtime imports and avoid mypy trying to resolve attributes on modules.
if TYPE_CHECKING:  # pragma: no cover - static-analysis only
    V2ReliabilityEngine: Any
    V3ReliabilityEngine: Any
    EnhancedReliabilityEngine: Any  # Backward compatibility
    ReliabilityEngine: Any  # Backward compatibility
    SimplePredictiveEngine: Any
    BusinessImpactCalculator: Any
    AdvancedAnomalyDetector: Any
    create_enhanced_ui: Any
    get_engine: Any
    get_agents: Any
    get_faiss_index: Any
    get_business_metrics: Any
    enhanced_engine: Any


def __getattr__(name: str) -> Any:
    """
    Lazy-load heavy modules on attribute access using importlib + getattr.
    This avoids importing heavy modules at package import time and prevents
    static analyzers from trying to resolve attributes of other modules.
    """
    map_module_attr: dict[str, tuple[str, str]] = {
        # V2 engine (basic functionality)
        "V2ReliabilityEngine": (".engine.reliability", "V2ReliabilityEngine"),
        "ReliabilityEngine": (".engine.reliability", "ReliabilityEngine"),
        
        # V3 engine (enhanced with RAG+MCP)
        "V3ReliabilityEngine": (".engine.v3_reliability", "V3ReliabilityEngine"),
        "EnhancedReliabilityEngine": (".engine.v3_reliability", "V3ReliabilityEngine"),  # Alias
        
        # Other engines
        "SimplePredictiveEngine": (".app", "SimplePredictiveEngine"),
        "BusinessImpactCalculator": (".app", "BusinessImpactCalculator"),
        "AdvancedAnomalyDetector": (".app", "AdvancedAnomalyDetector"),
        "create_enhanced_ui": (".app", "create_enhanced_ui"),
        
        # Lazy loaders
        "get_engine": (".lazy", "get_engine"),
        "get_agents": (".lazy", "get_agents"),
        "get_faiss_index": (".lazy", "get_faiss_index"),
        "get_business_metrics": (".lazy", "get_business_metrics"),
        "enhanced_engine": (".lazy", "enhanced_engine"),
    }

    entry = map_module_attr.get(name)
    if entry is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = entry
    # importlib.import_module returns ModuleType; cast to Any to avoid attribute checks
    module: Any = import_module(module_name, package=__package__)
    try:
        return getattr(module, attr_name)
    except AttributeError as exc:
        # Preserve a helpful message for runtime debugging
        raise AttributeError(
            f"module {module.__name__!r} has no attribute {attr_name!r}"
        ) from exc


def __dir__() -> list[str]:
    """Expose the declared public symbols for tab-completion and tooling."""
    std = set(globals().keys())
    return sorted(std.union(__all__))

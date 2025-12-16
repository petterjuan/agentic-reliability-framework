"""
Agentic Reliability Framework (ARF)
Production-grade multi-agent AI for reliability monitoring
"""

from typing import TYPE_CHECKING, Any

from .__version__ import __version__  # runtime import for __version__

# Exports (keep names here for runtime and for static analysis)
__all__ = [
    "__version__",
    "EnhancedReliabilityEngine",
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

# Static-only imports to satisfy linters / static analyzers (Ruff / pyflakes).
# These imports are only executed during type-checking / static analysis and
# will not cause runtime imports. This preserves the lazy import behavior.
if TYPE_CHECKING:  # pragma: no cover - static analysis only
    from .app import (
        AdvancedAnomalyDetector,
        BusinessImpactCalculator,
        EnhancedReliabilityEngine,
        SimplePredictiveEngine,
        create_enhanced_ui,
    )
    from .lazy import (
        enhanced_engine,
        get_agents,
        get_engine,
        get_faiss_index,
        get_business_metrics,
    )

# Lazy imports at runtime using PEP 562 (__getattr__ on modules).
# This avoids the cost of importing heavy modules at package import time.
def __getattr__(name: str) -> Any:
    if name == "EnhancedReliabilityEngine":
        from .app import EnhancedReliabilityEngine

        return EnhancedReliabilityEngine
    if name == "SimplePredictiveEngine":
        from .app import SimplePredictiveEngine

        return SimplePredictiveEngine
    if name == "BusinessImpactCalculator":
        from .app import BusinessImpactCalculator

        return BusinessImpactCalculator
    if name == "AdvancedAnomalyDetector":
        from .app import AdvancedAnomalyDetector

        return AdvancedAnomalyDetector
    if name == "create_enhanced_ui":
        from .app import create_enhanced_ui

        return create_enhanced_ui

    if name in {
        "get_engine",
        "get_agents",
        "get_faiss_index",
        "get_business_metrics",
        "enhanced_engine",
    }:
        from .lazy import (
            enhanced_engine,
            get_agents,
            get_engine,
            get_faiss_index,
            get_business_metrics,
        )

        if name == "get_engine":
            return get_engine
        if name == "get_agents":
            return get_agents
        if name == "get_faiss_index":
            return get_faiss_index
        if name == "get_business_metrics":
            return get_business_metrics
        if name == "enhanced_engine":
            return enhanced_engine

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    # Helpful for autocompletion and tooling: show our public API.
    std = globals().keys()
    return sorted(set(list(std) + list(__all__)))

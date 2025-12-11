"""
Agentic Reliability Framework (ARF)
Multi-Agent AI System for Production Reliability Monitoring

ULTRA LAZY VERSION: No imports trigger FAISS loading
"""

from .__version__ import __version__

# ========== LAZY IMPORTS ==========
# DO NOT import from .app here - that triggers FAISS loading immediately!
# Instead, use __getattr__ to import modules only when accessed

__all__ = [
    "__version__",
    "get_engine",
    "get_agents", 
    "get_faiss_index",
    "get_business_metrics",
    "enhanced_engine",
    "EnhancedReliabilityEngine",
    "SimplePredictiveEngine", 
    "BusinessImpactCalculator",
    "AdvancedAnomalyDetector",
    "create_enhanced_ui",
]

# ========== LAZY LOADER ==========
def __getattr__(name):
    """
    Dynamically import modules only when accessed
    This prevents FAISS from loading on import
    """
    if name == 'EnhancedReliabilityEngine':
        from .app import EnhancedReliabilityEngine
        return EnhancedReliabilityEngine
    elif name == 'SimplePredictiveEngine':
        from .app import SimplePredictiveEngine
        return SimplePredictiveEngine
    elif name == 'BusinessImpactCalculator':
        from .app import BusinessImpactCalculator
        return BusinessImpactCalculator
    elif name == 'AdvancedAnomalyDetector':
        from .app import AdvancedAnomalyDetector
        return AdvancedAnomalyDetector
    elif name == 'create_enhanced_ui':
        from .app import create_enhanced_ui
        return create_enhanced_ui
    elif name == 'get_engine':
        from .lazy_init import get_engine
        return get_engine
    elif name == 'get_agents':
        from .lazy_init import get_agents
        return get_agents
    elif name == 'get_faiss_index':
        from .lazy_init import get_faiss_index
        return get_faiss_index
    elif name == 'get_business_metrics':
        from .lazy_init import get_business_metrics
        return get_business_metrics
    elif name == 'enhanced_engine':
        from .lazy_init import enhanced_engine
        return enhanced_engine
    else:
        raise AttributeError(f"module 'arf' has no attribute '{name}'")

# ========== CONVENIENCE FUNCTIONS ==========
# These provide direct function access while maintaining laziness

def get_engine():
    """Get or create the EnhancedReliabilityEngine (lazy)"""
    from .lazy_init import get_engine as _get_engine
    return _get_engine()

def get_agents():
    """Get or create agent instances (lazy)"""
    from .lazy_init import get_agents as _get_agents
    return _get_agents()

def get_faiss_index():
    """Get or create FAISS index (lazy)"""
    from .lazy_init import get_faiss_index as _get_faiss_index
    return _get_faiss_index()

def get_business_metrics():
    """Get or create BusinessMetricsTracker (lazy)"""
    from .lazy_init import get_business_metrics as _get_business_metrics
    return _get_business_metrics()

def enhanced_engine():
    """Alias for get_engine() (backward compatibility)"""
    from .lazy_init import enhanced_engine as _enhanced_engine
    return _enhanced_engine()

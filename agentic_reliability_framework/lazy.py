"""
Simple lazy loading for ARF - No circular dependencies!
"""

import threading
from typing import Callable, Optional, TypeVar
# numpy import removed - not used

T = TypeVar('T')

class LazyLoader:
    """Simple thread-safe lazy loader"""
    def __init__(self, loader_func: Callable[[], T]):
        self._loader_func = loader_func
        self._lock = threading.Lock()
        self._instance: Optional[T] = None
    
    def __call__(self) -> T:
        if self._instance is None:
            with self._lock:
                if self._instance is None:
                    self._instance = self._loader_func()
        return self._instance
    
    def reset(self) -> None:
        """Reset instance (for testing)"""
        with self._lock:
            self._instance = None


# ========== MODULE-LEVEL IMPORTS ==========

def _load_engine():
    """Import and create EnhancedReliabilityEngine"""
    # Import here to avoid circular dependency
    from .app import EnhancedReliabilityEngine
    return EnhancedReliabilityEngine()

def _load_agents():
    """Create all agent instances"""
    from .app import (
        AnomalyDetectionAgent, 
        RootCauseAgent, 
        PredictiveAgent,
        OrchestrationManager,
        SimplePredictiveEngine
    )
    
    predictive_engine = SimplePredictiveEngine()
    detective = AnomalyDetectionAgent()
    diagnostician = RootCauseAgent()
    predictive = PredictiveAgent(predictive_engine)
    
    return {
        'detective': detective,
        'diagnostician': diagnostician,
        'predictive': predictive,
        'manager': OrchestrationManager(detective, diagnostician, predictive),
        'predictive_engine': predictive_engine
    }

def _load_faiss_index():
    """Load or create FAISS index"""
    import os
    import json
    import faiss
    from .config import config
    from .app import ProductionFAISSIndex
    from sentence_transformers import SentenceTransformer
    
    # Load or create index
    if os.path.exists(config.INDEX_FILE):
        index = faiss.read_index(config.INDEX_FILE)
        incident_texts = []
        if os.path.exists(config.TEXTS_FILE):
            with open(config.TEXTS_FILE, 'r') as f:
                incident_texts = json.load(f)
    else:
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        dimension = model.get_sentence_embedding_dimension()
        index = faiss.IndexFlatIP(dimension)
        incident_texts = []
    
    return ProductionFAISSIndex(index, incident_texts)

def _load_business_metrics():
    """Create BusinessMetricsTracker"""
    from .app import BusinessMetricsTracker
    return BusinessMetricsTracker()


# ========== CREATE LAZY LOADERS ==========

engine_loader = LazyLoader(_load_engine)
agents_loader = LazyLoader(_load_agents)
faiss_loader = LazyLoader(_load_faiss_index)
business_metrics_loader = LazyLoader(_load_business_metrics)


# ========== PUBLIC API ==========

def get_engine():
    """Get or create EnhancedReliabilityEngine"""
    return engine_loader()

def get_agents():
    """Get or create agent instances"""
    return agents_loader()

def get_faiss_index():
    """Get or create FAISS index"""
    return faiss_loader()

def get_business_metrics():
    """Get or create BusinessMetricsTracker"""
    return business_metrics_loader()

def enhanced_engine():
    """Alias for get_engine()"""
    return get_engine()

def reset_all():
    """Reset all loaders (for testing)"""
    engine_loader.reset()
    agents_loader.reset()
    faiss_loader.reset()
    business_metrics_loader.reset()

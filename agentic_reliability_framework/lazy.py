"""
Simple, thread-safe lazy loading for ARF.
One file to rule them all.
"""

import threading
from typing import Any, Callable, Optional, TypeVar

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


# ========== LAZY LOADERS ==========

# Import functions (defined separately to avoid circular imports)
def _load_engine():
    from .app import EnhancedReliabilityEngine
    return EnhancedReliabilityEngine()

def _load_agents():
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
    import os
    import json
    import faiss
    from .config import config
    from .app import ProductionFAISSIndex
    from sentence_transformers import SentenceTransformer
    
    # Load or create index
    if os.path.exists(config.index_file):
        index = faiss.read_index(config.index_file)
        incident_texts = {}
        if os.path.exists(config.incident_texts_file):
            with open(config.incident_texts_file, 'r') as f:
                incident_texts = json.load(f)
    else:
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        dimension = model.get_sentence_embedding_dimension()
        index = faiss.IndexFlatIP(dimension)
        incident_texts = {}
    
    return ProductionFAISSIndex(index, incident_texts)

def _load_business_metrics():
    from .app import BusinessMetricsTracker
    return BusinessMetricsTracker()

# Create lazy loader instances
engine_loader = LazyLoader(_load_engine)
agents_loader = LazyLoader(_load_agents)
faiss_loader = LazyLoader(_load_faiss_index)
business_metrics_loader = LazyLoader(_load_business_metrics)

# Convenience functions
def get_engine():
    return engine_loader()

def get_agents():
    return agents_loader()

def get_faiss_index():
    return faiss_loader()

def get_business_metrics():
    return business_metrics_loader()

def enhanced_engine():
    return get_engine()

def reset_all():
    """Reset all loaders (for testing)"""
    engine_loader.reset()
    agents_loader.reset()
    faiss_loader.reset()
    business_metrics_loader.reset()

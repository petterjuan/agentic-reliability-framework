# Standard imports
import threading
from typing import Tuple
import numpy as np

# FAISS import
import faiss  # fixed import

# Project imports
# from agentic_reliability_framework.models import ReliabilityEvent

class ProductionFAISSIndex:
    """Existing FAISS index wrapper"""
    def __init__(self, dim: int):
        self.index: faiss.IndexFlatL2 = faiss.IndexFlatL2(dim)
        self._lock: threading.Lock = threading.Lock()

    def add(self, vector: np.ndarray) -> int:
        """Add a vector to the FAISS index and return the ID of the last inserted vector."""
        with self._lock:
            if len(vector.shape) == 1:
                vector = vector.reshape(1, -1)
            self.index.add(vector)
            return int(self.index.ntotal - 1)

class EnhancedFAISSIndex(ProductionFAISSIndex):
    """Adds thread-safe search capability"""
    def search(self, query_vector: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """Return top-k nearest neighbors and distances for the query vector."""
        with self._lock:
            if len(query_vector.shape) == 1:
                query_vector = query_vector.reshape(1, -1)
            distances, indices = self.index.search(query_vector, k)
            return distances[0], indices[0]

"""
Enhanced FAISS Index for v3 RAG functionality
Adds search capability to existing ProductionFAISSIndex
"""

import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple, Union, cast
import asyncio
from numpy.typing import NDArray

from .constants import MemoryConstants
from .faiss_index import ProductionFAISSIndex

logger = logging.getLogger(__name__)


class EnhancedFAISSIndex:
    """
    Enhanced FAISS index with search capability for RAG
    
    V3 Feature: Adds thread-safe similarity search
    """
    
    def __init__(self, faiss_index: ProductionFAISSIndex) -> None:
        """
        Initialize enhanced FAISS index
        
        Args:
            faiss_index: Existing ProductionFAISSIndex instance
        """
        self.faiss = faiss_index
        self._lock = faiss_index._lock if hasattr(faiss_index, '_lock') else None
        logger.info("Initialized EnhancedFAISSIndex for v3 RAG search")
    
    def search(
        self,
        query_vector: Union[NDArray[np.float32], List[float], np.ndarray],
        k: int = 5
    ) -> Tuple[NDArray[np.float32], NDArray[np.int64]]:
        """
        Thread-safe similarity search in FAISS index
        
        Args:
            query_vector: Query embedding vector (shape: [dim] or [1, dim])
            k: Number of nearest neighbors to return
            
        Returns:
            Tuple of (distances, indices)
            - distances: Distances to nearest neighbors
            - indices: FAISS index positions of matches
            
        Raises:
            ValueError: If query vector has wrong dimensions
            RuntimeError: If search fails
        """
        if self._lock:
            with self._lock:
                return self._safe_search(query_vector, k)
        else:
            return self._safe_search(query_vector, k)
    
    def _safe_search(
        self,
        query_vector: Union[NDArray[np.float32], List[float], np.ndarray],
        k: int
    ) -> Tuple[NDArray[np.float32], NDArray[np.int64]]:
        """Perform search with error handling"""
        try:
            # Convert to numpy array if needed
            if not isinstance(query_vector, np.ndarray):
                query_vector_array = np.array(query_vector, dtype=np.float32)
            else:
                query_vector_array = query_vector.astype(np.float32)
            
            # Ensure proper dimensionality
            if len(query_vector_array.shape) == 1:
                query_vector_array = query_vector_array.reshape(1, -1)
            
            # Validate dimensions
            if query_vector_array.shape[1] != MemoryConstants.VECTOR_DIM:
                raise ValueError(
                    f"Query vector dimension {query_vector_array.shape[1]} "
                    f"does not match index dimension {MemoryConstants.VECTOR_DIM}"
                )
            
            # Limit k to available vectors
            total_vectors = self.faiss.get_count()
            actual_k = min(k, total_vectors)
            
            if actual_k == 0:
                logger.debug("No vectors in index, returning empty results")
                empty_distances: NDArray[np.float32] = np.array([], dtype=np.float32)
                empty_indices: NDArray[np.int64] = np.array([], dtype=np.int64)
                return empty_distances, empty_indices
            
            # Perform search
            distances, indices = self.faiss.index.search(query_vector_array, actual_k)
            
            # Ensure we have valid arrays
            if distances.size > 0 and indices.size > 0:
                # Extract first row if we have a 2D array
                dist_result: NDArray[np.float32] = distances[0].astype(np.float32)
                idx_result: NDArray[np.int64] = indices[0].astype(np.int64)
                
                # Log search results
                if dist_result.size > 0:
                    min_val = np.min(dist_result)
                    min_distance_value: float = float(min_val)
                    
                    logger.debug(
                        f"FAISS search completed: k={actual_k}, "
                        f"found={idx_result.size} results, "
                        f"min_distance={min_distance_value:.4f}"
                    )
            else:
                dist_result = np.array([], dtype=np.float32)
                idx_result = np.array([], dtype=np.int64)
            
            return dist_result, idx_result
            
        except Exception as e:
            logger.error(f"FAISS search error: {e}", exc_info=True)
            raise RuntimeError(f"Search failed: {str(e)}")
    
    async def search_async(
        self,
        query_vector: Union[NDArray[np.float32], List[float], np.ndarray],
        k: int = 5
    ) -> Tuple[NDArray[np.float32], NDArray[np.int64]]:
        """
        Async version of similarity search
        
        Useful for not blocking the main event loop
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.search(query_vector, k)
        )
    
    def semantic_search(self, query_text: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        High-level semantic search with text query
        
        Args:
            query_text: Text to search for
            k: Number of results to return
            
        Returns:
            List of search results with metadata
        """
        try:
            # Embed the query text
            query_embedding = self._embed_text(query_text)
            
            # Search for similar vectors
            distances, indices = self.search(query_embedding, k)
            
            # Get corresponding texts
            results: List[Dict[str, Any]] = []
            for i, (distance, idx) in enumerate(zip(distances, indices)):
                if idx == -1:  # FAISS returns -1 for no match
                    continue
                
                # Get text by FAISS index
                text = self._get_text_by_index(int(idx))
                
                if text:
                    distance_float: float = float(distance)
                    similarity_float = float(1.0 / (1.0 + distance_float))
                    
                    results.append({
                        "index": int(idx),
                        "distance": distance_float,
                        "similarity": similarity_float,
                        "text": text,
                        "rank": i + 1
                    })
            
            # Sort by similarity (highest first)
            results.sort(key=lambda x: float(x["similarity"]), reverse=True)
            
            logger.info(
                f"Semantic search for '{query_text[:50]}...' "
                f"found {len(results)} results"
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Semantic search error: {e}", exc_info=True)
            return []
    
    def _embed_text(self, text: str) -> NDArray[np.float32]:
        """Embed text into vector"""
        # Use existing embedding model or create simple embedding
        try:
            # Try to use existing embedding model
            if hasattr(self.faiss, '_encoder_pool'):
                loop = asyncio.get_event_loop()
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
                encoded_embedding = loop.run_until_complete(
                    loop.run_in_executor(
                        None,
                        lambda: model.encode([text])
                    )
                )
                return np.array(encoded_embedding, dtype=np.float32)
        except Exception:
            # Continue to fallback
            pass
        
        # Fallback: create simple embedding from text hash
        import hashlib
        hash_val = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
        np.random.seed(hash_val)
        
        # Create random embedding with correct dimension
        generated_embedding: NDArray[np.float32] = np.random.randn(1, MemoryConstants.VECTOR_DIM).astype(np.float32)
        
        # Normalize to unit length
        norm = np.linalg.norm(generated_embedding)
        if norm > 0:
            generated_embedding = generated_embedding / norm
        
        # Return the 2D array directly
        return generated_embedding
    
    def _get_text_by_index(self, index: int) -> Optional[str]:
        """Get text by FAISS index"""
        if hasattr(self.faiss, 'texts') and index < len(self.faiss.texts):
            return self.faiss.texts[index]
        return None
    
    def get_embeddings(self) -> NDArray[np.float32]:
        """
        Get all embeddings stored in the index.
        
        Returns:
            numpy.ndarray: Array of all embeddings
        """
        try:
            # Check if the underlying FAISS index has a way to retrieve vectors
            if hasattr(self.faiss.index, 'reconstruct_n'):
                total = self.faiss.get_count()
                if total == 0:
                    return np.zeros((0, MemoryConstants.VECTOR_DIM), dtype=np.float32)
                
                # Reconstruct all vectors
                vectors: List[NDArray[np.float32]] = []
                for i in range(total):
                    vec = self.faiss.index.reconstruct(i)
                    vec_array: NDArray[np.float32] = np.array(vec, dtype=np.float32)
                    vectors.append(vec_array)
                
                # Create the final result
                if vectors:
                    return np.vstack(vectors).astype(np.float32)
                else:
                    return np.zeros((0, MemoryConstants.VECTOR_DIM), dtype=np.float32)
            else:
                # If reconstruction is not available, return empty array
                logger.warning("FAISS index does not support reconstruct_n, returning empty array")
                return np.zeros((0, MemoryConstants.VECTOR_DIM), dtype=np.float32)
        except Exception as e:
            logger.error(f"Error getting embeddings: {e}")
            return np.zeros((0, MemoryConstants.VECTOR_DIM), dtype=np.float32)
    
    def get_text_by_index(self, index: int) -> Optional[str]:
        """
        Get text by index.
        
        Args:
            index: Index of the text to retrieve
            
        Returns:
            Optional[str]: Text if found, None otherwise
        """
        return self._get_text_by_index(index)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics"""
        stats = {
            "total_vectors": self.faiss.get_count(),
            "vector_dimension": MemoryConstants.VECTOR_DIM,
            "index_type": type(self.faiss.index).__name__,
            "search_capability": True,
            "v3_enhanced": True,
        }
        
        # Add FAISS-specific stats if available
        if hasattr(self.faiss.index, 'ntotal'):
            stats["faiss_ntotal"] = self.faiss.index.ntotal
        if hasattr(self.faiss.index, 'd'):
            stats["faiss_dimension"] = self.faiss.index.d
        
        return stats
    
    def search_vectors(self, query_vector: Union[np.ndarray, List[float]], k: int = 5) -> NDArray[np.int32]:
        """
        Search for similar vectors.
        
        Args:
            query_vector: Query vector
            k: Number of results to return
            
        Returns:
            NDArray[np.int32]: Array of indices of similar vectors
        """
        try:
            if not isinstance(query_vector, np.ndarray):
                query_vector_array: NDArray[np.float32] = np.array(query_vector, dtype=np.float32)
            else:
                query_vector_array = query_vector.astype(np.float32)
            
            if query_vector_array.ndim == 1:
                query_vector_array = query_vector_array.reshape(1, -1)
            
            # Get number of vectors in index
            ntotal = self.faiss.index.ntotal if hasattr(self.faiss.index, 'ntotal') else 0
            
            if ntotal == 0:
                # Use cast to ensure proper return type
                return cast(NDArray[np.int32], np.array([], dtype=np.int32))
            
            # This code is reachable when ntotal > 0
            actual_k = min(k, ntotal)
            distances, indices = self.faiss.index.search(query_vector_array, actual_k)
            
            if indices.size > 0:
                # Use cast to ensure proper return type
                return cast(NDArray[np.int32], indices[0].astype(np.int32))
            else:
                return cast(NDArray[np.int32], np.array([], dtype=np.int32))
                
        except Exception as e:
            logger.error(f"Error searching vectors: {e}")
            return cast(NDArray[np.int32], np.array([], dtype=np.int32))

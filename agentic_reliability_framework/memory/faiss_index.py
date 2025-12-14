# agentic_reliability_framework/memory/faiss_index.py
"""
FAISS Index for production vector storage and retrieval
Extracted from app.py for modularity
"""

import os
import json
import numpy as np
import datetime
import threading
import logging
import asyncio
import tempfile
from typing import List, Tuple, Dict, Any
from queue import Queue
from concurrent.futures import ProcessPoolExecutor

import atomicwrites

# Import from local modules
from ..config import config
from ..models import ReliabilityEvent
from .constants import MemoryConstants

logger = logging.getLogger(__name__)


class ProductionFAISSIndex:
    """
    Production-safe FAISS index with single-writer pattern
    
    CRITICAL FIX: FAISS is NOT thread-safe for concurrent writes
    Solution: Queue-based single writer thread + atomic saves
    
    ENHANCED (v3): Added search capability for RAG functionality
    """
    
    def __init__(self, index, texts: List[str]):
        self.index = index
        self.texts = texts
        self._lock = threading.RLock()
        
        # FIXED: Initialize shutdown event BEFORE starting thread
        self._shutdown = threading.Event()
        
        # Single writer thread (no concurrent write conflicts)
        self._write_queue: Queue = Queue()
        self._writer_thread = threading.Thread(
            target=self._writer_loop,
            daemon=True,
            name="FAISSWriter"
        )
        self._writer_thread.start()  # ← Only start ONCE, AFTER _shutdown exists
        
        # ProcessPool for encoding (avoids GIL + memory leaks)
        self._encoder_pool = ProcessPoolExecutor(max_workers=2)
        
        logger.info(
            f"Initialized ProductionFAISSIndex with {len(texts)} vectors, "
            f"single-writer pattern"
        )
    
    def add_async(self, vector: np.ndarray, text: str) -> None:
        """
        Add vector and text asynchronously (thread-safe)
        
        FIXED: Queue-based design - no concurrent FAISS writes
        """
        self._write_queue.put((vector, text))
        logger.debug(f"Queued vector for indexing: {text[:50]}...")
    
    def _writer_loop(self) -> None:
        """
        Single writer thread - processes queue in batches
        
        This ensures only ONE thread ever writes to FAISS index
        """
        batch = []
        last_save = datetime.datetime.now()
        save_interval = datetime.timedelta(
            seconds=MemoryConstants.FAISS_SAVE_INTERVAL_SECONDS
        )
        
        while not self._shutdown.is_set():
            try:
                # Collect batch (non-blocking with timeout)
                import queue
                try:
                    item = self._write_queue.get(timeout=1.0)
                    batch.append(item)
                except queue.Empty:
                    pass
                
                # Process batch when ready
                if len(batch) >= MemoryConstants.FAISS_BATCH_SIZE or \
                   (batch and datetime.datetime.now() - last_save > save_interval):
                    
                    self._flush_batch(batch)
                    batch = []
                    
                    # Periodic save
                    if datetime.datetime.now() - last_save > save_interval:
                        self._save_atomic()
                        last_save = datetime.datetime.now()
                        
            except Exception as e:
                logger.error(f"Writer loop error: {e}", exc_info=True)
    
    def _flush_batch(self, batch: List[Tuple[np.ndarray, str]]) -> None:
        """
        Flush batch to FAISS index
        
        SAFE: Only called from single writer thread
        """
        if not batch:
            return
        
        try:
            vectors = np.vstack([v for v, _ in batch])
            texts = [t for _, t in batch]
            
            # SAFE: Single writer - no concurrent access
            self.index.add(vectors)
            
            with self._lock:  # Only lock for text list modification
                self.texts.extend(texts)
            
            logger.info(f"Flushed batch of {len(batch)} vectors to FAISS index")
            
        except Exception as e:
            logger.error(f"Error flushing batch: {e}", exc_info=True)
    
    def _save_atomic(self) -> None:
        """
        Atomic save with fsync for durability
        
        FIXED: Prevents corruption on crash
        """
        try:
            import faiss
            
            # Write to temporary file first
            with tempfile.NamedTemporaryFile(
                mode='wb',
                delete=False,
                dir=os.path.dirname(config.index_file),
                prefix='index_',
                suffix='.tmp'
            ) as tmp:
                temp_path = tmp.name
            
            # Write index
            faiss.write_index(self.index, temp_path)
            
            # Fsync for durability
            with open(temp_path, 'r+b') as f:
                f.flush()
                os.fsync(f.fileno())
            
            # Atomic rename
            os.replace(temp_path, config.index_file)
            
            # Save texts with atomic write
            with self._lock:
                texts_copy = self.texts.copy()
            
            with atomicwrites.atomic_write(
                config.incident_texts_file,
                mode='w',
                overwrite=True
            ) as f:
                json.dump(texts_copy, f)
            
            logger.info(
                f"Atomically saved FAISS index with {len(texts_copy)} vectors"
            )
            
        except Exception as e:
            logger.error(f"Error saving index: {e}", exc_info=True)
    
    def get_count(self) -> int:
        """Get total count of vectors"""
        with self._lock:
            return len(self.texts) + self._write_queue.qsize()
    
    def search(self, query_vector: np.ndarray, k: int = 5) -> Tuple[List[float], List[int], List[str]]:
        """
        Thread-safe similarity search in FAISS index
        
        Args:
            query_vector: Query embedding vector (shape: [dim] or [1, dim])
            k: Number of nearest neighbors to return
            
        Returns:
            Tuple of (scores, indices, texts):
            - scores: Inner product scores (higher = more similar)
            - indices: FAISS index positions of matches  
            - texts: Corresponding text metadata
            
        Thread-safe: Uses lock to prevent concurrent modification during search
        Note: FAISS IndexFlatIP returns inner product (cosine similarity)
              Higher values = more similar
        """
        with self._lock:
            # Check if index has any vectors
            if len(self.texts) == 0:
                logger.debug("FAISS index empty, returning empty search results")
                return [], [], []
            
            # Ensure proper dimensionality
            if len(query_vector.shape) == 1:
                query_vector = query_vector.reshape(1, -1)
            
            # Get FAISS index dimension
            index_dim = self.index.d if hasattr(self.index, 'd') else MemoryConstants.VECTOR_DIM
            
            # Validate vector dimensions match
            if query_vector.shape[1] != index_dim:
                logger.warning(
                    f"Vector dimension mismatch: query {query_vector.shape[1]} != index {index_dim}"
                )
                # SAFE FIX: Truncate or pad to match dimension
                if query_vector.shape[1] > index_dim:
                    query_vector = query_vector[:, :index_dim]
                    logger.debug(f"Truncated query vector to {index_dim} dimensions")
                else:
                    padded = np.zeros((1, index_dim), dtype=np.float32)
                    padded[:, :query_vector.shape[1]] = query_vector
                    query_vector = padded
                    logger.debug(f"Padded query vector to {index_dim} dimensions")
            
            try:
                # FAISS search (IndexFlatIP returns inner product scores)
                # Higher score = more similar (cosine similarity)
                # Limit k to available vectors
                actual_k = min(k, len(self.texts))
                scores, indices = self.index.search(query_vector, actual_k)
                
                # Convert to lists
                scores_list = scores[0].tolist()
                indices_list = indices[0].tolist()
                
                # Get corresponding texts
                texts_list = []
                for idx in indices_list:
                    if 0 <= idx < len(self.texts):
                        texts_list.append(self.texts[idx])
                    else:
                        texts_list.append("")  # Placeholder for invalid index
                
                logger.debug(
                    f"FAISS search completed: k={actual_k}, found={len(indices_list)} results, "
                    f"max_score={max(scores_list) if scores_list else 0:.4f}"
                )
                
                return scores_list, indices_list, texts_list
                
            except Exception as e:
                logger.error(f"FAISS search error: {e}", exc_info=True)
                return [], [], []
    
    async def find_similar_incidents(self, query_text: str, k: int = 5, 
                                    min_similarity: float = 0.3) -> List[Dict[str, Any]]:
        """
        High-level semantic search for similar incidents
        
        Args:
            query_text: Text description to search for
            k: Number of similar incidents to return
            min_similarity: Minimum similarity threshold (0-1)
            
        Returns:
            List of similar incidents with parsed metadata
            Returns empty list if no matches or error
            
        Uses existing ProcessPoolExecutor for encoding (non-blocking)
        """
        try:
            # Check if index has any vectors
            if len(self.texts) == 0:
                logger.debug("FAISS index empty, no similar incidents found")
                return []
            
            # Use existing encoder pool (non-blocking, avoids GIL)
            loop = asyncio.get_event_loop()
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            
            # Encode query text in process pool
            query_vector = await loop.run_in_executor(
                self._encoder_pool,
                model.encode,
                [query_text]
            )
            
            # Search using the new search() method
            scores, indices, texts = self.search(
                np.array(query_vector, dtype=np.float32),
                k
            )
            
            # Parse and filter results
            results = []
            for score, idx, text in zip(scores, indices, texts):
                # Skip invalid or empty results
                if not text or idx < 0:
                    continue
                
                # Filter by similarity threshold
                # Note: FAISS IndexFlatIP returns inner product, normalize to 0-1
                # Typical range: -1 to 1, but embeddings usually positive
                normalized_similarity = (score + 1) / 2  # Convert -1..1 to 0..1
                
                if normalized_similarity >= min_similarity:
                    # Parse text: "component latency error_rate analysis_text"
                    # Using split with maxsplit=3 to handle spaces in analysis_text
                    parts = text.split(" ", 3)
                    
                    # SAFE PARSING: Handle potential parsing errors
                    try:
                        result: Dict[str, Any] = {
                            "similarity_score": round(normalized_similarity, 3),
                            "raw_score": round(score, 4),
                            "faiss_index": idx,
                            "component": parts[0] if len(parts) > 0 else "unknown",
                            "latency": float(parts[1]) if len(parts) > 1 and parts[1].replace('.', '', 1).isdigit() else 0.0,
                            "error_rate": float(parts[2]) if len(parts) > 2 and parts[2].replace('.', '', 1).isdigit() else 0.0,
                            "analysis_snippet": parts[3] if len(parts) > 3 else "",
                            "raw_text": text  # Keep original for debugging
                        }
                        results.append(result)
                    except (ValueError, IndexError) as parse_error:
                        logger.warning(f"Error parsing FAISS text '{text[:50]}...': {parse_error}")
                        # Include raw text even if parsing fails
                        results.append({
                            "similarity_score": round(normalized_similarity, 3),
                            "raw_score": round(score, 4),
                            "faiss_index": idx,
                            "component": "unknown",
                            "latency": 0.0,
                            "error_rate": 0.0,
                            "analysis_snippet": text[:100],
                            "raw_text": text,
                            "parse_error": str(parse_error)
                        })
            
            # FIXED: Add explicit type annotation for sort key
            # Sort by similarity (highest first) - explicitly return float
            results.sort(key=lambda x: float(x["similarity_score"]), reverse=True)
            
            logger.info(
                f"Semantic search: '{query_text[:50]}...' found {len(results)} similar incidents "
                f"(threshold={min_similarity})"
            )
            
            return results[:k]  # Return at most k results
            
        except Exception as e:
            logger.error(f"Error in find_similar_incidents: {e}", exc_info=True)
            return []  # Return empty list on error (fail-safe)
    
    def get_index_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the FAISS index
        
        Returns:
            Dictionary with index statistics for monitoring and debugging
        """
        with self._lock:
            try:
                # Get FAISS index statistics
                total_vectors = self.index.ntotal if hasattr(self.index, 'ntotal') else 0
                dimension = self.index.d if hasattr(self.index, 'd') else MemoryConstants.VECTOR_DIM
                
                # Check if index is trained (some FAISS indexes require training)
                is_trained = self.index.is_trained if hasattr(self.index, 'is_trained') else True
                
                # Get queue stats
                pending_writes = self._write_queue.qsize()
                
                # Calculate memory usage estimate (rough)
                memory_bytes = total_vectors * dimension * 4  # float32 = 4 bytes
                
                # FIXED: Use public API to get max workers
                # ProcessPoolExecutor doesn't expose _max_workers publicly
                # We can track it ourselves or use a workaround
                max_workers = 2  # Default from __init__
                if hasattr(self._encoder_pool, '_max_workers'):
                    # This might work in some Python versions, but not guaranteed
                    max_workers = self._encoder_pool._max_workers
                
                return {
                    "total_vectors": total_vectors,
                    "stored_texts": len(self.texts),
                    "pending_writes": pending_writes,
                    "dimension": dimension,
                    "is_trained": is_trained,
                    "estimated_memory_bytes": memory_bytes,
                    "estimated_memory_mb": round(memory_bytes / (1024 * 1024), 2),
                    "index_type": type(self.index).__name__,
                    "is_shutdown": self._shutdown.is_set(),
                    "writer_thread_alive": self._writer_thread.is_alive(),
                    "encoder_pool_workers": max_workers
                }
                
            except Exception as e:
                logger.error(f"Error getting index stats: {e}", exc_info=True)
                return {
                    "error": str(e),
                    "total_vectors": 0,
                    "stored_texts": len(self.texts) if hasattr(self, 'texts') else 0
                }
    
    def find_similar_by_metrics(self, component: str, latency: float, 
                               error_rate: float, k: int = 5) -> List[Dict[str, Any]]:
        """
        Find incidents with similar metrics (synchronous version)
        
        Useful when you need to search without async context
        Creates a synthetic query from metrics
        """
        # Create synthetic query text from metrics
        query_text = f"{component} latency {latency:.0f}ms error {error_rate:.3f}"
        
        # Run async function in sync context
        try:
            # Create new event loop if none exists
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            # Run the async function
            return loop.run_until_complete(
                self.find_similar_incidents(query_text, k)
            )
            
        except Exception as e:
            logger.error(f"Error in find_similar_by_metrics: {e}", exc_info=True)
            return []
    
    def force_save(self) -> None:
        """Force immediate save of pending vectors"""
        logger.info("Forcing FAISS index save...")
        
        # Wait for queue to drain (with timeout)
        timeout = 10.0
        start = datetime.datetime.now()
        
        while not self._write_queue.empty():
            if (datetime.datetime.now() - start).total_seconds() > timeout:
                logger.warning("Force save timeout - queue not empty")
                break
            import time
            time.sleep(0.1)
        
        self._save_atomic()
    
    def shutdown(self) -> None:
        """Graceful shutdown"""
        logger.info("Shutting down FAISS index...")
        self._shutdown.set()
        self.force_save()
        self._writer_thread.join(timeout=5.0)
        self._encoder_pool.shutdown(wait=True)

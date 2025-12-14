"""
Production FAISS Index for v2/v3 compatibility
Extracted from app.py for proper module structure
"""

import threading
import numpy as np
import logging
import datetime
import json
import os
import tempfile
import time
from typing import List, Tuple, Optional
from queue import Queue
from concurrent.futures import ProcessPoolExecutor

from ..config import config
from .constants import MemoryConstants

logger = logging.getLogger(__name__)

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS not available. Search functionality will be limited.")

try:
    import atomicwrites
    ATOMIC_WRITES_AVAILABLE = True
except ImportError:
    ATOMIC_WRITES_AVAILABLE = False
    logger.warning("atomicwrites not available. Using regular file writes.")


class ProductionFAISSIndex:
    """Production-safe FAISS index with single-writer pattern"""
    
    def __init__(self, index, texts: List[str]):
        self.index = index
        self.texts = texts
        self._lock = threading.RLock()
        
        self._shutdown = threading.Event()
        
        # Single writer thread
        self._write_queue: Queue = Queue()
        self._writer_thread = threading.Thread(
            target=self._writer_loop,
            daemon=True,
            name="FAISSWriter"
        )
        self._writer_thread.start()
        
        self._encoder_pool = ProcessPoolExecutor(max_workers=2)
        
        logger.info(
            f"Initialized ProductionFAISSIndex with {len(texts)} vectors"
        )
    
    def add_async(self, vector: np.ndarray, text: str) -> None:
        """Add vector and text asynchronously"""
        self._write_queue.put((vector, text))
        logger.debug(f"Queued vector for indexing: {text[:50]}...")
    
    def _writer_loop(self) -> None:
        """Single writer thread - processes queue in batches"""
        batch = []
        last_save = datetime.datetime.now()
        save_interval = datetime.timedelta(
            seconds=MemoryConstants.FAISS_SAVE_INTERVAL_SECONDS
        )
        
        while not self._shutdown.is_set():
            try:
                import queue
                try:
                    item = self._write_queue.get(timeout=1.0)
                    batch.append(item)
                except queue.Empty:
                    pass
                
                if len(batch) >= MemoryConstants.FAISS_BATCH_SIZE or \
                   (batch and datetime.datetime.now() - last_save > save_interval):
                    self._flush_batch(batch)
                    batch = []
                    
                    if datetime.datetime.now() - last_save > save_interval:
                        self._save_atomic()
                        last_save = datetime.datetime.now()
                        
            except Exception as e:
                logger.error(f"Writer loop error: {e}", exc_info=True)
    
    def _flush_batch(self, batch: List[Tuple[np.ndarray, str]]) -> None:
        """Flush batch to FAISS index"""
        if not batch:
            return
        
        try:
            vectors = np.vstack([v for v, _ in batch])
            texts = [t for _, t in batch]
            
            self.index.add(vectors)
            
            with self._lock:
                self.texts.extend(texts)
            
            logger.info(f"Flushed batch of {len(batch)} vectors to FAISS index")
            
        except Exception as e:
            logger.error(f"Error flushing batch: {e}", exc_info=True)
    
    def _save_atomic(self) -> None:
        """Atomic save with fsync for durability"""
        try:
            if not FAISS_AVAILABLE:
                return
            
            with tempfile.NamedTemporaryFile(
                mode='wb',
                delete=False,
                dir=os.path.dirname(config.index_file),
                prefix='index_',
                suffix='.tmp'
            ) as tmp:
                temp_path = tmp.name
            
            faiss.write_index(self.index, temp_path)
            
            with open(temp_path, 'r+b') as f:
                f.flush()
                os.fsync(f.fileno())
            
            os.replace(temp_path, config.index_file)
            
            with self._lock:
                texts_copy = self.texts.copy()
            
            # Use atomic writes if available
            if ATOMIC_WRITES_AVAILABLE:
                with atomicwrites.atomic_write(
                    config.incident_texts_file,
                    mode='w',
                    overwrite=True
                ) as f:
                    json.dump(texts_copy, f)
            else:
                with open(config.incident_texts_file, 'w') as f:
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
    
    def force_save(self) -> None:
        """Force immediate save of pending vectors"""
        logger.info("Forcing FAISS index save...")
        
        timeout = 10.0
        start = datetime.datetime.now()
        
        while not self._write_queue.empty():
            if (datetime.datetime.now() - start).total_seconds() > timeout:
                logger.warning("Force save timeout - queue not empty")
                break
            time.sleep(0.1)
        
        self._save_atomic()
    
    def shutdown(self) -> None:
        """Graceful shutdown"""
        logger.info("Shutting down FAISS index...")
        self._shutdown.set()
        self.force_save()
        self._writer_thread.join(timeout=5.0)
        self._encoder_pool.shutdown(wait=True)
    
    def add_text(self, text: str, embedding: List[float]) -> int:
        """Add text with embedding to FAISS"""
        vector = np.array(embedding, dtype=np.float32).reshape(1, -1)
        self.add_async(vector, text)
        return len(self.texts) - 1


# Factory function for lazy loading
def create_faiss_index():
    """Create FAISS index with proper error handling"""
    if not FAISS_AVAILABLE:
        logger.warning("FAISS not available. Creating dummy index.")
        return None
    
    try:
        if os.path.exists(config.index_file):
            logger.info(f"Loading existing FAISS index from {config.index_file}")
            index = faiss.read_index(config.index_file)
            
            if index.d != MemoryConstants.VECTOR_DIM:
                logger.warning(
                    f"Index dimension mismatch: {index.d} != {MemoryConstants.VECTOR_DIM}. "
                    f"Creating new index."
                )
                index = faiss.IndexFlatL2(MemoryConstants.VECTOR_DIM)
                incident_texts = []
            else:
                try:
                    with open(config.incident_texts_file, "r") as f:
                        incident_texts = json.load(f)
                    logger.info(f"Loaded {len(incident_texts)} incident texts")
                except FileNotFoundError:
                    logger.warning("Incident texts file not found, starting fresh")
                    incident_texts = []
        else:
            logger.info("Creating new FAISS index")
            index = faiss.IndexFlatL2(MemoryConstants.VECTOR_DIM)
            incident_texts = []
        
        return ProductionFAISSIndex(index, incident_texts)
        
    except Exception as e:
        logger.error(f"Error creating FAISS index: {e}", exc_info=True)
        return None

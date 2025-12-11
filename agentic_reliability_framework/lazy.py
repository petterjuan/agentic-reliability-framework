"""
Lazy loading utilities for ARF
Ensures heavy components only load when needed
"""

import threading
from typing import Any, Callable, TypeVar

T = TypeVar('T')

class LazyLoader:
    """Thread-safe lazy loader for expensive resources"""
    def __init__(self, loader_func: Callable[[], T]):
        self._loader_func = loader_func
        self._lock = threading.RLock()
        self._instance = None
    
    def __call__(self) -> T:
        if self._instance is None:
            with self._lock:
                if self._instance is None:
                    self._instance = self._loader_func()
        return self._instance
    
    def clear(self) -> None:
        """Clear cached instance (for testing)"""
        with self._lock:
            self._instance = None

"""
Engine module with lazy initialization
Separated from app.py for modularity
"""

import logging
from typing import Optional
from .lazy import LazyLoader

logger = logging.getLogger(__name__)

# These will be set by app.py after refactoring
_engine_loader: Optional[LazyLoader] = None
_agents_loader: Optional[LazyLoader] = None

def set_engine_loader(loader: LazyLoader) -> None:
    global _engine_loader
    _engine_loader = loader

def set_agents_loader(loader: LazyLoader) -> None:
    global _agents_loader
    _agents_loader = loader

def get_engine():
    """Lazy access to EnhancedReliabilityEngine"""
    if _engine_loader is None:
        raise RuntimeError("Engine loader not initialized")
    return _engine_loader()

def get_agents():
    """Lazy access to all agents"""
    if _agents_loader is None:
        raise RuntimeError("Agents loader not initialized")
    return _agents_loader()

def initialize_if_needed() -> bool:
    """Force initialization if not already done"""
    if _engine_loader is not None:
        get_engine()
        get_agents()
        return True
    return False

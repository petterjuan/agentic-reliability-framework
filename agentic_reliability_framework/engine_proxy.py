# LAZY INITIALIZATION BACKWARD COMPATIBILITY
# This ensures any direct references to 'enhanced_engine' variable still work
import sys

class EngineProxy:
    """Proxy that lazily loads the real engine"""
    def __getattr__(self, name):
        # Lazy load the engine when any attribute is accessed
        from .lazy_init import get_engine
        engine = get_engine()
        return getattr(engine, name)
    
    def __call__(self, *args, **kwargs):
        # Handle if someone tries to call enhanced_engine directly
        from .lazy_init import get_engine
        engine = get_engine()
        return engine(*args, **kwargs)

# Create proxy instance
enhanced_engine = EngineProxy()

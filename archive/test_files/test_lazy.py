import time
import threading

class HeavyResource:
    def __init__(self, name):
        print(f'Loading {name}...')
        time.sleep(2)  # Simulate heavy load
        self.name = name
        print(f'{name} loaded')
    
    def use(self):
        return f"Using {self.name}"

class LazyLoader:
    def __init__(self, resource_class, *args):
        self.resource_class = resource_class
        self.args = args
        self._instance = None
        self._lock = threading.RLock()
    
    def get(self):
        if self._instance is None:
            with self._lock:
                if self._instance is None:
                    self._instance = self.resource_class(*self.args)
        return self._instance

# Test
loader = LazyLoader(HeavyResource, "DatabaseEngine")
print("Loader created (no heavy load yet)")
print("First call:", loader.get().use())
print("Second call:", loader.get().use())  # Should be instant

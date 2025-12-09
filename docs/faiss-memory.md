# FAISS Vector Memory

**How ARF Remembers and Learns from Past Incidents**

---

## Table of Contents

- [What is FAISS?](#what-is-faiss)
- [Why Vector Memory?](#why-vector-memory)
- [How ARF Uses FAISS](#how-arf-uses-faiss)
- [Vector Embeddings Explained](#vector-embeddings-explained)
- [Similarity Search](#similarity-search)
- [Implementation Details](#implementation-details)
- [Performance Characteristics](#performance-characteristics)
- [Best Practices](#best-practices)

---

## What is FAISS?

**FAISS** (Facebook AI Similarity Search) is a library for efficient similarity search of dense vectors.

### Key Features

- ⚡ **Fast:** Optimized C++ implementation
- 📊 **Scalable:** Handles billions of vectors
- 🎯 **Accurate:** Multiple index types for speed/accuracy trade-offs
- 🔓 **Open Source:** Free, battle-tested by Meta

### Why Facebook Built It

Facebook needed to search billions of images, videos, and text embeddings quickly. Traditional databases are too slow for high-dimensional vector search.

**Result:** FAISS can search millions of vectors in milliseconds.

---

## Why Vector Memory?

### The Problem: Forgetting

**Traditional approach:**
```python
# No memory
def analyze_incident(current_incident):
    # Analyze in isolation
    return analysis
```

**Problems:**
- ❌ No learning from past
- ❌ Repeats same mistakes
- ❌ No pattern recognition
- ❌ Can't identify recurring issues

---

### The Solution: Vector Memory

**ARF approach:**
```python
# With memory
def analyze_incident(current_incident):
    # Convert to vector
    vector = vectorize(current_incident)
    
    # Search similar past incidents
    similar = faiss_index.search(vector, k=5)
    
    # Learn from history
    return analysis_with_context(current_incident, similar)
```

**Benefits:**
- ✅ Learns from history
- ✅ Recognizes patterns
- ✅ Faster diagnosis
- ✅ Better predictions

---

## How ARF Uses FAISS

### High-Level Flow

```
New Incident
     ↓
Vectorize (SentenceTransformer)
     ↓
384-dimensional vector
     ↓
Search FAISS Index
     ↓
Top 5 Similar Past Incidents
     ↓
Use for Context
```

---

### Step-by-Step Example

#### **1. Incident Arrives**

```python
incident = ReliabilityEvent(
    component="api-service",
    latency_p99=450.0,
    error_rate=0.15,
    throughput=1200.0,
    cpu_util=0.85,
    memory_util=0.72
)
```

---

#### **2. Create Text Description**

```python
text = f"""
Component: {incident.component}
Latency P99: {incident.latency_p99}ms
Error Rate: {incident.error_rate}
CPU: {incident.cpu_util}
Memory: {incident.memory_util}
"""
```

---

#### **3. Vectorize**

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
vector = model.encode(text)

# Result: numpy array of shape (384,)
print(vector.shape)  # (384,)
```

---

#### **4. Search Similar**

```python
# Search FAISS index
distances, indices = index.search(vector.reshape(1, -1), k=5)

# Get similar incidents
similar_incidents = [incident_texts[i] for i in indices[0]]
```

---

#### **5. Use Context**

```python
analysis = {
    "current_incident": incident,
    "similar_past_incidents": similar_incidents,
    "patterns_identified": extract_patterns(similar_incidents),
    "recommended_actions": infer_from_history(similar_incidents)
}
```

---

## Vector Embeddings Explained

### What is a Vector Embedding?

**Simple explanation:**

A **vector embedding** is a list of numbers that represents meaning.

**Example:**

```python
# Text
text1 = "High latency on payment service"
text2 = "Payment API is slow"
text3 = "Weather is sunny today"

# Vectors (simplified to 3 dimensions for illustration)
vector1 = [0.8, 0.9, 0.1]  # High latency, payment, API
vector2 = [0.7, 0.8, 0.2]  # Slow, payment, API
vector3 = [0.1, 0.1, 0.9]  # Weather, sunny, today

# Similar concepts → Similar vectors
similarity(vector1, vector2) = 0.95  # Very similar!
similarity(vector1, vector3) = 0.15  # Not similar
```

---

### Why 384 Dimensions?

ARF uses **all-MiniLM-L6-v2** model, which produces **384-dimensional vectors**.

**Each dimension captures different semantic features:**
- Dimension 0: "service type"
- Dimension 1: "severity"
- Dimension 2: "resource type"
- ... (381 more dimensions)

**Why so many?**
- More dimensions = more nuanced meaning
- Can capture subtle differences
- Better accuracy

---

### Visualization (Simplified)

**2D Projection:**

```
High CPU  │       ×payment-slow
          │     ×api-latency
          │   ×db-timeout
          │
          │
Low CPU   │ ×cache-miss
          └────────────────
        Low Latency    High Latency
```

**Clusters emerge naturally!**

---

## Similarity Search

### How Similarity is Measured

**Cosine Similarity:**

```python
def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    magnitude = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    return dot_product / magnitude

# Example
vec1 = [0.8, 0.9, 0.1]
vec2 = [0.7, 0.8, 0.2]

similarity = cosine_similarity(vec1, vec2)
print(similarity)  # 0.95 (very similar!)
```

**Interpretation:**
- 1.0 = Identical
- 0.8-0.9 = Very similar
- 0.5-0.7 = Somewhat similar
- <0.5 = Not similar

---

### FAISS IndexFlatL2

ARF uses **IndexFlatL2** (L2 distance):

```python
import faiss

# Create index
dimension = 384
index = faiss.IndexFlatL2(dimension)

# Add vectors
index.add(vectors)

# Search
distances, indices = index.search(query_vector, k=5)
```

**Why L2?**
- Exact search (no approximation)
- Simple & reliable
- Fast for <1M vectors

---

### Search Example

```python
# Query
query = "Database connection pool exhausted"
query_vector = model.encode(query)

# Search top 5 similar
k = 5
distances, indices = index.search(query_vector.reshape(1, -1), k)

# Results
for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
    similarity_score = 1 / (1 + dist)  # Convert distance to similarity
    print(f"{i+1}. {incident_texts[idx]} (similarity: {similarity_score:.2f})")
```

**Output:**
```
1. DB pool exhausted on auth-service (similarity: 0.95)
2. Connection timeout to PostgreSQL (similarity: 0.87)
3. High latency due to DB overload (similarity: 0.82)
4. Database slow query detected (similarity: 0.76)
5. Redis connection pool full (similarity: 0.68)
```

---

## Implementation Details

### ProductionFAISSIndex

ARF's thread-safe FAISS wrapper:

```python
class ProductionFAISSIndex:
    def __init__(self, index: faiss.Index, texts: List[str]):
        self.index = index
        self.texts = texts
        self._writer_lock = threading.RLock()  # Single-writer
        self._encoder_pool = ThreadPoolExecutor(max_workers=4)
        self._pending_vectors = []
        self._last_save = time.time()
    
    async def add_async(self, vector: np.ndarray, text: str):
        """Add vector asynchronously with batching"""
        self._pending_vectors.append((vector, text))
        
        if len(self._pending_vectors) >= BATCH_SIZE:
            await self._flush_batch()
    
    def _flush_batch(self):
        """Write batch to FAISS index"""
        with self._writer_lock:
            vectors = np.array([v for v, _ in self._pending_vectors])
            texts = [t for _, t in self._pending_vectors]
            
            self.index.add(vectors)
            self.texts.extend(texts)
            
            self._pending_vectors.clear()
            self._save_if_needed()
```

---

### Key Design Patterns

#### **1. Lazy Loading**

```python
# Model only loads when first needed
model = None

def get_model():
    global model
    if model is None:
        model = SentenceTransformer('all-MiniLM-L6-v2')
    return model
```

**Why?**
- Faster startup (8.6s → 7.9s)
- No overhead if not used

---

#### **2. Batch Processing**

```python
# Don't write every vector immediately
if len(pending_vectors) >= BATCH_SIZE:
    flush_to_disk()
```

**Why?**
- Reduces I/O operations
- Better performance
- Less disk wear

---

#### **3. Thread Safety**

```python
# Single-writer pattern
with self._writer_lock:
    self.index.add(vectors)
    self.texts.append(text)
```

**Why?**
- Prevents race conditions
- Ensures data consistency
- Safe for concurrent reads

---

#### **4. Atomic Saves**

```python
from atomicwrites import atomic_write

# Write to temp file, then rename (atomic)
with atomic_write(filepath, overwrite=True) as f:
    faiss.write_index(self.index, faiss.IOWriter(f))
```

**Why?**
- Never corrupts data
- Safe even if crashes mid-write

---

## Performance Characteristics

### Time Complexity

| Operation | Complexity | ARF Performance |
|-----------|-----------|----------------|
| Add vector | O(d) | <1ms |
| Search (k results) | O(n·d) | <50ms for 1000 vectors |
| Save index | O(n·d) | ~100ms per 1000 vectors |

Where:
- n = number of vectors
- d = dimension (384)
- k = results requested

---

### Memory Usage

```python
# Memory per vector
vector_size = 384 dimensions × 4 bytes/float = 1,536 bytes
text_size = ~500 bytes average

total_per_incident = 1,536 + 500 ≈ 2KB

# For 1000 incidents
memory = 1000 × 2KB = 2MB

# For 10,000 incidents
memory = 10,000 × 2KB = 20MB
```

**Very efficient!**

---

### Disk Usage

```python
# FAISS index file
index_size = num_vectors × 1.5KB

# Text file (JSON)
texts_size = num_vectors × 500 bytes

# Total
total = index_size + texts_size

# Example for 10,000 incidents
total = (10,000 × 1.5KB) + (10,000 × 0.5KB)
      = 15MB + 5MB = 20MB
```

---

## Best Practices

### 1. Regular Persistence

```python
# Save periodically
if time.time() - last_save > SAVE_INTERVAL:
    save_index()
```

**Why?**
- Prevents data loss
- Enables recovery

---

### 2. Bounded Memory

```python
# Limit index size
MAX_VECTORS = 10000

if index.ntotal >= MAX_VECTORS:
    # Evict oldest or least relevant
    prune_index()
```

**Why?**
- Prevents unbounded growth
- Maintains performance

---

### 3. Incremental Updates

```python
# Don't rebuild entire index
# Add new vectors incrementally
index.add(new_vector)
```

**Why?**
- O(1) vs O(n)
- No downtime

---

### 4. Backup Strategy

```python
# Daily backups
os.system("cp faiss_index.bin faiss_index.bin.backup")
```

**Why?**
- Recover from corruption
- Historical snapshots

---

## Advanced Topics

### Index Types

ARF uses **IndexFlatL2** (exact search), but FAISS offers others:

| Index Type | Speed | Accuracy | Use Case |
|------------|-------|----------|----------|
| IndexFlatL2 | Slow | 100% | <1M vectors |
| IndexIVFFlat | Fast | ~95% | 1M-100M vectors |
| IndexHNSW | Fastest | ~98% | Any size |

**ARF's choice:**
- **IndexFlatL2** = Simple, accurate, fast enough for our use case

---

### GPU Acceleration

FAISS supports GPU:

```python
# CPU version (current)
index = faiss.IndexFlatL2(384)

# GPU version (optional)
gpu_index = faiss.index_cpu_to_gpu(
    faiss.StandardGpuResources(), 
    0,  # GPU ID
    index
)
```

**When to use?**
- >1M vectors
- Need <10ms search latency

---

### Approximate Search

For huge datasets (>10M vectors):

```python
# Train IVF index
quantizer = faiss.IndexFlatL2(dimension)
index = faiss.IndexIVFFlat(quantizer, dimension, n_clusters)

# Train on sample data
index.train(sample_vectors)

# Add vectors
index.add(vectors)

# Search (approximate, but fast)
index.nprobe = 10  # accuracy/speed trade-off
distances, indices = index.search(query, k)
```

---

## Troubleshooting

### Issue: Slow Search

**Symptom:** Search takes >1 second

**Diagnosis:**
```python
print(f"Vectors in index: {index.ntotal}")
```

**Solution:**
- If >100k vectors: Switch to IndexIVFFlat
- If >1M vectors: Use GPU or HNSW

---

### Issue: Index Corruption

**Symptom:** Can't load index

**Diagnosis:**
```python
try:
    index = faiss.read_index("faiss_index.bin")
except Exception as e:
    print(f"Corrupted: {e}")
```

**Solution:**
- Restore from backup
- Rebuild from event store

---

### Issue: High Memory Usage

**Symptom:** Memory grows unbounded

**Diagnosis:**
```python
import psutil
print(f"Memory: {psutil.Process().memory_info().rss / 1024**2} MB")
```

**Solution:**
- Implement eviction policy
- Set MAX_VECTORS limit

---

## Example: Building Your Own

```python
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# 1. Create model
model = SentenceTransformer('all-MiniLM-L6-v2')

# 2. Create FAISS index
dimension = 384
index = faiss.IndexFlatL2(dimension)

# 3. Add incidents
incidents = [
    "High latency on payment service",
    "Database connection timeout",
    "Memory leak in API server"
]

for text in incidents:
    vector = model.encode(text)
    index.add(vector.reshape(1, -1))

# 4. Search
query = "Payment processing is slow"
query_vector = model.encode(query)

distances, indices = index.search(query_vector.reshape(1, -1), k=2)

# 5. Results
for idx in indices[0]:
    print(f"Similar: {incidents[idx]}")
```

**Output:**
```
Similar: High latency on payment service
Similar: Database connection timeout
```

---

## Further Reading

- [Architecture Overview](./architecture.md) - How FAISS fits in
- [Multi-Agent System](./multi-agent.md) - How agents use memory
- [API Reference](./api.md) - FAISS APIs

---

**Built by [LGCY Labs](https://lgcylabs.vercel.app/) | [Back to README](../README.md)**

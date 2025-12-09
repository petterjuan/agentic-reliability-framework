# Architecture Overview

**Agentic Reliability Framework (ARF) - System Design Document**

---

## Table of Contents

- [System Overview](#system-overview)
- [Multi-Agent Architecture](#multi-agent-architecture)
- [Component Interactions](#component-interactions)
- [Data Flow](#data-flow)
- [Memory & Persistence](#memory--persistence)
- [Self-Healing Mechanisms](#self-healing-mechanisms)
- [Scalability Patterns](#scalability-patterns)
- [Design Decisions](#design-decisions)

---

## System Overview

ARF is a **production-grade multi-agent AI system** designed to monitor, diagnose, and self-heal production infrastructure failures before they impact revenue.

### Core Principles

1. **Agent Specialization** - Each agent has a focused responsibility
2. **Coordinated Reasoning** - Agents work together through orchestration
3. **Memory-Driven Learning** - FAISS vector memory enables pattern recognition
4. **Policy-Based Automation** - Self-healing through declarative policies
5. **Business-Centric** - Every decision mapped to revenue impact

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Gradio UI Layer                      │
│              (User Interaction & Metrics)               │
└─────────────┬───────────────────────────────────────────┘
              │
┌─────────────▼───────────────────────────────────────────┐
│           Orchestration Manager                         │
│     (Coordinates Multi-Agent Analysis)                  │
└──┬──────────┬──────────┬──────────────────────────────┘
   │          │          │
   ▼          ▼          ▼
┌────────┐ ┌────────┐ ┌────────┐
│Detective│ │Diagnos-│ │Predict-│
│ Agent  │ │tician  │ │ive     │
│        │ │ Agent  │ │ Agent  │
└───┬────┘ └───┬────┘ └───┬────┘
    │          │          │
    └──────────┼──────────┘
               │
        ┌──────▼───────┐
        │ Policy Engine│
        │  (Healing)   │
        └──────┬───────┘
               │
    ┌──────────┴──────────┐
    │                     │
┌───▼────────┐    ┌──────▼─────────┐
│FAISS Vector│    │Event Store     │
│Memory      │    │(Thread-Safe)   │
└────────────┘    └────────────────┘
```

---

## Multi-Agent Architecture

### 1. Detective Agent 🕵️

**Responsibility:** Anomaly detection and pattern recognition

**Capabilities:**
- Real-time statistical anomaly scoring
- Adaptive threshold learning
- Historical pattern matching via FAISS
- Multi-metric correlation analysis

**Algorithm:**
```python
anomaly_score = calculate_composite_score([
    latency_score,      # Based on percentile + thresholds
    error_rate_score,   # Binary + weighted by severity
    throughput_score,   # Rate of change analysis
    resource_score      # CPU/memory utilization
])

if anomaly_score > adaptive_threshold:
    classify_severity()  # CRITICAL, HIGH, MEDIUM, LOW
    recall_similar_incidents()  # FAISS similarity search
```

**Output:**
```json
{
  "specialization": "detective",
  "confidence": 0.89,
  "findings": {
    "anomaly_score": 0.89,
    "severity_tier": "HIGH",
    "primary_metrics_affected": ["latency_p99", "error_rate"]
  },
  "recommendations": [...]
}
```

---

### 2. Diagnostician Agent 🔍

**Responsibility:** Root cause analysis and evidence correlation

**Capabilities:**
- Causal reasoning from observed symptoms
- Dependency tree analysis
- Evidence-based hypothesis generation
- Investigation prioritization

**Algorithm:**
```python
potential_causes = [
    analyze_resource_exhaustion(),
    check_dependency_failures(),
    evaluate_config_changes(),
    detect_capacity_limits(),
    assess_external_factors()
]

evidence_patterns = correlate_with_history()
priority = rank_by_likelihood_and_impact()
```

**Output:**
```json
{
  "specialization": "diagnostician",
  "confidence": 0.75,
  "findings": {
    "likely_root_causes": [
      "Database connection pool exhaustion",
      "Cascading failure from auth-service"
    ],
    "evidence_patterns": [...],
    "investigation_priority": "CRITICAL"
  }
}
```

---

### 3. Predictive Agent 🔮

**Responsibility:** Forecasting future failures and resource needs

**Capabilities:**
- Time-series trend analysis
- Risk-level classification
- Time-to-failure estimation
- Resource utilization forecasting

**Algorithm:**
```python
# Simple linear regression + exponential smoothing
def forecast(component, metric, lookahead_minutes=15):
    historical_data = get_telemetry_window(component)
    
    # Forecast value
    predicted_value = linear_regression(historical_data, lookahead_minutes)
    
    # Smoothing
    smoothed = exponential_smoothing(predicted_value, alpha=0.3)
    
    # Risk assessment
    risk = classify_risk(smoothed, thresholds)
    
    return {
        "predicted_value": smoothed,
        "trend": calculate_trend(historical_data),
        "risk_level": risk,
        "time_to_critical": estimate_time_to_critical(smoothed, trend)
    }
```

**Output:**
```json
{
  "specialization": "predictive",
  "confidence": 0.82,
  "findings": {
    "forecast_latency_p99": 285.5,
    "trend": "increasing",
    "risk_level": "HIGH",
    "time_to_critical_minutes": 12
  }
}
```

---

## Component Interactions

### Event Processing Flow

1. **Event Submission**
   ```
   User/System → Reliability Event → Validation
   ```

2. **Multi-Agent Analysis**
   ```
   Event → Orchestration Manager → [Detective, Diagnostician, Predictive]
   ```

3. **Agent Coordination**
   ```python
   async def analyze_event(event):
       # Parallel execution
       results = await asyncio.gather(
           detective.analyze(event),
           diagnostician.analyze(event),
           predictive.analyze(event)
       )
       
       # Aggregate insights
       return aggregate_findings(results)
   ```

4. **Policy Evaluation**
   ```
   Agent Results → Policy Engine → Matching Policies → Recovery Actions
   ```

5. **Persistence**
   ```
   Event → Event Store (thread-safe)
   Event Vector → FAISS Index (async batch)
   ```

---

## Data Flow

### Inbound Data Path

```
External System
    ↓
Reliability Event
    ↓
Input Validation
    ↓
Event Store (append-only)
    ↓
Orchestration Manager
    ↓
[Detective | Diagnostician | Predictive]
    ↓
Policy Engine
    ↓
Recovery Actions
```

### Memory Path

```
Event
    ↓
Vectorization (SentenceTransformer)
    ↓
FAISS Index (lazy-loaded)
    ↓
Similarity Search (on future events)
    ↓
Historical Context
```

---

## Memory & Persistence

### FAISS Vector Memory

**Purpose:** Enable semantic similarity search for incident recall

**Implementation:**
```python
class ProductionFAISSIndex:
    def __init__(self, index, texts):
        self.index = index           # FAISS IndexFlatL2
        self.texts = texts           # Original incident descriptions
        self._encoder_pool = ...     # Thread pool for encoding
        self._writer_lock = RLock()  # Single-writer pattern
```

**Key Features:**
- **Lazy-loaded model** (10% faster startup)
- **Thread-safe writes** (single-writer with RLock)
- **Atomic saves** (atomicwrites library)
- **Batch processing** (configurable batch size)

**Vector Generation:**
```python
def vectorize_event(event):
    text = f"""
    Component: {event.component}
    Latency: {event.latency_p99}ms
    Error Rate: {event.error_rate}
    Severity: {event.severity}
    """
    
    vector = model.encode(text)  # 384-dimensional
    return vector
```

---

### Event Store

**Purpose:** Thread-safe, bounded event history

**Implementation:**
```python
class ThreadSafeEventStore:
    def __init__(self, max_size=1000):
        self._events = deque(maxlen=max_size)
        self._lock = threading.Lock()
    
    def add(self, event):
        with self._lock:
            self._events.append(event)  # Auto-evicts oldest
```

**Characteristics:**
- **Thread-safe** (threading.Lock)
- **Bounded memory** (deque with maxlen)
- **Fast access** (O(1) append, O(n) search)

---

## Self-Healing Mechanisms

### Policy Engine

**Architecture:**
```python
class PolicyEngine:
    def __init__(self):
        self.policies = load_healing_policies()
        self._evaluation_lock = RLock()
        self._cooldown_tracker = {}
        self._rate_limiter = RateLimiter()
```

**Policy Structure:**
```python
@dataclass(frozen=True)
class HealingPolicy:
    name: str
    conditions: List[PolicyCondition]  # All must match
    actions: List[str]                 # Recovery steps
    priority: int                      # 1-10 (10 = highest)
    cooldown_seconds: int              # Prevent thrashing
    enabled: bool = True
```

**Evaluation Flow:**
```python
def evaluate(event):
    matching_policies = []
    
    for policy in sorted(policies, key=lambda p: -p.priority):
        if policy.enabled and not in_cooldown(policy):
            if all(condition_matches(c, event) for c in policy.conditions):
                matching_policies.append(policy)
                apply_cooldown(policy)
    
    return matching_policies
```

---

### Recovery Actions

**Supported Actions:**
- `restart-service` - Graceful service restart
- `scale-up` - Increase capacity
- `enable-circuit-breaker` - Prevent cascading failures
- `fallback-mode` - Degrade gracefully
- `alert-oncall` - Human escalation
- `rollback-deployment` - Revert changes

**Rate Limiting:**
```python
class RateLimiter:
    def __init__(self, max_per_minute=60, max_per_hour=500):
        self.requests = deque()  # Sliding window
    
    def allow_request(self):
        now = time.time()
        # Remove old requests
        while self.requests and now - self.requests[0] > 3600:
            self.requests.popleft()
        
        # Check limits
        recent = sum(1 for r in self.requests if now - r < 60)
        if recent >= self.max_per_minute:
            return False
        
        self.requests.append(now)
        return True
```

---

## Scalability Patterns

### Horizontal Scaling

**Stateless Design:**
- Each ARF instance is independent
- FAISS index can be shared (read-only replicas)
- Event store can use external DB (Redis, PostgreSQL)

**Load Balancing:**
```
┌──────────┐
│  LB      │
└─┬─┬─┬─┬──┘
  │ │ │ │
  ▼ ▼ ▼ ▼
[ARF][ARF][ARF][ARF]
  │ │ │ │
  └─┴─┴─┴─► Shared FAISS (read-only)
  │ │ │ │
  └─┴─┴─┴─► Shared Event Store
```

---

### Vertical Scaling

**Resource Allocation:**
- **CPU:** Agent analysis (parallel async)
- **Memory:** FAISS index (384-dim × events)
- **Disk:** Event store + FAISS persistence

**Optimization:**
- Lazy-loaded ML models
- Bounded memory structures
- Async I/O operations

---

## Design Decisions

### Why Multi-Agent?

**Single-Agent Alternative:**
```python
def analyze(event):
    # One agent does everything
    anomaly = detect_anomaly()
    diagnosis = find_root_cause()
    forecast = predict_future()
    return combine(anomaly, diagnosis, forecast)
```

**Problems:**
- ❌ Difficult to test individual capabilities
- ❌ Hard to swap implementations
- ❌ Poor separation of concerns
- ❌ Difficult to optimize independently

**Multi-Agent Benefits:**
- ✅ **Testability:** Each agent tested in isolation
- ✅ **Maintainability:** Clear boundaries
- ✅ **Extensibility:** Add new agents easily
- ✅ **Performance:** Parallel execution

---

### Why FAISS Over Database?

**Database Alternative:**
```sql
SELECT * FROM incidents 
WHERE similarity(embedding, ?) > threshold
ORDER BY similarity DESC
LIMIT 5;
```

**Problems:**
- ❌ Slower for high-dimensional vectors
- ❌ Requires PostgreSQL + pgvector
- ❌ More infrastructure complexity
- ❌ Harder to deploy

**FAISS Benefits:**
- ✅ **Speed:** Optimized for vector similarity
- ✅ **Simplicity:** File-based persistence
- ✅ **Portability:** No external dependencies
- ✅ **Performance:** GPU-accelerated (optional)

---

### Why Lazy-Loading?

**Eager Loading:**
```python
# On import
model = SentenceTransformer(...)  # 0.7s load time
```

**Impact:**
- ❌ Slows down every import (tests, CLI, etc.)
- ❌ Memory overhead even if not used
- ❌ Startup time compounds in CI/CD

**Lazy Loading Benefits:**
- ✅ **Faster startup:** 8.6s → 7.9s (10% improvement)
- ✅ **Efficient tests:** Model only loads when needed
- ✅ **Lower memory:** No overhead for non-ML operations

---

### Why Pydantic Models?

**Alternative:** Plain dictionaries

**Problems:**
- ❌ No type safety
- ❌ Manual validation
- ❌ Runtime errors

**Pydantic Benefits:**
- ✅ **Type safety:** Catch errors at validation
- ✅ **Automatic validation:** Input sanitization
- ✅ **Immutability:** Frozen dataclasses
- ✅ **Serialization:** Easy JSON export

---

## Performance Characteristics

### Latency Profile

| Operation | Latency | Notes |
|-----------|---------|-------|
| Event submission | <10ms | Validation + storage |
| Agent analysis | 50-200ms | Parallel execution |
| FAISS similarity search | <50ms | 1000+ vectors |
| Policy evaluation | <10ms | Rule matching |
| Total end-to-end | <300ms | User-facing operations |

---

### Memory Footprint

| Component | Memory | Configurable |
|-----------|--------|-------------|
| FAISS index | ~1.5MB per 1000 events | ✅ Max events |
| SentenceTransformer | ~90MB | ❌ Model size |
| Event store | ~1KB per event | ✅ Max size |
| Policy engine | <1MB | ❌ Fixed |

---

## Future Architecture Enhancements

### Planned Improvements

1. **Distributed FAISS**
   - Shared index across multiple instances
   - Redis-backed index updates

2. **Agent Marketplace**
   - Plugin system for custom agents
   - Community-contributed specializations

3. **Advanced Orchestration**
   - Dynamic agent selection
   - Confidence-weighted voting

4. **Real-Time Streaming**
   - Kafka/Pulsar integration
   - Continuous event processing

5. **Multi-Tenancy**
   - Isolated environments per customer
   - Resource quotas & rate limits

---

## References

- [FAISS Documentation](https://github.com/facebookresearch/faiss)
- [SentenceTransformers](https://www.sbert.net/)
- [Multi-Agent Systems](https://en.wikipedia.org/wiki/Multi-agent_system)
- [Circuit Breaker Pattern](https://martinfowler.com/bliki/CircuitBreaker.html)

---

**Built by [LGCY Labs](https://lgcylabs.vercel.app/) | [Back to README](../README.md)**

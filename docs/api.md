# API Reference

**Agentic Reliability Framework (ARF) - Complete API Documentation**

---

## Table of Contents

- [Core Data Models](#core-data-models)
- [Agent APIs](#agent-apis)
- [Policy Engine API](#policy-engine-api)
- [Memory APIs](#memory-apis)
- [Business Metrics API](#business-metrics-api)
- [Utility Functions](#utility-functions)
- [Configuration](#configuration)
- [Error Handling](#error-handling)

---

## Core Data Models

### ReliabilityEvent

**Primary data model for system events**

```python
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum

class Severity(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"

@dataclass(frozen=True)
class ReliabilityEvent:
    component: str              # Service/component name
    latency_p99: float         # 99th percentile latency (ms)
    error_rate: float          # Error rate (0.0-1.0)
    throughput: float          # Requests per second
    cpu_util: float            # CPU utilization (0.0-1.0)
    memory_util: float         # Memory utilization (0.0-1.0)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    severity: Severity = Severity.MEDIUM
    upstream_deps: List[str] = field(default_factory=list)
    
    @property
    def fingerprint(self) -> str:
        """Generate SHA-256 fingerprint for deduplication"""
        data = f"{self.component}:{self.timestamp.isoformat()}"
        return hashlib.sha256(data.encode()).hexdigest()
```

**Validation Rules:**
- `component`: 1-255 chars, lowercase alphanumeric + hyphens
- `latency_p99`: 0-10,000 ms
- `error_rate`: 0.0-1.0 (0-100%)
- `throughput`: ≥ 0 req/sec
- `cpu_util`: 0.0-1.0 (0-100%)
- `memory_util`: 0.0-1.0 (0-100%)

**Example:**
```python
event = ReliabilityEvent(
    component="api-service",
    latency_p99=450.5,
    error_rate=0.15,
    throughput=1250.0,
    cpu_util=0.85,
    memory_util=0.72,
    severity=Severity.HIGH,
    upstream_deps=["auth-service", "database"]
)
```

---

### PolicyCondition

**Condition for policy matching**

```python
@dataclass(frozen=True)
class PolicyCondition:
    metric: str         # latency_p99, error_rate, cpu_util, etc.
    operator: str       # gt, lt, gte, lte, eq
    threshold: float    # Comparison value
    
    def matches(self, event: ReliabilityEvent) -> bool:
        """Check if condition matches event"""
        value = getattr(event, self.metric, None)
        if value is None:
            return False
        
        if self.operator == "gt":
            return value > self.threshold
        elif self.operator == "gte":
            return value >= self.threshold
        elif self.operator == "lt":
            return value < self.threshold
        elif self.operator == "lte":
            return value <= self.threshold
        elif self.operator == "eq":
            return value == self.threshold
        
        return False
```

**Supported Operators:**
- `gt` - Greater than
- `gte` - Greater than or equal
- `lt` - Less than
- `lte` - Less than or equal
- `eq` - Equal

**Example:**
```python
condition = PolicyCondition(
    metric="latency_p99",
    operator="gt",
    threshold=300.0
)
```

---

### HealingPolicy

**Self-healing policy definition**

```python
@dataclass(frozen=True)
class HealingPolicy:
    name: str
    conditions: List[PolicyCondition]
    actions: List[str]
    priority: int = 5              # 1-10 (10 = highest)
    cooldown_seconds: int = 300    # Prevent re-execution
    enabled: bool = True
```

**Example:**
```python
policy = HealingPolicy(
    name="High Latency Recovery",
    conditions=[
        PolicyCondition("latency_p99", "gt", 300.0),
        PolicyCondition("error_rate", "lt", 0.5)
    ],
    actions=["restart-service", "scale-up"],
    priority=8,
    cooldown_seconds=600
)
```

---

## Agent APIs

### BaseAgent

**Abstract base class for all agents**

```python
class BaseAgent:
    def __init__(self, specialization: AgentSpecialization):
        self.specialization = specialization
        self.stats = {
            'processed_events': 0,
            'successful_analyses': 0,
            'average_confidence': 0.0
        }
    
    async def analyze(self, event: ReliabilityEvent) -> Dict[str, Any]:
        """Must be implemented by subclasses"""
        raise NotImplementedError
```

---

### AnomalyDetectionAgent (Detective)

**Detects anomalies in system metrics**

```python
class AnomalyDetectionAgent(BaseAgent):
    async def analyze(self, event: ReliabilityEvent) -> Dict[str, Any]:
        """
        Performs comprehensive anomaly analysis
        
        Args:
            event: ReliabilityEvent to analyze
        
        Returns:
            Dict containing:
                - specialization: "detective"
                - confidence: float (0.0-1.0)
                - findings: Dict with analysis results
                - recommendations: List[str]
        """
```

**Response Structure:**
```json
{
  "specialization": "detective",
  "confidence": 0.89,
  "findings": {
    "anomaly_score": 0.89,
    "severity_tier": "HIGH",
    "primary_metrics_affected": ["latency_p99", "error_rate"],
    "similar_incidents": [
      {
        "incident": "api-service spike on 2025-12-08",
        "similarity": 0.92
      }
    ]
  },
  "recommendations": [
    "Investigate database connection pool",
    "Check recent deployments",
    "Review error logs for patterns"
  ]
}
```

**Methods:**

#### `_calculate_anomaly_score(event: ReliabilityEvent) -> float`

```python
def _calculate_anomaly_score(self, event: ReliabilityEvent) -> float:
    """
    Calculate composite anomaly score
    
    Returns: float (0.0-1.0) where 1.0 = most anomalous
    """
```

#### `_classify_severity(score: float) -> str`

```python
def _classify_severity(self, score: float) -> str:
    """
    Map anomaly score to severity tier
    
    Returns: "CRITICAL", "HIGH", "MEDIUM", or "LOW"
    """
```

---

### RootCauseAgent (Diagnostician)

**Performs root cause analysis**

```python
class RootCauseAgent(BaseAgent):
    async def analyze(self, event: ReliabilityEvent) -> Dict[str, Any]:
        """
        Performs root cause analysis
        
        Returns:
            Dict containing:
                - likely_root_causes: List[str]
                - evidence_patterns: List[str]
                - investigation_priority: str
        """
```

**Response Structure:**
```json
{
  "specialization": "diagnostician",
  "confidence": 0.75,
  "findings": {
    "likely_root_causes": [
      "Database connection pool exhaustion",
      "Cascading failure from auth-service",
      "Memory leak in request handler"
    ],
    "evidence_patterns": [
      "CPU steady, memory climbing",
      "Error rate correlates with latency",
      "Throughput drop matches deployment"
    ],
    "investigation_priority": "CRITICAL"
  },
  "recommendations": [...]
}
```

---

### PredictiveAgent (Forecaster)

**Forecasts future failures**

```python
class PredictiveAgent(BaseAgent):
    def __init__(self, engine: SimplePredictiveEngine):
        super().__init__(AgentSpecialization.PREDICTIVE)
        self.engine = engine
    
    async def analyze(self, event: ReliabilityEvent) -> Dict[str, Any]:
        """
        Performs predictive analysis
        
        Returns:
            Dict containing:
                - forecast_latency_p99: float
                - trend: "increasing", "stable", "decreasing"
                - risk_level: "HIGH", "MEDIUM", "LOW"
                - time_to_critical_minutes: float
        """
```

**Response Structure:**
```json
{
  "specialization": "predictive",
  "confidence": 0.82,
  "findings": {
    "forecast_latency_p99": 385.5,
    "forecast_error_rate": 0.18,
    "trend": "increasing",
    "risk_level": "HIGH",
    "time_to_critical_minutes": 12.5
  },
  "recommendations": [
    "Scale up within 10 minutes",
    "Monitor closely for next 30 minutes"
  ]
}
```

---

## Policy Engine API

### PolicyEngine

**Evaluates and executes healing policies**

```python
class PolicyEngine:
    def __init__(self, policies: List[HealingPolicy] = None):
        """
        Initialize policy engine
        
        Args:
            policies: List of healing policies (default: load from config)
        """
    
    def evaluate(self, event: ReliabilityEvent) -> List[HealingPolicy]:
        """
        Evaluate event against all policies
        
        Args:
            event: Event to evaluate
        
        Returns:
            List of matching policies (sorted by priority)
        
        Respects:
            - Policy cooldowns
            - Rate limiting
            - Priority ordering
        """
```

**Example:**
```python
engine = PolicyEngine()
matching_policies = engine.evaluate(event)

for policy in matching_policies:
    print(f"Executing: {policy.name}")
    for action in policy.actions:
        execute_action(action, event.component)
```

---

### PolicyEngine Methods

#### `add_policy(policy: HealingPolicy) -> None`

```python
def add_policy(self, policy: HealingPolicy) -> None:
    """Add new policy at runtime"""
```

#### `remove_policy(name: str) -> None`

```python
def remove_policy(self, name: str) -> None:
    """Remove policy by name"""
```

#### `get_policy_stats() -> Dict[str, Any]`

```python
def get_policy_stats(self) -> Dict[str, Any]:
    """
    Get execution statistics
    
    Returns:
        {
            "total_evaluations": int,
            "policies_triggered": int,
            "avg_execution_time_ms": float,
            "cooldowns_active": int
        }
    """
```

---

## Memory APIs

### ProductionFAISSIndex

**FAISS-based vector memory for incident recall**

```python
class ProductionFAISSIndex:
    def __init__(self, index: faiss.Index, texts: List[str]):
        """
        Initialize FAISS index
        
        Args:
            index: FAISS index instance
            texts: Corresponding text descriptions
        """
    
    async def add_async(self, vector: np.ndarray, text: str) -> None:
        """
        Add vector to index (async, batched)
        
        Args:
            vector: 384-dimensional embedding
            text: Original text description
        """
    
    def search(self, query_vector: np.ndarray, k: int = 5) -> List[Tuple[str, float]]:
        """
        Search for similar incidents
        
        Args:
            query_vector: Query embedding
            k: Number of results
        
        Returns:
            List of (text, similarity_score) tuples
        """
```

**Example:**
```python
# Vectorize event
vector = model.encode(event_description)

# Search similar incidents
similar = index.search(vector, k=5)

for text, score in similar:
    print(f"Similarity: {score:.2f} - {text}")
```

---

### ThreadSafeEventStore

**Thread-safe event storage**

```python
class ThreadSafeEventStore:
    def __init__(self, max_size: int = 1000):
        """
        Initialize event store
        
        Args:
            max_size: Maximum events to retain (FIFO eviction)
        """
    
    def add(self, event: ReliabilityEvent) -> None:
        """Add event (thread-safe)"""
    
    def get_recent(self, n: int = 15) -> List[ReliabilityEvent]:
        """Get N most recent events"""
    
    def get_by_component(self, component: str) -> List[ReliabilityEvent]:
        """Get all events for component"""
    
    def get_by_severity(self, severity: Severity) -> List[ReliabilityEvent]:
        """Get events by severity"""
```

---

## Business Metrics API

### BusinessImpactCalculator

**Calculate revenue impact of incidents**

```python
class BusinessImpactCalculator:
    def calculate_impact(self, event: ReliabilityEvent) -> Dict[str, float]:
        """
        Calculate business impact
        
        Returns:
            {
                "revenue_loss_estimate": float,      # $ lost
                "users_affected": int,               # Count
                "transactions_failed": int,          # Count
                "recovery_time_minutes": float,      # Estimate
                "total_cost": float                  # $ total impact
            }
        """
```

**Example:**
```python
calculator = BusinessImpactCalculator()
impact = calculator.calculate_impact(event)

print(f"Revenue Loss: ${impact['revenue_loss_estimate']:,.2f}")
print(f"Users Affected: {impact['users_affected']:,}")
```

---

### BusinessMetricsTracker

**Track cumulative business metrics**

```python
class BusinessMetricsTracker:
    def record_incident(
        self,
        downtime_minutes: float,
        revenue_lost: float,
        severity: Severity
    ) -> None:
        """Record incident metrics"""
    
    def get_totals(self) -> Dict[str, Any]:
        """
        Get cumulative totals
        
        Returns:
            {
                "total_incidents": int,
                "total_downtime_minutes": float,
                "total_revenue_saved": float,
                "mttr_minutes": float,
                "availability_percentage": float
            }
        """
```

---

## Utility Functions

### Validation

```python
def validate_component_id(component_id: str) -> str:
    """
    Validate component identifier
    
    Args:
        component_id: Component name to validate
    
    Returns:
        Validated component_id
    
    Raises:
        ValueError: If validation fails
    """
```

```python
def validate_numeric_input(
    value: Any,
    field_name: str,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
    allow_none: bool = False
) -> float:
    """Validate numeric input with range checking"""
```

---

### Time & Calculation

```python
def calculate_mttr(
    incident_start: datetime,
    incident_detected: datetime,
    incident_resolved: datetime
) -> float:
    """
    Calculate Mean Time To Resolve
    
    Returns: Resolution time in minutes
    """
```

```python
def format_timeline(metrics: Dict) -> str:
    """
    Format timeline comparison
    
    Returns: Markdown-formatted timeline
    """
```

---

## Configuration

### Environment Variables

Access via `config.py`:

```python
from config import config

# API Configuration
api_url = config.hf_api_url
api_key = config.hf_api_key

# System Configuration
max_events = config.max_events_stored
batch_size = config.faiss_batch_size

# Business Metrics
revenue_per_min = config.base_revenue_per_minute
```

See [configuration.md](./configuration.md) for complete reference.

---

## Error Handling

### Exception Hierarchy

```python
class ReliabilityFrameworkError(Exception):
    """Base exception for ARF"""
    pass

class ValidationError(ReliabilityFrameworkError):
    """Input validation failed"""
    pass

class FAISSIndexError(ReliabilityFrameworkError):
    """FAISS operations failed"""
    pass

class AgentExecutionError(ReliabilityFrameworkError):
    """Agent analysis failed"""
    pass

class PolicyEngineError(ReliabilityFrameworkError):
    """Policy evaluation/execution failed"""
    pass
```

### Error Handling Patterns

```python
try:
    event = ReliabilityEvent(**event_data)
except ValidationError as e:
    logger.error(f"❌ Validation failed: {e}")
    return {"error": str(e), "status": "invalid_input"}

try:
    results = await orchestrator.analyze_event(event)
except AgentExecutionError as e:
    logger.error(f"❌ Agent failed: {e}", exc_info=True)
    return {"error": "analysis_failed", "details": str(e)}
```

---

## Code Examples

### Complete Event Processing

```python
from app import (
    ReliabilityEvent,
    OrchestrationManager,
    PolicyEngine,
    BusinessImpactCalculator
)

# Create event
event = ReliabilityEvent(
    component="payment-service",
    latency_p99=850.0,
    error_rate=0.25,
    throughput=500.0,
    cpu_util=0.92,
    memory_util=0.85,
    severity=Severity.CRITICAL
)

# Multi-agent analysis
orchestrator = OrchestrationManager(agents)
analysis = await orchestrator.analyze_event(event)

# Policy evaluation
engine = PolicyEngine()
policies = engine.evaluate(event)

# Business impact
calculator = BusinessImpactCalculator()
impact = calculator.calculate_impact(event)

# Execute recovery
for policy in policies:
    for action in policy.actions:
        execute_recovery_action(action, event.component)
```

---

### Custom Agent Implementation

```python
class CustomAgent(BaseAgent):
    def __init__(self):
        super().__init__(AgentSpecialization.CUSTOM)
    
    async def analyze(self, event: ReliabilityEvent) -> Dict[str, Any]:
        # Your custom logic here
        analysis = self._perform_analysis(event)
        
        return {
            "specialization": "custom",
            "confidence": analysis.confidence,
            "findings": analysis.findings,
            "recommendations": analysis.recommendations
        }

# Register with orchestrator
orchestrator.add_agent(CustomAgent())
```

---

## Testing

### Unit Testing APIs

```python
import pytest
from app import ReliabilityEvent, AnomalyDetectionAgent

@pytest.mark.asyncio
async def test_anomaly_detection():
    agent = AnomalyDetectionAgent()
    
    event = ReliabilityEvent(
        component="test-service",
        latency_p99=500.0,
        error_rate=0.15,
        throughput=1000.0,
        cpu_util=0.75,
        memory_util=0.60
    )
    
    result = await agent.analyze(event)
    
    assert result["specialization"] == "detective"
    assert 0.0 <= result["confidence"] <= 1.0
    assert "anomaly_score" in result["findings"]
```

---

## Rate Limiting

All APIs respect rate limits:

- **Event submission:** 60 requests/minute
- **Policy evaluation:** 500 requests/hour
- **FAISS queries:** No limit (local operation)

Exceeding limits returns:
```json
{
  "error": "rate_limit_exceeded",
  "retry_after_seconds": 30
}
```

---

## API Versioning

Current version: **v1.0.0**

Future breaking changes will use new major version.

---

## Support

**Need help integrating ARF?**

- 📧 Email: petter2025us@outlook.com
- 📅 [Book consultation](https://calendly.com/petter2025us/30min)
- 💼 [Professional services](https://lgcylabs.vercel.app/)

---

**Built by [LGCY Labs](https://lgcylabs.vercel.app/) | [Back to README](../README.md)**

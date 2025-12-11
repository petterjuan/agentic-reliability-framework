# Self-Healing Patterns

**Automated Recovery Through Policy-Based Healing**

---

## Table of Contents

- [What is Self-Healing?](#what-is-self-healing)
- [Why Self-Healing?](#why-self-healing)
- [ARF's Self-Healing System](#arfs-self-healing-system)
- [Healing Policies](#healing-policies)
- [Recovery Actions](#recovery-actions)
- [Safety Mechanisms](#safety-mechanisms)
- [Real-World Patterns](#real-world-patterns)
- [Best Practices](#best-practices)

---

## What is Self-Healing?

**Self-healing** systems automatically detect, diagnose, and recover from failures without human intervention.

### Traditional vs Self-Healing

**Traditional Approach:**
```
Failure Occurs
     ↓
Alert Fires
     ↓
Human Woken Up (3 AM)
     ↓
Human Investigates
     ↓
Human Takes Action
     ↓
System Recovers
```

**Time to recovery:** 30-60 minutes  
**Human cost:** Sleep deprivation, burnout

---

**Self-Healing Approach:**
```
Failure Occurs
     ↓
System Detects (milliseconds)
     ↓
System Diagnoses (seconds)
     ↓
System Heals (seconds)
     ↓
System Recovers
     ↓
Human Notified (optional)
```

**Time to recovery:** <1 minute  
**Human cost:** Zero (unless escalation needed)

---

## Why Self-Healing?

### The Problem: Manual Incident Response

**Costs:**
- ⏰ **Time:** Average MTTR = 30-60 minutes
- 💰 **Money:** $50K-$250K per hour of downtime
- 😴 **Humans:** On-call fatigue, burnout
- 📈 **Scale:** Doesn't scale with system growth

**Example:**

```
10 PM: Payment service crashes
10:05 PM: Alert fires
10:10 PM: Engineer wakes up
10:20 PM: Investigation starts
10:45 PM: Root cause found
11:00 PM: Fix deployed
11:15 PM: System recovers

Total downtime: 75 minutes
Revenue lost: $125K
Engineer: Exhausted
```

---

### The Solution: Self-Healing

**Benefits:**
- ⚡ **Speed:** Recovery in seconds, not minutes
- 💰 **Cost:** 90% reduction in downtime costs
- 😊 **Humans:** Sleep through the night
- 📈 **Scale:** Handles 1000x more incidents

**Example:**

```
10:00 PM: Payment service crashes
10:00:05 PM: ARF detects anomaly
10:00:10 PM: ARF diagnoses: connection pool exhausted
10:00:15 PM: ARF restarts service
10:00:30 PM: System recovers

Total downtime: 30 seconds
Revenue lost: $833
Engineer: Sleeping peacefully
```

---

## ARF's Self-Healing System

### Architecture

```
Event Detected
     ↓
Multi-Agent Analysis
     ↓
Policy Engine
     ↓
Matching Policies?
     ↓ Yes
Execute Recovery Actions
     ↓
Monitor Recovery
     ↓
Success? → Done
     ↓ No
Escalate to Human
```

---

### Components

#### **1. Policy Engine**

Evaluates events against healing policies:

```python
class PolicyEngine:
    def evaluate(self, event: ReliabilityEvent) -> List[HealingPolicy]:
        """
        Find policies that match event
        
        Returns matching policies sorted by priority
        """
        matching = []
        
        for policy in self.policies:
            if policy.enabled and not self._in_cooldown(policy):
                if self._all_conditions_match(policy, event):
                    matching.append(policy)
        
        return sorted(matching, key=lambda p: -p.priority)
```

---

#### **2. Healing Policies**

Declarative rules for recovery:

```python
@dataclass(frozen=True)
class HealingPolicy:
    name: str
    conditions: List[PolicyCondition]  # When to trigger
    actions: List[str]                 # What to do
    priority: int = 5                  # 1-10 (higher = more urgent)
    cooldown_seconds: int = 300        # Prevent thrashing
    enabled: bool = True
```

---

#### **3. Recovery Actions**

Concrete steps to fix problems:

```python
RECOVERY_ACTIONS = {
    "restart-service": restart_service,
    "scale-up": scale_up_instances,
    "enable-circuit-breaker": enable_circuit_breaker,
    "fallback-mode": enable_fallback,
    "alert-oncall": alert_human,
    "rollback-deployment": rollback_deploy
}
```

---

## Healing Policies

### Policy Structure

```python
policy = HealingPolicy(
    name="High Latency Recovery",
    
    # Conditions (ALL must match)
    conditions=[
        PolicyCondition(
            metric="latency_p99",
            operator="gt",
            threshold=300.0
        ),
        PolicyCondition(
            metric="error_rate",
            operator="lt",
            threshold=0.5  # Not too many errors
        )
    ],
    
    # Actions (executed in order)
    actions=[
        "restart-service",
        "scale-up"
    ],
    
    priority=8,           # High priority
    cooldown_seconds=600  # 10 minutes
)
```

---

### ARF's Default Policies

#### **Policy 1: Critical Latency Spike**

```python
HealingPolicy(
    name="Critical Latency Spike",
    conditions=[
        PolicyCondition("latency_p99", "gt", 500.0)
    ],
    actions=["restart-service", "alert-oncall"],
    priority=10  # Highest priority
)
```

**When:** Latency > 500ms  
**Do:** Restart service + alert human  
**Why:** Extreme latency requires immediate action

---

#### **Policy 2: High Error Rate**

```python
HealingPolicy(
    name="High Error Rate",
    conditions=[
        PolicyCondition("error_rate", "gt", 0.15),
        PolicyCondition("throughput", "gt", 100.0)
    ],
    actions=["enable-circuit-breaker", "fallback-mode"],
    priority=9
)
```

**When:** Error rate > 15% with traffic  
**Do:** Circuit breaker + fallback  
**Why:** Prevent cascading failures

---

#### **Policy 3: Resource Exhaustion**

```python
HealingPolicy(
    name="Resource Exhaustion",
    conditions=[
        PolicyCondition("cpu_util", "gt", 0.9),
        PolicyCondition("memory_util", "gt", 0.85)
    ],
    actions=["scale-up", "restart-service"],
    priority=7
)
```

**When:** CPU > 90% AND Memory > 85%  
**Do:** Scale up + restart  
**Why:** Capacity overload

---

#### **Policy 4: Moderate Latency**

```python
HealingPolicy(
    name="Moderate Latency",
    conditions=[
        PolicyCondition("latency_p99", "gt", 300.0),
        PolicyCondition("latency_p99", "lt", 500.0)
    ],
    actions=["scale-up"],
    priority=5
)
```

**When:** 300ms < Latency < 500ms  
**Do:** Scale up only  
**Why:** Preventive scaling

---

#### **Policy 5: Throughput Drop**

```python
HealingPolicy(
    name="Throughput Drop",
    conditions=[
        PolicyCondition("throughput", "lt", 500.0)
    ],
    actions=["restart-service"],
    priority=4
)
```

**When:** Throughput < 500 req/sec  
**Do:** Restart service  
**Why:** Possible deadlock or hang

---

### Creating Custom Policies

```python
# Add your own policy
custom_policy = HealingPolicy(
    name="Database Connection Pool Exhaustion",
    conditions=[
        PolicyCondition("latency_p99", "gt", 400.0),
        PolicyCondition("error_rate", "gt", 0.1),
        # Check specific error pattern (would need custom logic)
    ],
    actions=[
        "increase-db-pool-size",
        "restart-connection-manager"
    ],
    priority=8,
    cooldown_seconds=900  # 15 minutes
)

# Register policy
engine.add_policy(custom_policy)
```

---

## Recovery Actions

### Built-In Actions

#### **restart-service**

```python
def restart_service(component: str):
    """
    Gracefully restart service
    
    1. Stop accepting new requests
    2. Wait for in-flight requests (30s timeout)
    3. Kill process
    4. Start new process
    5. Health check
    """
    logger.info(f"Restarting {component}")
    
    # Implementation depends on orchestrator
    # Kubernetes: kubectl rollout restart
    # Docker: docker restart
    # Systemd: systemctl restart
```

**When to use:**
- Memory leaks
- Deadlocks
- Resource exhaustion
- Connection pool issues

---

#### **scale-up**

```python
def scale_up(component: str, increment: int = 1):
    """
    Add more instances
    
    1. Request new instances
    2. Wait for ready state
    3. Add to load balancer
    4. Monitor metrics
    """
    logger.info(f"Scaling up {component} by {increment}")
    
    # Kubernetes: kubectl scale deployment
    # AWS: Auto Scaling Group
    # GCP: Instance Group Manager
```

**When to use:**
- High CPU/memory
- Increasing traffic
- Response time degradation

---

#### **enable-circuit-breaker**

```python
def enable_circuit_breaker(component: str):
    """
    Stop sending traffic to failing component
    
    1. Mark component as unhealthy
    2. Remove from routing
    3. Monitor for recovery
    4. Re-enable when healthy
    """
    logger.info(f"Circuit breaker enabled for {component}")
    
    # Prevents cascading failures
    # Gives component time to recover
```

**When to use:**
- Cascading failures
- Dependency failures
- Overloaded services

---

#### **fallback-mode**

```python
def enable_fallback(component: str):
    """
    Switch to degraded but functional mode
    
    Examples:
    - Use cached data instead of live
    - Disable non-critical features
    - Serve static content
    """
    logger.info(f"Fallback mode for {component}")
    
    # Better than full outage
    # Maintains core functionality
```

**When to use:**
- Dependency unavailable
- Database overload
- External API failures

---

#### **alert-oncall**

```python
def alert_oncall(component: str, severity: str):
    """
    Escalate to human
    
    1. Send PagerDuty/OpsGenie alert
    2. Include incident context
    3. Suggest actions already tried
    """
    logger.warning(f"Alerting on-call for {component}")
    
    # Used when automation can't fix
    # Provides full context to engineer
```

**When to use:**
- Critical failures
- Unknown failure modes
- After multiple recovery attempts failed

---

#### **rollback-deployment**

```python
def rollback_deployment(component: str):
    """
    Revert to previous version
    
    1. Identify previous stable version
    2. Deploy previous version
    3. Verify health
    4. Monitor for stability
    """
    logger.info(f"Rolling back {component}")
    
    # Often the fastest fix for bad deploys
```

**When to use:**
- New deployment causing issues
- Increased error rate after deploy
- Regression detected

---

## Safety Mechanisms

### 1. Cooldown Periods

**Problem:** Repeated execution causes thrashing

```python
# Without cooldown
10:00 - Service crashes → Restart
10:01 - Crashes again → Restart
10:02 - Crashes again → Restart
# Infinite loop!
```

**Solution:**

```python
# With cooldown
10:00 - Service crashes → Restart (cooldown: 10 min)
10:01 - Crashes again → SKIP (in cooldown)
10:02 - Crashes again → SKIP (in cooldown)
10:10 - Cooldown expires
10:11 - Crashes again → Try different policy
```

---

### 2. Rate Limiting

**Problem:** Too many actions overwhelm system

```python
# Without rate limiting
100 failures → 100 restarts in 1 second
# System overload!
```

**Solution:**

```python
class RateLimiter:
    max_per_minute = 60
    max_per_hour = 500
    
    def allow_action(self) -> bool:
        recent = count_recent_actions()
        if recent >= self.max_per_minute:
            return False
        return True
```

---

### 3. Priority Ordering

**Problem:** Multiple policies match

```python
# Event: Latency = 600ms, Error rate = 20%

# Policy A (priority 10): Critical Latency → Restart
# Policy B (priority 9): High Error Rate → Circuit breaker
# Policy C (priority 5): Moderate Latency → Scale up

# Execute highest priority first
```

---

### 4. Validation

**Problem:** Invalid actions cause harm

```python
# Validate before execution
def execute_action(action: str, component: str):
    # Check action exists
    if action not in RECOVERY_ACTIONS:
        raise ValueError(f"Unknown action: {action}")
    
    # Check component exists
    if not component_exists(component):
        raise ValueError(f"Unknown component: {component}")
    
    # Check permissions
    if not has_permission(action, component):
        raise PermissionError(f"No permission for {action}")
    
    # Execute
    RECOVERY_ACTIONS[action](component)
```

---

## Real-World Patterns

### Pattern 1: Circuit Breaker

**Use case:** Prevent cascading failures

```
Service A → Service B (failing)
         ↓
Service A detects B failing
         ↓
Service A stops calling B
         ↓
B recovers without load
         ↓
Service A resumes calling B
```

**Implementation:**

```python
class CircuitBreaker:
    def __init__(self, failure_threshold=5, timeout=60):
        self.failures = 0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.last_failure_time = None
    
    def call(self, func):
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.timeout:
                self.state = "HALF_OPEN"
            else:
                raise CircuitBreakerOpen()
        
        try:
            result = func()
            self.on_success()
            return result
        except Exception as e:
            self.on_failure()
            raise
```

---

### Pattern 2: Retry with Exponential Backoff

**Use case:** Transient failures

```python
def retry_with_backoff(func, max_retries=3):
    for attempt in range(max_retries):
        try:
            return func()
        except TransientError:
            if attempt == max_retries - 1:
                raise
            
            # Exponential backoff
            delay = 2 ** attempt  # 1s, 2s, 4s
            time.sleep(delay)
```

---

### Pattern 3: Bulkhead

**Use case:** Isolate failures

```python
# Separate thread pools per service
payment_pool = ThreadPoolExecutor(max_workers=10)
auth_pool = ThreadPoolExecutor(max_workers=5)
search_pool = ThreadPoolExecutor(max_workers=20)

# If payment fails, auth/search unaffected
```

---

### Pattern 4: Timeout

**Use case:** Prevent hanging

```python
import signal

def with_timeout(func, timeout_seconds):
    def timeout_handler(signum, frame):
        raise TimeoutError()
    
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_seconds)
    
    try:
        result = func()
        signal.alarm(0)  # Cancel alarm
        return result
    except TimeoutError:
        logger.error("Function timed out")
        raise
```

---

### Pattern 5: Health Checks

**Use case:** Detect issues early

```python
def health_check(component: str) -> bool:
    try:
        # Check critical paths
        response = requests.get(f"{component}/health", timeout=5)
        
        if response.status_code != 200:
            return False
        
        health = response.json()
        
        # Verify dependencies
        if not health.get("database_ok"):
            return False
        if not health.get("cache_ok"):
            return False
        
        return True
    except Exception:
        return False
```

---

## Best Practices

### 1. Start Conservative

```python
# Bad: Aggressive healing
if latency > 100ms:
    restart_service()  # Too aggressive!

# Good: Conservative healing
if latency > 500ms:
    restart_service()  # Only critical issues
```

---

### 2. Test Policies in Dev

```python
# Simulate failures in dev environment
def test_policy():
    # Inject high latency
    inject_latency(500)
    
    # Trigger policy
    engine.evaluate(event)
    
    # Verify correct action
    assert action_taken == "restart-service"
    assert cooldown_active
```

---

### 3. Monitor Healing Success

```python
# Track metrics
healing_attempts = Counter()
healing_successes = Counter()
healing_failures = Counter()

def execute_healing(policy, component):
    healing_attempts.inc()
    
    try:
        result = execute_actions(policy.actions, component)
        
        if result.success:
            healing_successes.inc()
        else:
            healing_failures.inc()
            alert_oncall(component, result.error)
    except Exception as e:
        healing_failures.inc()
        logger.error(f"Healing failed: {e}")
```

---

### 4. Have Escalation Path

```python
# Attempt 1: Restart
if not successful:
    # Attempt 2: Scale up
    if not successful:
        # Attempt 3: Alert human
        alert_oncall()
```

---

### 5. Document Policies

```yaml
# policies.yaml
- name: Critical Latency Spike
  description: |
    Handles extreme latency (>500ms) that indicates
    system degradation or failure.
  conditions:
    - metric: latency_p99
      operator: gt
      threshold: 500.0
  actions:
    - restart-service
    - alert-oncall
  reasoning: |
    Restart often fixes memory leaks, deadlocks, or
    resource exhaustion. Alert ensures human oversight.
```

---

## Advanced Topics

### Self-Healing ML Models

```python
# Detect model drift
if prediction_accuracy < 0.7:
    # Retrain model
    retrain_model()
    
    # Deploy new version
    deploy_model(new_version)
    
    # Monitor for improvement
    monitor_accuracy()
```

---

### Chaos Engineering

```python
# Intentionally inject failures to test healing
def chaos_test():
    # Kill random service
    kill_random_service()
    
    # Verify self-healing
    assert service_recovered_within(60)  # seconds
```

---

### Predictive Healing

```python
# Heal BEFORE failure occurs
if predicted_cpu_util > 0.9:
    # Pre-emptively scale up
    scale_up()
    
    # Prevented outage!
```

---

## Further Reading

- [Architecture Overview](./architecture.md) - Complete system design
- [Multi-Agent System](./multi-agent.md) - How agents coordinate healing
- [API Reference](./api.md) - Policy Engine APIs

---

**Built by [LGCY Labs](https://lgcylabs.vercel.app/) | [Back to README](../README.md)**

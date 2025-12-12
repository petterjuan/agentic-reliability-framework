"""
Custom Healing Policies Example
Define domain-specific auto-healing logic for your infrastructure
"""

from agentic_reliability_framework import (
    HealingPolicy,
    HealingAction,
    PolicyCondition,
    PolicyEngine,
    # get_engine  # REMOVED: imported but unused
)

print("="*60)
print("ARF Custom Policies Example")
print("="*60)

# Define custom policies for your infrastructure
custom_policies = [
    # E-commerce: Protect checkout flow at all costs
    HealingPolicy(
        name="checkout_latency_emergency",
        conditions=[
            PolicyCondition(metric="latency_p99", operator="gt", threshold=200.0)
        ],
        actions=[
            HealingAction.TRAFFIC_SHIFT,  # Route to backup
            HealingAction.SCALE_OUT,       # Add capacity
            HealingAction.ALERT_TEAM       # Wake up humans
        ],
        priority=1,  # Highest priority
        cool_down_seconds=120,
        max_executions_per_hour=10
    ),
    
    # Database: Prevent connection pool exhaustion
    HealingPolicy(
        name="db_connection_pool_protection",
        conditions=[
            PolicyCondition(metric="error_rate", operator="gt", threshold=0.20),
            PolicyCondition(metric="latency_p99", operator="gt", threshold=500.0)
        ],
        actions=[
            HealingAction.CIRCUIT_BREAKER,  # Stop the bleeding
            HealingAction.RESTART_CONTAINER # Fresh connection pool
        ],
        priority=1,
        cool_down_seconds=300,
        max_executions_per_hour=5
    ),
    
    # API: Gradual degradation over catastrophic failure
    HealingPolicy(
        name="api_graceful_degradation",
        conditions=[
            PolicyCondition(metric="cpu_util", operator="gt", threshold=0.85)
        ],
        actions=[
            HealingAction.TRAFFIC_SHIFT  # Route to less loaded instances
        ],
        priority=2,
        cool_down_seconds=180,
        max_executions_per_hour=15
    ),
    
    # Memory leak protection
    HealingPolicy(
        name="memory_leak_protection",
        conditions=[
            PolicyCondition(metric="memory_util", operator="gt", threshold=0.90),
            PolicyCondition(metric="memory_util", operator="increase", threshold=0.10, window_minutes=5)
        ],
        actions=[
            HealingAction.RESTART_CONTAINER,  # Kill and restart
            HealingAction.ALERT_TEAM          # Notify for investigation
        ],
        priority=1,
        cool_down_seconds=600,
        max_executions_per_hour=3
    )
]

# Initialize policy engine with custom policies
policy_engine = PolicyEngine(policies=custom_policies)

print("\n✅ Custom policies loaded:")
for policy in custom_policies:
    print(f"  • {policy.name} (priority {policy.priority})")

# Test the policies
print("\n🧪 Testing Policies:")
test_cases = [
    {
        "name": "Checkout Emergency",
        "data": {"component": "checkout-service", "latency_p99": 250.0, "error_rate": 0.05}
    },
    {
        "name": "Database Failure", 
        "data": {"component": "database", "latency_p99": 650.0, "error_rate": 0.35}
    },
    {
        "name": "API High CPU",
        "data": {"component": "api-gateway", "cpu_util": 0.92, "latency_p99": 150.0}
    },
    {
        "name": "Memory Leak",
        "data": {"component": "user-service", "memory_util": 0.95, "memory_util_increase": 0.15}
    }
]

for test in test_cases:
    print(f"\n📋 {test['name']}:")
    actions = policy_engine.evaluate_conditions(test['data'])
    if actions:
        print(f"  🚨 Triggered: {[a.value for a in actions]}")
    else:
        print("  ✅ No actions triggered")  # FIXED: Removed f prefix

print("\n" + "="*60)
print("💡 Integration Example:")
print("="*60)

integration_example = '''
# Integrate custom policies with ARF engine
engine = get_engine()
engine.policy_engine = policy_engine  # Replace default policies

# Now ARF will use your custom policies for auto-healing
result = await engine.process_event_enhanced(
    component="checkout-service",
    latency_p99=280.0,  # Will trigger checkout_latency_emergency
    error_rate=0.08,
    cpu_util=0.65
)

print(f"Applied actions: {result.get('healing_actions', [])}")
'''

print(integration_example)
print("="*60)
print("✅ Example complete! Copy the integration code to use custom policies.")
print("="*60)

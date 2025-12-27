"""Pytest configuration - OSS EDITION COMPATIBLE VERSION"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime, timezone
import sys


# OSS-compatible imports with fallback
try:
    # Try to import models - they might not be available in minimal OSS install
    from agentic_reliability_framework.models import (
        ReliabilityEvent, HealingPolicy, PolicyCondition,
        HealingAction, EventSeverity
    )
    MODELS_AVAILABLE = True
    print("✅ Models module available for tests")
except ImportError as e:
    MODELS_AVAILABLE = False
    print(f"⚠️  Models module not available in OSS edition (using mocks): {e}")
    
    # Create minimal mock classes for OSS testing
    class EventSeverity:
        LOW = "low"
        MEDIUM = "medium" 
        HIGH = "high"
        CRITICAL = "critical"
    
    class HealingAction:
        RESTART_CONTAINER = "restart_container"
        SCALE_HORIZONTAL = "scale_horizontal"
        ROLLBACK_DEPLOYMENT = "rollback_deployment"
        TRAFFIC_SHIFT = "traffic_shift"
        CIRCUIT_BREAKER = "circuit_breaker"
        ROLLBACK = "rollback"
        ALERT_TEAM = "alert_team"
        NO_ACTION = "no_action"
    
    # Minimal classes for fixtures
    class PolicyCondition:
        def __init__(self, metric="error_rate", operator="gt", threshold=0.1):
            self.metric = metric
            self.operator = operator
            self.threshold = threshold
    
    class ReliabilityEvent:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    class HealingPolicy:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)


# Try to import PolicyEngine with fallback
try:
    from agentic_reliability_framework.healing_policies import PolicyEngine
    POLICY_ENGINE_AVAILABLE = True
    print("✅ PolicyEngine available for tests")
except ImportError as e:
    POLICY_ENGINE_AVAILABLE = False
    print(f"⚠️  PolicyEngine not available in OSS edition (using mocks): {e}")
    
    class PolicyEngine:
        def __init__(self):
            self.policies = []
        
        def add_policy(self, policy):
            self.policies.append(policy)


@pytest.fixture
def sample_event():
    if MODELS_AVAILABLE:
        return ReliabilityEvent(
            component="test-service",
            timestamp=datetime.now(timezone.utc),
            latency_p99=250.0,
            error_rate=0.15,
            throughput=1000,
            cpu_util=0.65,
            memory_util=0.70,
            service_mesh="default",
            severity=EventSeverity.MEDIUM
        )
    else:
        # Return mock event
        mock_event = type('MockEvent', (), {
            'component': 'test-service',
            'timestamp': datetime.now(timezone.utc),
            'latency_p99': 250.0,
            'error_rate': 0.15,
            'throughput': 1000,
            'cpu_util': 0.65,
            'memory_util': 0.70,
            'service_mesh': 'default',
            'severity': 'medium'
        })()
        return mock_event


@pytest.fixture
def normal_event():
    if MODELS_AVAILABLE:
        return ReliabilityEvent(
            component="test-service",
            timestamp=datetime.now(timezone.utc),
            latency_p99=150.0,
            error_rate=0.02,
            throughput=2000,
            cpu_util=0.50,
            memory_util=0.55,
            service_mesh="default",
            severity=EventSeverity.LOW
        )
    else:
        mock_event = type('MockEvent', (), {
            'component': 'test-service',
            'timestamp': datetime.now(timezone.utc),
            'latency_p99': 150.0,
            'error_rate': 0.02,
            'throughput': 2000,
            'cpu_util': 0.50,
            'memory_util': 0.55,
            'service_mesh': 'default',
            'severity': 'low'
        })()
        return mock_event


@pytest.fixture
def critical_event():
    if MODELS_AVAILABLE:
        return ReliabilityEvent(
            component="critical-service",
            timestamp=datetime.now(timezone.utc),
            latency_p99=5000.0,
            error_rate=0.45,
            throughput=100,
            cpu_util=0.95,
            memory_util=0.90,
            severity=EventSeverity.CRITICAL
        )
    else:
        mock_event = type('MockEvent', (), {
            'component': 'critical-service',
            'timestamp': datetime.now(timezone.utc),
            'latency_p99': 5000.0,
            'error_rate': 0.45,
            'throughput': 100,
            'cpu_util': 0.95,
            'memory_util': 0.90,
            'severity': 'critical'
        })()
        return mock_event


@pytest.fixture
def sample_policy():
    if MODELS_AVAILABLE:
        return HealingPolicy(
            name="Restart on High Errors",
            description="Restart when error rate > 10%",
            conditions=[PolicyCondition(
                metric="error_rate",
                operator="gt",
                threshold=0.10
            )],
            actions=[HealingAction.RESTART_CONTAINER],
            cooldown_seconds=300,
            enabled=True
        )
    else:
        mock_policy = type('MockPolicy', (), {
            'name': 'Restart on High Errors',
            'description': 'Restart when error rate > 10%',
            'conditions': [PolicyCondition(metric="error_rate", operator="gt", threshold=0.10)],
            'actions': ['restart_container'],
            'cooldown_seconds': 300,
            'enabled': True
        })()
        return mock_policy


@pytest.fixture
def scale_policy():
    if MODELS_AVAILABLE:
        return HealingPolicy(
            name="Scale on High CPU",
            description="Scale when CPU > 80%",
            conditions=[PolicyCondition(
                metric="cpu_util",
                operator="gt",
                threshold=0.80
            )],
            actions=[HealingAction.SCALE_HORIZONTAL],
            cooldown_seconds=600,
            enabled=True
        )
    else:
        mock_policy = type('MockPolicy', (), {
            'name': 'Scale on High CPU',
            'description': 'Scale when CPU > 80%',
            'conditions': [PolicyCondition(metric="cpu_util", operator="gt", threshold=0.80)],
            'actions': ['scale_horizontal'],
            'cooldown_seconds': 600,
            'enabled': True
        })()
        return mock_policy


@pytest.fixture
def rollback_policy():
    if MODELS_AVAILABLE:
        return HealingPolicy(
            name="Rollback on Critical",
            description="Rollback on error rate > 30%",
            conditions=[PolicyCondition(
                metric="error_rate",
                operator="gt",
                threshold=0.30
            )],
            actions=[HealingAction.ROLLBACK_DEPLOYMENT],
            cooldown_seconds=900,
            enabled=True
        )
    else:
        mock_policy = type('MockPolicy', (), {
            'name': 'Rollback on Critical',
            'description': 'Rollback on error rate > 30%',
            'conditions': [PolicyCondition(metric="error_rate", operator="gt", threshold=0.30)],
            'actions': ['rollback_deployment'],
            'cooldown_seconds': 900,
            'enabled': True
        })()
        return mock_policy


@pytest.fixture
def disabled_policy():
    if MODELS_AVAILABLE:
        return HealingPolicy(
            name="Disabled Policy",
            description="Should never execute",
            conditions=[PolicyCondition(
                metric="error_rate",
                operator="gt",
                threshold=0.01
            )],
            actions=[HealingAction.RESTART_CONTAINER],
            cooldown_seconds=300,
            enabled=False
        )
    else:
        mock_policy = type('MockPolicy', (), {
            'name': 'Disabled Policy',
            'description': 'Should never execute',
            'conditions': [PolicyCondition(metric="error_rate", operator="gt", threshold=0.01)],
            'actions': ['restart_container'],
            'cooldown_seconds': 300,
            'enabled': False
        })()
        return mock_policy


@pytest.fixture
def policy_engine():
    if POLICY_ENGINE_AVAILABLE:
        return PolicyEngine()
    else:
        return PolicyEngine()  # Use mock class


@pytest.fixture
def policy_engine_with_policies(sample_policy, scale_policy):
    engine = policy_engine()  # Use the fixture
    engine.add_policy(sample_policy)
    engine.add_policy(scale_policy)
    return engine


@pytest.fixture
def mock_faiss_memory():
    mock = MagicMock()
    mock.search_similar = AsyncMock(return_value=[])
    mock.add_incident = AsyncMock()
    return mock


@pytest.fixture
def event_factory():
    def _create_event(
        component: str = "test-service",
        latency_p99: float = 150.0,
        error_rate: float = 0.05,
        throughput: int = 1000,
        cpu_util: float = 0.60,
        memory_util: float = 0.65,
        severity: str = "medium"
    ):
        if MODELS_AVAILABLE:
            severity_enum = getattr(EventSeverity, severity.upper())
            return ReliabilityEvent(
                component=component,
                timestamp=datetime.now(timezone.utc),
                latency_p99=latency_p99,
                error_rate=error_rate,
                throughput=throughput,
                cpu_util=cpu_util,
                memory_util=memory_util,
                severity=severity_enum
            )
        else:
            mock_event = type('MockEvent', (), {
                'component': component,
                'timestamp': datetime.now(timezone.utc),
                'latency_p99': latency_p99,
                'error_rate': error_rate,
                'throughput': throughput,
                'cpu_util': cpu_util,
                'memory_util': memory_util,
                'severity': severity
            })()
            return mock_event
    return _create_event


@pytest.fixture
def policy_factory():
    def _create_policy(
        name: str = "Test Policy",
        metric: str = "error_rate",
        operator: str = "gt",
        threshold: float = 0.10,
        action: str = "restart_container",
        cooldown_seconds: int = 300,
        enabled: bool = True
    ):
        if MODELS_AVAILABLE:
            action_enum = getattr(HealingAction, action.upper()) if hasattr(HealingAction, action.upper()) else HealingAction.RESTART_CONTAINER
            return HealingPolicy(
                name=name,
                description=f"Policy for {metric} {operator} {threshold}",
                conditions=[PolicyCondition(
                    metric=metric,
                    operator=operator,
                    threshold=threshold
                )],
                actions=[action_enum],
                cooldown_seconds=cooldown_seconds,
                enabled=enabled
            )
        else:
            mock_policy = type('MockPolicy', (), {
                'name': name,
                'description': f'Policy for {metric} {operator} {threshold}',
                'conditions': [PolicyCondition(metric=metric, operator=operator, threshold=threshold)],
                'actions': [action],
                'cooldown_seconds': cooldown_seconds,
                'enabled': enabled
            })()
            return mock_policy
    return _create_policy


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "integration: integration tests")
    config.addinivalue_line("markers", "unit: unit tests")
    config.addinivalue_line("markers", "oss: OSS-specific tests")


@pytest.fixture
def trigger_event():
    """Event that triggers sample_policy (error_rate > 0.10)"""
    if MODELS_AVAILABLE:
        return ReliabilityEvent(
            component="failing-service",
            timestamp=datetime.now(timezone.utc),
            latency_p99=300.0,
            error_rate=0.15,  # 15% - WILL trigger policy
            throughput=1000,
            cpu_util=0.70,
            memory_util=0.65,
            service_mesh="default",
            severity=EventSeverity.HIGH
        )
    else:
        mock_event = type('MockEvent', (), {
            'component': 'failing-service',
            'timestamp': datetime.now(timezone.utc),
            'latency_p99': 300.0,
            'error_rate': 0.15,
            'throughput': 1000,
            'cpu_util': 0.70,
            'memory_util': 0.65,
            'service_mesh': 'default',
            'severity': 'high'
        })()
        return mock_event


@pytest.fixture
def sample_metrics():
    """Sample timeline metrics for testing"""
    return {
        'incident_start': '2025-12-09T09:00:00Z',
        'incident_detected': '2025-12-09T09:02:00Z',
        'incident_resolved': '2025-12-09T09:15:00Z',
        'industry_mttr_minutes': 14.0,
        'arf_mttr_minutes': 2.0,
        'time_saved_minutes': 12.0,
        'cost_per_minute': 1000.0,
        'cost_savings': 12000.0
    }


@pytest.fixture
def clear_module_cache():
    """Fixture to clear module cache for import testing"""
    import sys
    
    def _clear_cache():
        modules_to_clear = [
            'agentic_reliability_framework',
            'agentic_reliability_framework.arf_core',
            'agentic_reliability_framework.arf_core.models.healing_intent',
            'agentic_reliability_framework.models',
            'agentic_reliability_framework.healing_policies',
        ]
        
        cleared = []
        for module in modules_to_clear:
            if module in sys.modules:
                del sys.modules[module]
                cleared.append(module)
        
        return cleared
    
    return _clear_cache

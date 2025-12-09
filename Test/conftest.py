"""
Pytest configuration and shared fixtures for ARF tests - COMPLETE FIXED VERSION

FIXES APPLIED:
- latency → latency_p99 (required field)
- cpu_percent → cpu_util (fraction 0.0-1.0, not percentage)
- memory_percent → memory_util (fraction 0.0-1.0, not percentage)
- operator: '>' → 'gt', '<' → 'lt', '=' → 'eq', '>=' → 'gte', '<=' → 'lte'
"""

import pytest
from unittest.mock import Mock, AsyncMock, MagicMock
from datetime import datetime, timezone, timedelta
from typing import Dict, List
import asyncio

from models import (
    ReliabilityEvent,
    HealingPolicy,
    PolicyCondition,
    HealingAction,
    EventSeverity,
    AnomalyResult,
    ForecastResult
)

from healing_policies import PolicyEngine


# ============================================================================
# CORE EVENT FIXTURES
# ============================================================================

@pytest.fixture
def sample_event():
    """Sample test event with correct field names and ranges"""
    return ReliabilityEvent(
        component="test-service",
        timestamp=datetime.now(timezone.utc),
        latency_p99=250.0,
        error_rate=0.05,
        throughput=1000,
        cpu_util=0.65,
        memory_util=0.70,
        service_mesh="default",
        severity=EventSeverity.MEDIUM
    )


@pytest.fixture
def critical_event():
    """Critical severity event"""
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


@pytest.fixture
def low_severity_event():
    """Low severity healthy event"""
    return ReliabilityEvent(
        component="healthy-service",
        timestamp=datetime.now(timezone.utc),
        latency_p99=50.0,
        error_rate=0.001,
        throughput=5000,
        cpu_util=0.30,
        memory_util=0.40,
        severity=EventSeverity.LOW
    )


@pytest.fixture
def memory_leak_event():
    """Event simulating memory leak"""
    return ReliabilityEvent(
        component="leaky-service",
        timestamp=datetime.now(timezone.utc),
        latency_p99=150.0,
        error_rate=0.02,
        throughput=1000,
        cpu_util=0.50,
        memory_util=0.94,
        severity=EventSeverity.HIGH
    )


# ============================================================================
# POLICY FIXTURES
# ============================================================================

@pytest.fixture
def sample_policy():
    """Sample policy with correct operator format"""
    return HealingPolicy(
        name="Restart on High Errors",
        description="Restart when error rate > 10%",
        condition=PolicyCondition(
            metric="error_rate",
            operator="gt",
            threshold=0.10
        ),
        action=HealingAction.RESTART_CONTAINER,
        cooldown_seconds=300,
        enabled=True
    )


@pytest.fixture
def scale_policy():
    """Policy for scaling on high CPU"""
    return HealingPolicy(
        name="Scale on High CPU",
        description="Scale when CPU > 80%",
        condition=PolicyCondition(
            metric="cpu_util",
            operator="gt",
            threshold=0.80
        ),
        action=HealingAction.SCALE_HORIZONTAL,
        cooldown_seconds=600,
        enabled=True
    )


@pytest.fixture
def rollback_policy():
    """Policy for rollback on critical errors"""
    return HealingPolicy(
        name="Rollback on Critical",
        description="Rollback on error rate > 30%",
        condition=PolicyCondition(
            metric="error_rate",
            operator="gt",
            threshold=0.30
        ),
        action=HealingAction.ROLLBACK_DEPLOYMENT,
        cooldown_seconds=900,
        enabled=True
    )


@pytest.fixture
def circuit_breaker_policy():
    """Policy for circuit breaker"""
    return HealingPolicy(
        name="Circuit Breaker",
        description="Enable circuit breaker on high errors",
        condition=PolicyCondition(
            metric="error_rate",
            operator="gt",
            threshold=0.25
        ),
        action=HealingAction.CIRCUIT_BREAKER,
        cooldown_seconds=180,
        enabled=True
    )


@pytest.fixture
def disabled_policy():
    """Disabled policy that should never trigger"""
    return HealingPolicy(
        name="Disabled Policy",
        description="Should never execute",
        condition=PolicyCondition(
            metric="error_rate",
            operator="gt",
            threshold=0.01
        ),
        action=HealingAction.RESTART_CONTAINER,
        cooldown_seconds=300,
        enabled=False
    )


# ============================================================================
# POLICY ENGINE FIXTURES
# ============================================================================

@pytest.fixture
def policy_engine():
    """PolicyEngine instance"""
    return PolicyEngine()


@pytest.fixture
def policy_engine_with_policies(sample_policy, scale_policy):
    """PolicyEngine pre-loaded with policies"""
    engine = PolicyEngine()
    engine.add_policy(sample_policy)
    engine.add_policy(scale_policy)
    return engine


# ============================================================================
# MOCK FIXTURES
# ============================================================================

@pytest.fixture
def mock_faiss_memory():
    """Mock FAISS memory"""
    mock = MagicMock()
    mock.search_similar = AsyncMock(return_value=[])
    mock.add_incident = AsyncMock()
    mock.save = AsyncMock()
    return mock


@pytest.fixture
def mock_anomaly_detector():
    """Mock anomaly detector"""
    mock = MagicMock()
    mock.detect_anomaly = Mock(return_value=AnomalyResult(
        is_anomaly=True,
        confidence=0.85,
        affected_metrics=["latency_p99", "error_rate"],
        severity=EventSeverity.HIGH
    ))
    return mock


# ============================================================================
# FACTORY FIXTURES
# ============================================================================

@pytest.fixture
def event_factory():
    """Factory to create custom events in tests"""
    def _create_event(
        component: str = "test-service",
        latency_p99: float = 150.0,
        error_rate: float = 0.05,
        throughput: int = 1000,
        cpu_util: float = 0.60,
        memory_util: float = 0.65,
        severity: EventSeverity = EventSeverity.MEDIUM
    ) -> ReliabilityEvent:
        return ReliabilityEvent(
            component=component,
            timestamp=datetime.now(timezone.utc),
            latency_p99=latency_p99,
            error_rate=error_rate,
            throughput=throughput,
            cpu_util=cpu_util,
            memory_util=memory_util,
            severity=severity
        )
    return _create_event


@pytest.fixture
def policy_factory():
    """Factory to create custom policies in tests"""
    def _create_policy(
        name: str = "Test Policy",
        metric: str = "error_rate",
        operator: str = "gt",
        threshold: float = 0.10,
        action: HealingAction = HealingAction.RESTART_CONTAINER,
        cooldown_seconds: int = 300,
        enabled: bool = True
    ) -> HealingPolicy:
        return HealingPolicy(
            name=name,
            description=f"Policy for {metric} {operator} {threshold}",
            condition=PolicyCondition(
                metric=metric,
                operator=operator,
                threshold=threshold
            ),
            action=action,
            cooldown_seconds=cooldown_seconds,
            enabled=enabled
        )
    return _create_policy


# ============================================================================
# TIME-BASED FIXTURES
# ============================================================================

@pytest.fixture
def past_event():
    """Event from 1 hour ago"""
    return ReliabilityEvent(
        component="test-service",
        timestamp=datetime.now(timezone.utc) - timedelta(hours=1),
        latency_p99=200.0,
        error_rate=0.03,
        throughput=1500,
        cpu_util=0.60,
        memory_util=0.65
    )


@pytest.fixture
def event_sequence():
    """Sequence of events showing degradation"""
    base_time = datetime.now(timezone.utc)
    
    return [
        ReliabilityEvent(
            component="degrading-service",
            timestamp=base_time,
            latency_p99=100.0,
            error_rate=0.01,
            throughput=3000,
            cpu_util=0.50,
            memory_util=0.50,
            severity=EventSeverity.LOW
        ),
        ReliabilityEvent(
            component="degrading-service",
            timestamp=base_time + timedelta(minutes=5),
            latency_p99=200.0,
            error_rate=0.05,
            throughput=2500,
            cpu_util=0.65,
            memory_util=0.60,
            severity=EventSeverity.MEDIUM
        ),
        ReliabilityEvent(
            component="degrading-service",
            timestamp=base_time + timedelta(minutes=10),
            latency_p99=500.0,
            error_rate=0.15,
            throughput=1500,
            cpu_util=0.80,
            memory_util=0.75,
            severity=EventSeverity.HIGH
        ),
        ReliabilityEvent(
            component="degrading-service",
            timestamp=base_time + timedelta(minutes=15),
            latency_p99=2000.0,
            error_rate=0.40,
            throughput=500,
            cpu_util=0.95,
            memory_util=0.90,
            severity=EventSeverity.CRITICAL
        )
    ]


# ============================================================================
# ASYNC UTILITIES
# ============================================================================

@pytest.fixture
def event_loop():
    """Event loop for async tests"""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ============================================================================
# PYTEST CONFIGURATION
# ============================================================================

def pytest_configure(config):
    """Register custom markers"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "performance: marks performance benchmark tests"
    )

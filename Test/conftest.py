"""Pytest configuration - FINAL FIXED VERSION"""

import pytest
from unittest.mock import Mock, AsyncMock, MagicMock
from datetime import datetime, timezone, timedelta
from typing import Dict, List
import asyncio

from models import (
    ReliabilityEvent, HealingPolicy, PolicyCondition,
    HealingAction, EventSeverity, AnomalyResult, ForecastResult
)
from healing_policies import PolicyEngine


@pytest.fixture
def sample_event():
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


@pytest.fixture
def normal_event():
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


@pytest.fixture
def critical_event():
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
def sample_policy():
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


@pytest.fixture
def scale_policy():
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


@pytest.fixture
def rollback_policy():
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


@pytest.fixture
def disabled_policy():
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


@pytest.fixture
def policy_engine():
    return PolicyEngine()


@pytest.fixture
def policy_engine_with_policies(sample_policy, scale_policy):
    engine = PolicyEngine()
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
            conditions=[PolicyCondition(
                metric=metric,
                operator=operator,
                threshold=threshold
            )],
            actions=[action],
            cooldown_seconds=cooldown_seconds,
            enabled=enabled
        )
    return _create_policy


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "integration: integration tests")
    config.addinivalue_line("markers", "unit: unit tests")


@pytest.fixture
def trigger_event():
    """Event that triggers sample_policy (error_rate > 0.10)"""
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

"""
Unit tests for Pydantic models with validation and security tests
"""

import pytest
from agentic_reliability_framework.datetime import datetime, timezone
from agentic_reliability_framework.pydantic import ValidationError
from agentic_reliability_framework.models import (
    ReliabilityEvent,
    EventSeverity,
    HealingPolicy,
    HealingAction,
    PolicyCondition,
    AnomalyResult,
    ForecastResult
)


class TestReliabilityEventValidation:
    """Test ReliabilityEvent validation"""
    
    def test_valid_event_creation(self):
        """Test creating a valid event"""
        event = ReliabilityEvent(
            component="api-service",
            latency_p99=150.0,
            error_rate=0.05,
            throughput=1000.0,
            cpu_util=0.7,
            memory_util=0.6
        )
        
        assert event.component == "api-service"
        assert event.latency_p99 == 150.0
        assert event.error_rate == 0.05
        assert isinstance(event.timestamp, datetime)
        assert event.severity == EventSeverity.LOW
    
    def test_component_validation_valid(self):
        """Test valid component IDs"""
        valid_ids = ["api-service", "auth-service", "payment-service-v2", "db-01"]
        
        for component_id in valid_ids:
            event = ReliabilityEvent(
                component=component_id,
                latency_p99=100.0,
                error_rate=0.01,
                throughput=1000.0
            )
            assert event.component == component_id
    
    def test_component_validation_invalid(self):
        """Test invalid component IDs are rejected"""
        invalid_ids = [
            "API-SERVICE",  # Uppercase
            "api_service",  # Underscore
            "api service",  # Space
            "api@service",  # Special char
            "",  # Empty
        ]
        
        for component_id in invalid_ids:
            with pytest.raises(ValidationError) as exc_info:
                ReliabilityEvent(
                    component=component_id,
                    latency_p99=100.0,
                    error_rate=0.01,
                    throughput=1000.0
                )
            assert "component" in str(exc_info.value).lower()
    
    def test_latency_bounds(self):
        """Test latency validation bounds"""
        # Valid latency
        event = ReliabilityEvent(
            component="test-service",
            latency_p99=100.0,
            error_rate=0.01,
            throughput=1000.0
        )
        assert event.latency_p99 == 100.0
        
        # Negative latency should fail
        with pytest.raises(ValidationError):
            ReliabilityEvent(
                component="test-service",
                latency_p99=-10.0,
                error_rate=0.01,
                throughput=1000.0
            )
        
        # Extremely high latency should fail (> 5 minutes)
        with pytest.raises(ValidationError):
            ReliabilityEvent(
                component="test-service",
                latency_p99=400000.0,  # > 300000ms limit
                error_rate=0.01,
                throughput=1000.0
            )
    
    def test_error_rate_bounds(self):
        """Test error rate validation"""
        # Valid error rate
        event = ReliabilityEvent(
            component="test-service",
            latency_p99=100.0,
            error_rate=0.5,
            throughput=1000.0
        )
        assert event.error_rate == 0.5
        
        # Negative error rate should fail
        with pytest.raises(ValidationError):
            ReliabilityEvent(
                component="test-service",
                latency_p99=100.0,
                error_rate=-0.1,
                throughput=1000.0
            )
        
        # Error rate > 1 should fail
        with pytest.raises(ValidationError):
            ReliabilityEvent(
                component="test-service",
                latency_p99=100.0,
                error_rate=1.5,
                throughput=1000.0
            )
    
    def test_resource_utilization_bounds(self):
        """Test CPU and memory utilization bounds"""
        # Valid utilization
        event = ReliabilityEvent(
            component="test-service",
            latency_p99=100.0,
            error_rate=0.01,
            throughput=1000.0,
            cpu_util=0.85,
            memory_util=0.75
        )
        assert event.cpu_util == 0.85
        assert event.memory_util == 0.75
        
        # CPU > 1 should fail
        with pytest.raises(ValidationError):
            ReliabilityEvent(
                component="test-service",
                latency_p99=100.0,
                error_rate=0.01,
                throughput=1000.0,
                cpu_util=1.5
            )
        
        # Memory < 0 should fail
        with pytest.raises(ValidationError):
            ReliabilityEvent(
                component="test-service",
                latency_p99=100.0,
                error_rate=0.01,
                throughput=1000.0,
                memory_util=-0.1
            )


class TestEventFingerprint:
    """Test event fingerprint generation (SHA-256)"""
    
    def test_fingerprint_is_sha256(self):
        """Test that fingerprint uses SHA-256 (64 hex chars)"""
        event = ReliabilityEvent(
            component="test-service",
            latency_p99=100.0,
            error_rate=0.05,
            throughput=1000.0
        )
        
        # SHA-256 produces 64 hex characters
        assert len(event.fingerprint) == 64
        assert all(c in '0123456789abcdef' for c in event.fingerprint)
    
    def test_fingerprint_deterministic(self):
        """Test that same inputs produce same fingerprint"""
        event1 = ReliabilityEvent(
            component="test-service",
            service_mesh="default",
            latency_p99=100.0,
            error_rate=0.05,
            throughput=1000.0
        )
        
        event2 = ReliabilityEvent(
            component="test-service",
            service_mesh="default",
            latency_p99=100.0,
            error_rate=0.05,
            throughput=1000.0
        )
        
        # Should produce same fingerprint (timestamp not included)
        assert event1.fingerprint == event2.fingerprint
    
    def test_fingerprint_different_for_different_events(self):
        """Test that different events produce different fingerprints"""
        event1 = ReliabilityEvent(
            component="service-1",
            latency_p99=100.0,
            error_rate=0.05,
            throughput=1000.0
        )
        
        event2 = ReliabilityEvent(
            component="service-2",
            latency_p99=100.0,
            error_rate=0.05,
            throughput=1000.0
        )
        
        assert event1.fingerprint != event2.fingerprint
    
    def test_fingerprint_not_md5(self):
        """Test that fingerprint is NOT MD5 (security fix verification)"""
        event = ReliabilityEvent(
            component="test-service",
            latency_p99=100.0,
            error_rate=0.05,
            throughput=1000.0
        )
        
        # MD5 produces 32 hex chars, SHA-256 produces 64
        assert len(event.fingerprint) != 32
        assert len(event.fingerprint) == 64


class TestEventImmutability:
    """Test that events are immutable (frozen)"""
    
    def test_event_is_frozen(self):
        """Test that ReliabilityEvent is frozen"""
        event = ReliabilityEvent(
            component="test-service",
            latency_p99=100.0,
            error_rate=0.05,
            throughput=1000.0
        )
        
        # Attempting to modify should raise ValidationError
        with pytest.raises(ValidationError):
            event.latency_p99 = 200.0
    
    def test_model_copy_with_update(self):
        """Test that model_copy creates new instance with updates"""
        event1 = ReliabilityEvent(
            component="test-service",
            latency_p99=100.0,
            error_rate=0.05,
            throughput=1000.0,
            severity=EventSeverity.LOW
        )
        
        # Create modified copy
        event2 = event1.model_copy(update={'severity': EventSeverity.HIGH})
        
        # Original unchanged
        assert event1.severity == EventSeverity.LOW
        # Copy updated
        assert event2.severity == EventSeverity.HIGH
        # Other fields same
        assert event2.component == event1.component
        assert event2.latency_p99 == event1.latency_p99


class TestDependencyValidation:
    """Test dependency cycle detection"""
    
    def test_valid_dependencies(self):
        """Test valid dependency configuration"""
        event = ReliabilityEvent(
            component="api-service",
            latency_p99=100.0,
            error_rate=0.05,
            throughput=1000.0,
            upstream_deps=["auth-service", "database"],
            downstream_deps=["frontend", "mobile-app"]
        )
        
        assert "auth-service" in event.upstream_deps
        assert "frontend" in event.downstream_deps
    
    def test_circular_dependency_detected(self):
        """Test that circular dependencies are detected"""
        with pytest.raises(ValidationError) as exc_info:
            ReliabilityEvent(
                component="api-service",
                latency_p99=100.0,
                error_rate=0.05,
                throughput=1000.0,
                upstream_deps=["auth-service", "database"],
                downstream_deps=["database", "frontend"]  # 'database' in both
            )
        
        error_msg = str(exc_info.value).lower()
        assert "circular" in error_msg or "database" in error_msg
    
    def test_dependency_name_validation(self):
        """Test that dependency names follow same rules as component IDs"""
        # Valid dependency names
        event = ReliabilityEvent(
            component="api-service",
            latency_p99=100.0,
            error_rate=0.05,
            throughput=1000.0,
            upstream_deps=["auth-service", "db-01", "cache-v2"]
        )
        assert len(event.upstream_deps) == 3
        
        # Invalid dependency names
        with pytest.raises(ValidationError):
            ReliabilityEvent(
                component="api-service",
                latency_p99=100.0,
                error_rate=0.05,
                throughput=1000.0,
                upstream_deps=["AUTH_SERVICE"]  # Uppercase/underscore
            )


class TestPolicyConditionModel:
    """Test PolicyCondition structured model"""
    
    def test_valid_policy_condition(self):
        """Test creating valid policy conditions"""
        condition = PolicyCondition(
            metric="latency_p99",
            operator="gt",
            threshold=150.0
        )
        
        assert condition.metric == "latency_p99"
        assert condition.operator == "gt"
        assert condition.threshold == 150.0
    
    def test_policy_condition_frozen(self):
        """Test that PolicyCondition is immutable"""
        condition = PolicyCondition(
            metric="error_rate",
            operator="gt",
            threshold=0.1
        )
        
        with pytest.raises(ValidationError):
            condition.threshold = 0.2
    
    def test_invalid_metric(self):
        """Test that invalid metrics are rejected"""
        with pytest.raises(ValidationError):
            PolicyCondition(
                metric="invalid_metric",
                operator="gt",
                threshold=100.0
            )
    
    def test_invalid_operator(self):
        """Test that invalid operators are rejected"""
        with pytest.raises(ValidationError):
            PolicyCondition(
                metric="latency_p99",
                operator="invalid_op",
                threshold=100.0
            )
    
    def test_negative_threshold(self):
        """Test that negative thresholds are rejected"""
        with pytest.raises(ValidationError):
            PolicyCondition(
                metric="latency_p99",
                operator="gt",
                threshold=-100.0
            )


class TestHealingPolicyModel:
    """Test HealingPolicy model"""
    
    def test_valid_healing_policy(self):
        """Test creating valid healing policy"""
        policy = HealingPolicy(
            name="high_latency_restart",
            conditions=[
                PolicyCondition(metric="latency_p99", operator="gt", threshold=300.0)
            ],
            actions=[HealingAction.RESTART_CONTAINER, HealingAction.ALERT_TEAM],
            priority=1,
            cool_down_seconds=300
        )
        
        assert policy.name == "high_latency_restart"
        assert len(policy.conditions) == 1
        assert len(policy.actions) == 2
        assert policy.priority == 1
    
    def test_policy_frozen(self):
        """Test that HealingPolicy is immutable"""
        policy = HealingPolicy(
            name="test_policy",
            conditions=[
                PolicyCondition(metric="error_rate", operator="gt", threshold=0.1)
            ],
            actions=[HealingAction.ROLLBACK],
            priority=2
        )
        
        with pytest.raises(ValidationError):
            policy.priority = 5
    
    def test_empty_conditions_rejected(self):
        """Test that policies must have at least one condition"""
        with pytest.raises(ValidationError):
            HealingPolicy(
                name="empty_policy",
                conditions=[],  # Empty
                actions=[HealingAction.ALERT_TEAM],
                priority=3
            )
    
    def test_empty_actions_rejected(self):
        """Test that policies must have at least one action"""
        with pytest.raises(ValidationError):
            HealingPolicy(
                name="empty_actions",
                conditions=[
                    PolicyCondition(metric="latency_p99", operator="gt", threshold=100.0)
                ],
                actions=[],  # Empty
                priority=3
            )
    
    def test_priority_bounds(self):
        """Test priority validation (1-5)"""
        # Valid priority
        policy = HealingPolicy(
            name="test",
            conditions=[PolicyCondition(metric="latency_p99", operator="gt", threshold=100.0)],
            actions=[HealingAction.ALERT_TEAM],
            priority=3
        )
        assert policy.priority == 3
        
        # Priority < 1 should fail
        with pytest.raises(ValidationError):
            HealingPolicy(
                name="test",
                conditions=[PolicyCondition(metric="latency_p99", operator="gt", threshold=100.0)],
                actions=[HealingAction.ALERT_TEAM],
                priority=0
            )
        
        # Priority > 5 should fail
        with pytest.raises(ValidationError):
            HealingPolicy(
                name="test",
                conditions=[PolicyCondition(metric="latency_p99", operator="gt", threshold=100.0)],
                actions=[HealingAction.ALERT_TEAM],
                priority=10
            )


class TestAnomalyResultModel:
    """Test AnomalyResult model"""
    
    def test_valid_anomaly_result(self):
        """Test creating valid anomaly result"""
        result = AnomalyResult(
            is_anomaly=True,
            confidence=0.85,
            anomaly_score=0.75,
            affected_metrics=["latency", "error_rate"]
        )
        
        assert result.is_anomaly is True
        assert result.confidence == 0.85
        assert isinstance(result.detection_timestamp, datetime)
    
    def test_confidence_bounds(self):
        """Test confidence is bounded 0-1"""
        # Valid
        result = AnomalyResult(
            is_anomaly=True,
            confidence=0.5,
            anomaly_score=0.6
        )
        assert result.confidence == 0.5
        
        # Confidence > 1 should fail
        with pytest.raises(ValidationError):
            AnomalyResult(
                is_anomaly=True,
                confidence=1.5,
                anomaly_score=0.5
            )


class TestForecastResultModel:
    """Test ForecastResult model"""
    
    def test_valid_forecast(self):
        """Test creating valid forecast"""
        result = ForecastResult(
            metric="latency",
            predicted_value=250.0,
            confidence=0.75,
            trend="increasing",
            time_to_threshold=15.5,
            risk_level="high"
        )
        
        assert result.metric == "latency"
        assert result.trend == "increasing"
        assert result.risk_level == "high"
    
    def test_trend_validation(self):
        """Test that only valid trends are accepted"""
        valid_trends = ["increasing", "decreasing", "stable"]
        
        for trend in valid_trends:
            result = ForecastResult(
                metric="latency",
                predicted_value=200.0,
                confidence=0.7,
                trend=trend,
                risk_level="medium"
            )
            assert result.trend == trend
        
        # Invalid trend
        with pytest.raises(ValidationError):
            ForecastResult(
                metric="latency",
                predicted_value=200.0,
                confidence=0.7,
                trend="invalid_trend",
                risk_level="medium"
            )
    
    def test_risk_level_validation(self):
        """Test that only valid risk levels are accepted"""
        valid_levels = ["low", "medium", "high", "critical"]
        
        for level in valid_levels:
            result = ForecastResult(
                metric="error_rate",
                predicted_value=0.08,
                confidence=0.8,
                trend="stable",
                risk_level=level
            )
            assert result.risk_level == level
        
        # Invalid risk level
        with pytest.raises(ValidationError):
            ForecastResult(
                metric="error_rate",
                predicted_value=0.08,
                confidence=0.8,
                trend="stable",
                risk_level="extreme"
            )


class TestTimestampHandling:
    """Test datetime timestamp handling"""
    
    def test_timestamp_is_datetime(self):
        """Test that timestamp is datetime, not string"""
        event = ReliabilityEvent(
            component="test-service",
            latency_p99=100.0,
            error_rate=0.05,
            throughput=1000.0
        )
        
        # Should be datetime object
        assert isinstance(event.timestamp, datetime)
        
        # Should have timezone
        assert event.timestamp.tzinfo is not None
    
    def test_timestamp_is_utc(self):
        """Test that timestamp uses UTC"""
        event = ReliabilityEvent(
            component="test-service",
            latency_p99=100.0,
            error_rate=0.05,
            throughput=1000.0
        )
        
        assert event.timestamp.tzinfo == timezone.utc
    
    def test_timestamp_serialization(self):
        """Test that timestamp can be serialized"""
        event = ReliabilityEvent(
            component="test-service",
            latency_p99=100.0,
            error_rate=0.05,
            throughput=1000.0
        )
        
        # Can convert to ISO format
        iso_str = event.timestamp.isoformat()
        assert isinstance(iso_str, str)
        assert 'T' in iso_str  # ISO format


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
"""
Unit tests for PolicyEngine with thread safety and concurrency tests
"""

import pytest
import threading
import time
from datetime import datetime, timezone
from models import ReliabilityEvent, EventSeverity, HealingPolicy, HealingAction, PolicyCondition
from healing_policies import PolicyEngine


class TestPolicyEngineBasics:
    """Basic policy engine functionality tests"""
    
    def test_initialization(self, policy_engine):
        """Test policy engine initializes correctly"""
        assert policy_engine is not None
        assert len(policy_engine.policies) > 0
        assert policy_engine.max_cooldown_history == 100
    
    def test_policy_evaluation_no_match(self, policy_engine, normal_event):
        """Test that normal events don't trigger policies"""
        actions = policy_engine.evaluate_policies(normal_event)
        assert actions == [HealingAction.NO_ACTION]
    
    def test_policy_evaluation_match(self, policy_engine, critical_event):
        """Test that critical events trigger policies"""
        actions = policy_engine.evaluate_policies(critical_event)
        assert len(actions) > 0
        assert HealingAction.NO_ACTION not in actions
    
    def test_policy_disabled(self, sample_policy, sample_event):
        """Test that disabled policies don't execute"""
        disabled_policy = sample_policy.model_copy(update={'enabled': False})
        engine = PolicyEngine(policies=[disabled_policy])
        
        actions = engine.evaluate_policies(sample_event)
        assert actions == [HealingAction.NO_ACTION]


class TestPolicyCooldown:
    """Test cooldown mechanism"""
    
    def test_cooldown_prevents_immediate_re_execution(self, sample_policy, sample_event):
        """Test that cooldown prevents immediate re-execution"""
        policy = sample_policy.model_copy(update={'cool_down_seconds': 60})
        engine = PolicyEngine(policies=[policy])
        
        # First execution should work
        actions1 = engine.evaluate_policies(sample_event)
        assert HealingAction.RESTART_CONTAINER in actions1
        
        # Second execution should be blocked by cooldown
        actions2 = engine.evaluate_policies(sample_event)
        assert actions2 == [HealingAction.NO_ACTION]
    
    def test_cooldown_expires(self, sample_policy, sample_event):
        """Test that actions work again after cooldown expires"""
        policy = sample_policy.model_copy(update={'cool_down_seconds': 1})
        engine = PolicyEngine(policies=[policy])
        
        # First execution
        actions1 = engine.evaluate_policies(sample_event)
        assert HealingAction.RESTART_CONTAINER in actions1
        
        # Wait for cooldown to expire
        time.sleep(1.1)
        
        # Should work again
        actions2 = engine.evaluate_policies(sample_event)
        assert HealingAction.RESTART_CONTAINER in actions2


class TestRateLimiting:
    """Test rate limiting functionality"""
    
    def test_rate_limit_enforcement(self, sample_policy, sample_event):
        """Test that rate limiting prevents excessive executions"""
        policy = sample_policy.model_copy(update={
            'cool_down_seconds': 0,  # No cooldown
            'max_executions_per_hour': 3
        })
        engine = PolicyEngine(policies=[policy])
        
        # Execute 3 times (should all work)
        for i in range(3):
            actions = engine.evaluate_policies(sample_event)
            assert HealingAction.RESTART_CONTAINER in actions
            time.sleep(0.1)  # Small delay to avoid race
        
        # 4th execution should be rate limited
        actions = engine.evaluate_policies(sample_event)
        assert actions == [HealingAction.NO_ACTION]


class TestThreadSafety:
    """Test thread safety of policy engine"""
    
    def test_concurrent_evaluations_no_race_condition(self, sample_policy, sample_event):
        """
        CRITICAL TEST: Verify no race condition in cooldown check
        
        This tests the fix for the race condition where multiple threads
        could simultaneously pass the cooldown check
        """
        policy = sample_policy.model_copy(update={'cool_down_seconds': 5})
        engine = PolicyEngine(policies=[policy])
        
        results = []
        
        def evaluate():
            actions = engine.evaluate_policies(sample_event)
            results.append(actions)
        
        # Launch 10 concurrent threads
        threads = [threading.Thread(target=evaluate) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # Count how many actually triggered the policy
        trigger_count = sum(
            1 for actions in results
            if HealingAction.RESTART_CONTAINER in actions
        )
        
        # Only ONE should have triggered (atomic check + update)
        assert trigger_count == 1, f"Expected 1 trigger, got {trigger_count}"
    
    def test_concurrent_different_components(self, sample_policy):
        """Test that different components don't interfere with each other"""
        engine = PolicyEngine(policies=[sample_policy])
        
        results = {'service-1': [], 'service-2': []}
        
        def evaluate_service(service_name):
            event = ReliabilityEvent(
                component=service_name,
                latency_p99=400.0,
                error_rate=0.11,
                throughput=1000.0
            )
            actions = engine.evaluate_policies(event)
            results[service_name].append(actions)
        
        # Run both services concurrently multiple times
        threads = []
        for _ in range(5):
            threads.append(threading.Thread(target=evaluate_service, args=('service-1',)))
            threads.append(threading.Thread(target=evaluate_service, args=('service-2',)))
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # Each service should have triggered at least once
        assert any(HealingAction.RESTART_CONTAINER in actions 
                   for actions in results['service-1'])
        assert any(HealingAction.RESTART_CONTAINER in actions 
                   for actions in results['service-2'])


class TestMemoryManagement:
    """Test memory leak prevention"""
    
    def test_cooldown_history_bounded(self, sample_policy):
        """Test that cooldown history doesn't grow unbounded"""
        engine = PolicyEngine(
            policies=[sample_policy],
            max_cooldown_history=100
        )
        
        # Trigger policy for many different components
        for i in range(500):
            event = ReliabilityEvent(
                component=f"service-{i}",
                latency_p99=400.0,
                error_rate=0.11,
                throughput=1000.0
            )
            engine.evaluate_policies(event)
        
        # Cooldown history should be capped
        assert len(engine.last_execution) <= engine.max_cooldown_history
    
    def test_execution_history_bounded(self, sample_policy):
        """Test that execution history is bounded"""
        engine = PolicyEngine(
            policies=[sample_policy],
            max_execution_history=50
        )
        
        # Trigger many times
        for i in range(200):
            event = ReliabilityEvent(
                component="test-service",
                latency_p99=400.0,
                error_rate=0.11,
                throughput=1000.0
            )
            engine.evaluate_policies(event)
            time.sleep(0.01)
        
        # Check execution history size
        for timestamps in engine.execution_timestamps.values():
            assert len(timestamps) <= engine.max_execution_history


class TestPriorityHandling:
    """Test priority-based policy evaluation"""
    
    def test_policies_evaluated_by_priority(self):
        """Test that higher priority policies are evaluated first"""
        high_priority = HealingPolicy(
            name="high_priority",
            conditions=[PolicyCondition(metric="latency_p99", operator="gt", threshold=100.0)],
            actions=[HealingAction.ROLLBACK],
            priority=1
        )
        
        low_priority = HealingPolicy(
            name="low_priority",
            conditions=[PolicyCondition(metric="latency_p99", operator="gt", threshold=100.0)],
            actions=[HealingAction.ALERT_TEAM],
            priority=5
        )
        
        # Add in reverse priority order
        engine = PolicyEngine(policies=[low_priority, high_priority])
        
        event = ReliabilityEvent(
            component="test",
            latency_p99=200.0,
            error_rate=0.05,
            throughput=1000.0
        )
        
        actions = engine.evaluate_policies(event)
        
        # Both should execute, but high priority action should come first
        assert HealingAction.ROLLBACK in actions
        assert HealingAction.ALERT_TEAM in actions
        assert actions.index(HealingAction.ROLLBACK) < actions.index(HealingAction.ALERT_TEAM)


class TestOperatorComparisons:
    """Test operator comparison logic"""
    
    def test_greater_than_operator(self, policy_engine):
        """Test > operator"""
        result = policy_engine._compare_values(100.0, "gt", 50.0)
        assert result is True
        
        result = policy_engine._compare_values(50.0, "gt", 100.0)
        assert result is False
    
    def test_less_than_operator(self, policy_engine):
        """Test < operator"""
        result = policy_engine._compare_values(50.0, "lt", 100.0)
        assert result is True
        
        result = policy_engine._compare_values(100.0, "lt", 50.0)
        assert result is False
    
    def test_type_mismatch_handling(self, policy_engine):
        """Test that type mismatches are handled gracefully"""
        result = policy_engine._compare_values("invalid", "gt", 50.0)
        assert result is False
    
    def test_none_value_handling(self, sample_policy):
        """Test that None values are handled correctly"""
        engine = PolicyEngine(policies=[sample_policy])
        
        event = ReliabilityEvent(
            component="test",
            latency_p99=100.0,
            error_rate=0.05,
            throughput=1000.0,
            cpu_util=None  # None value
        )
        
        # Should not crash
        actions = engine.evaluate_policies(event)
        assert actions is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
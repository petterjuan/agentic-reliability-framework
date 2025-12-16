"""
Policy Engine for Automated Healing Actions
Fixed version with thread safety and memory leak prevention
"""

import datetime
import threading
import logging
from collections import OrderedDict
from typing import Dict, List, Optional, Any, cast
from .models import HealingPolicy, HealingAction, ReliabilityEvent, PolicyCondition

logger = logging.getLogger(__name__)


# Default healing policies with structured conditions
DEFAULT_HEALING_POLICIES = [
    HealingPolicy(
        name="high_latency_restart",
        conditions=[
            PolicyCondition(metric="latency_p99", operator="gt", threshold=500.0)
        ],
        actions=[HealingAction.RESTART_CONTAINER, HealingAction.ALERT_TEAM],
        priority=1,
        cool_down_seconds=300,
        max_executions_per_hour=5
    ),
    HealingPolicy(
        name="critical_error_rate_rollback",
        conditions=[
            PolicyCondition(metric="error_rate", operator="gt", threshold=0.3)
        ],
        actions=[HealingAction.ROLLBACK, HealingAction.CIRCUIT_BREAKER, HealingAction.ALERT_TEAM],
        priority=1,
        cool_down_seconds=600,
        max_executions_per_hour=3
    ),
    HealingPolicy(
        name="high_error_rate_traffic_shift",
        conditions=[
            PolicyCondition(metric="error_rate", operator="gt", threshold=0.15)
        ],
        actions=[HealingAction.TRAFFIC_SHIFT, HealingAction.ALERT_TEAM],
        priority=2,
        cool_down_seconds=300,
        max_executions_per_hour=5
    ),
    HealingPolicy(
        name="resource_exhaustion_scale",
        conditions=[
            PolicyCondition(metric="cpu_util", operator="gt", threshold=0.9),
            PolicyCondition(metric="memory_util", operator="gt", threshold=0.9)
        ],
        actions=[HealingAction.SCALE_OUT],
        priority=2,
        cool_down_seconds=600,
        max_executions_per_hour=10
    ),
    HealingPolicy(
        name="moderate_latency_circuit_breaker",
        conditions=[
            PolicyCondition(metric="latency_p99", operator="gt", threshold=300.0)
        ],
        actions=[HealingAction.CIRCUIT_BREAKER],
        priority=3,
        cool_down_seconds=180,
        max_executions_per_hour=8
    )
]


class PolicyEngine:
    """
    Thread-safe policy engine with cooldown and rate limiting
    
    CRITICAL FIXES:
    - Added RLock for thread safety
    - Fixed cooldown race condition (atomic check + update)
    - Implemented LRU eviction to prevent memory leak
    - Added priority-based policy evaluation
    - Added rate limiting per policy
    """
    
    def __init__(
        self,
        policies: Optional[List[HealingPolicy]] = None,
        max_cooldown_history: int = 100,
        max_execution_history: int = 1000
    ) -> None:
        """
        Initialize policy engine
        
        Args:
            policies: List of healing policies (uses defaults if None)
            max_cooldown_history: Maximum cooldown entries to keep (LRU)
            max_execution_history: Maximum execution history per policy
        """
        self.policies = policies or DEFAULT_HEALING_POLICIES
        
        # FIXED: Added RLock for thread safety
        self._lock = threading.RLock()
        
        # FIXED: Use OrderedDict for LRU eviction (prevents memory leak)
        self.last_execution: OrderedDict[str, float] = OrderedDict()
        self.max_cooldown_history = max_cooldown_history
        
        # Rate limiting: track executions per hour per policy
        self.execution_timestamps: Dict[str, List[float]] = {}
        self.max_execution_history = max_execution_history
        
        # Sort policies by priority (lower number = higher priority)
        self.policies = sorted(self.policies, key=lambda p: p.priority)
        
        logger.info(
            f"Initialized PolicyEngine with {len(self.policies)} policies, "
            f"max_cooldown_history={max_cooldown_history}"
        )
    
    def evaluate_policies(self, event: ReliabilityEvent) -> List[HealingAction]:
        """
        Evaluate all policies against the event and return matching actions
        
        FIXED: Atomic check + update under lock (prevents race condition)
        FIXED: Priority-based evaluation
        
        Args:
            event: Reliability event to evaluate
            
        Returns:
            List of healing actions to execute
        """
        applicable_actions: List[HealingAction] = []
        current_time = datetime.datetime.now(datetime.timezone.utc).timestamp()
        
        # Evaluate policies in priority order
        for policy in self.policies:
            if not policy.enabled:
                continue
            
            policy_key = f"{policy.name}_{event.component}"
            
            # FIXED: All cooldown operations under lock (atomic)
            with self._lock:
                # Check cooldown
                last_exec = self.last_execution.get(policy_key, 0)
                
                if current_time - last_exec < policy.cool_down_seconds:
                    logger.debug(
                        f"Policy {policy.name} for {event.component} on cooldown "
                        f"({current_time - last_exec:.0f}s / {policy.cool_down_seconds}s)"
                    )
                    continue
                
                # Check rate limit
                if self._is_rate_limited(policy_key, policy, current_time):
                    logger.warning(
                        f"Policy {policy.name} for {event.component} rate limited "
                        f"(max {policy.max_executions_per_hour}/hour)"
                    )
                    continue
                
                # Evaluate conditions first (FIX: only update if conditions match)
                should_execute = self._evaluate_conditions(policy.conditions, event)
                
                if should_execute:
                    applicable_actions.extend(policy.actions)
                    
                    # Update cooldown timestamp (INSIDE lock, AFTER condition check)
                    self._update_cooldown(policy_key, current_time)
                    
                    # Track execution for rate limiting
                    self._record_execution(policy_key, current_time)
                    
                    logger.info(
                        f"Policy {policy.name} triggered for {event.component}: "
                        f"actions={[a.value for a in policy.actions]}"
                    )
        
        # Deduplicate actions while preserving order
        seen = set()
        unique_actions: List[HealingAction] = []
        for action in applicable_actions:
            if action not in seen:
                seen.add(action)
                unique_actions.append(action)
        
        return unique_actions if unique_actions else [HealingAction.NO_ACTION]
    
    def _evaluate_conditions(
        self,
        conditions: List[PolicyCondition],
        event: ReliabilityEvent
    ) -> bool:
        """
        Evaluate all conditions against event (AND logic)
        
        Args:
            conditions: List of policy conditions
            event: Reliability event
            
        Returns:
            True if all conditions match, False otherwise
        """
        for condition in conditions:
            # Get event value
            event_value = getattr(event, condition.metric, None)
            
            # Handle None values
            if event_value is None:
                logger.debug(
                    f"Condition failed: {condition.metric} is None on event"
                )
                return False
            
            # Evaluate operator
            if not self._compare_values(
                float(event_value),  # Ensure float type
                condition.operator,
                condition.threshold
            ):
                logger.debug(
                    f"Condition failed: {event_value} {condition.operator} "
                    f"{condition.threshold} = False"
                )
                return False
        
        return True
    
    def _compare_values(
        self,
        event_value: float,
        operator: str,
        threshold: float
    ) -> bool:
        """
        Compare values based on operator with type safety
        
        FIXED: Added type checking and better error handling
        
        Args:
            event_value: Value from event
            operator: Comparison operator
            threshold: Threshold value
            
        Returns:
            Comparison result
        """
        try:
            # Operator evaluation
            if operator == "gt":
                return event_value > threshold
            elif operator == "lt":
                return event_value < threshold
            elif operator == "eq":
                return abs(event_value - threshold) < 1e-6  # Float equality
            elif operator == "gte":
                return event_value >= threshold
            elif operator == "lte":
                return event_value <= threshold
            else:
                logger.error(f"Unknown operator: {operator}")
                return False
                
        except (TypeError, ValueError) as e:
            logger.error(f"Comparison error: {e}", exc_info=True)
            return False
    
    def _update_cooldown(self, policy_key: str, timestamp: float) -> None:
        """
        Update cooldown timestamp with LRU eviction
        
        FIXED: Prevents unbounded memory growth
        
        Args:
            policy_key: Policy identifier
            timestamp: Current timestamp
        """
        # Update timestamp
        self.last_execution[policy_key] = timestamp
        
        # Move to end (most recently used)
        self.last_execution.move_to_end(policy_key)
        
        # LRU eviction if too large
        while len(self.last_execution) > self.max_cooldown_history:
            old_key = next(iter(self.last_execution))
            self.last_execution.popitem(last=False)
            logger.debug(f"Evicted cooldown entry: {old_key}")
    
    def _is_rate_limited(
        self,
        policy_key: str,
        policy: HealingPolicy,
        current_time: float
    ) -> bool:
        """
        Check if policy is rate limited
        
        Args:
            policy_key: Policy identifier
            policy: Policy configuration
            current_time: Current timestamp
            
        Returns:
            True if rate limited, False otherwise
        """
        if policy_key not in self.execution_timestamps:
            return False
        
        # Remove executions older than 1 hour
        one_hour_ago = current_time - 3600
        timestamps = self.execution_timestamps[policy_key]
        recent_executions = [
            ts for ts in timestamps
            if ts > one_hour_ago
        ]
        
        self.execution_timestamps[policy_key] = recent_executions
        
        # Check rate limit
        return len(recent_executions) >= policy.max_executions_per_hour
    
    def _record_execution(self, policy_key: str, timestamp: float) -> None:
        """
        Record policy execution for rate limiting
        
        Args:
            policy_key: Policy identifier
            timestamp: Execution timestamp
        """
        if policy_key not in self.execution_timestamps:
            self.execution_timestamps[policy_key] = []
        
        self.execution_timestamps[policy_key].append(timestamp)
        
        # Limit history size (memory management)
        if len(self.execution_timestamps[policy_key]) > self.max_execution_history:
            # Keep only the most recent entries
            timestamps = self.execution_timestamps[policy_key]
            self.execution_timestamps[policy_key] = \
                timestamps[-self.max_execution_history:]
    
    def get_policy_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics about policy execution
        
        Returns:
            Dictionary of policy statistics
        """
        with self._lock:
            stats: Dict[str, Dict[str, Any]] = {}
            
            for policy in self.policies:
                # Count components for this policy
                total_components = 0
                for key in self.last_execution.keys():
                    if key.startswith(f"{policy.name}_"):
                        total_components += 1
                
                policy_stats = {
                    "name": policy.name,
                    "priority": policy.priority,
                    "enabled": policy.enabled,
                    "cooldown_seconds": policy.cool_down_seconds,
                    "max_per_hour": policy.max_executions_per_hour,
                    "total_components": total_components
                }
                
                stats[policy.name] = policy_stats
            
            return stats


# Helper function to create a default policy engine
def create_default_policy_engine() -> PolicyEngine:
    """
    Create a default policy engine with standard policies
    
    Returns:
        PolicyEngine instance
    """
    return PolicyEngine()


# Helper function to test if a policy would trigger for an event
def would_policy_trigger(
    policy: HealingPolicy,
    event: ReliabilityEvent,
    last_execution_time: Optional[float] = None,
    execution_count_last_hour: int = 0
) -> bool:
    """
    Test if a policy would trigger for an event without actually executing it
    
    Args:
        policy: The policy to test
        event: The event to test against
        last_execution_time: Optional last execution time (for cooldown check)
        execution_count_last_hour: Number of executions in last hour (for rate limit)
        
    Returns:
        True if policy would trigger, False otherwise
    """
    # Check if policy is enabled
    if not policy.enabled:
        return False
    
    # Check cooldown if last_execution_time provided
    if last_execution_time is not None:
        current_time = datetime.datetime.now(datetime.timezone.utc).timestamp()
        if current_time - last_execution_time < policy.cool_down_seconds:
            return False
    
    # Check rate limit if execution count provided
    if execution_count_last_hour >= policy.max_executions_per_hour:
        return False
    
    # Check conditions
    engine = PolicyEngine(policies=[policy], max_cooldown_history=0)
    
    # Use a temporary evaluation (not thread-safe, but okay for testing)
    # We need to access the protected method
    for condition in policy.conditions:
        # Get event value
        event_value = getattr(event, condition.metric, None)
        
        # Handle None values
        if event_value is None:
            return False
        
        # Evaluate operator
        if not engine._compare_values(
            float(event_value),
            condition.operator,
            condition.threshold
        ):
            return False
    
    return True

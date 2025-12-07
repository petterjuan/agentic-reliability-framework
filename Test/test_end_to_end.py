"""
End-to-end integration tests for complete ARF workflow

Save this as: tests/test_end_to_end.py
"""

import pytest
import asyncio
from datetime import datetime
from models import SystemEvent, AgentSynthesis
from app import (
    ReliabilityCore,
    DetectiveAgent,
    DiagnosticianAgent,
    PredictiveAgent,
    PolicyEngine,
    FAISSIncidentMemory
)


class TestEndToEndWorkflow:
    """Test complete ARF workflow from event to healing action"""
    
    @pytest.fixture
    def reliability_core(self):
        """Create ReliabilityCore instance with all agents"""
        return ReliabilityCore()
    
    @pytest.fixture
    def policy_engine(self):
        """Create PolicyEngine instance"""
        return PolicyEngine()
    
    @pytest.mark.asyncio
    async def test_complete_incident_workflow(self, reliability_core, policy_engine):
        """Test full workflow: event → detection → diagnosis → prediction → healing"""
        # 1. Create critical event
        critical_event = SystemEvent(
            service_name="api",
            timestamp=datetime.now(),
            latency_ms=750.0,  # High latency
            error_rate=0.28,  # High errors
            error_message="database connection pool exhausted",
            cpu_percent=85.0,
            memory_percent=80.0,
            throughput=400
        )
        
        # 2. Analyze with all agents
        synthesis = await reliability_core.analyze_event(critical_event)
        
        # Verify synthesis
        assert isinstance(synthesis, AgentSynthesis)
        assert synthesis.consensus_confidence > 0.6, "Should detect serious issue"
        assert len(synthesis.recommended_action) > 0
        
        # 3. Check if healing is needed
        healing_actions = policy_engine.evaluate_and_execute(critical_event, synthesis)
        
        # Verify healing actions were triggered
        assert len(healing_actions) > 0, "Should trigger healing actions"
        assert any(action in str(healing_actions) for action in ["restart", "scale", "rollback"])
    
    @pytest.mark.asyncio
    async def test_normal_event_no_action(self, reliability_core, policy_engine):
        """Test that normal events don't trigger unnecessary healing"""
        # Create normal event
        normal_event = SystemEvent(
            service_name="api",
            timestamp=datetime.now(),
            latency_ms=100.0,
            error_rate=0.01,
            cpu_percent=45.0,
            memory_percent=60.0,
            throughput=1000
        )
        
        # Analyze
        synthesis = await reliability_core.analyze_event(normal_event)
        
        # Should have low confidence
        assert synthesis.consensus_confidence < 0.5, "Normal event should have low confidence"
        
        # Should not trigger healing
        healing_actions = policy_engine.evaluate_and_execute(normal_event, synthesis)
        assert len(healing_actions) == 0, "Normal event should not trigger healing"
    
    @pytest.mark.asyncio
    async def test_memory_helps_future_incidents(self):
        """Test that FAISS memory improves handling of similar incidents"""
        memory = FAISSIncidentMemory()
        
        # 1. Add historical incident to memory
        await memory.add_incident(
            "database connection pool exhausted causing 500ms latency",
            {
                "root_cause": "db pool size too small",
                "resolution": "increased pool from 10 to 50 connections",
                "resolved": True
            }
        )
        
        # 2. New similar incident occurs
        current_event_description = "high latency and database connection errors"
        
        # 3. Search memory for similar incidents
        similar_incidents = await memory.search_similar(current_event_description, k=3)
        
        # 4. Verify memory recall
        assert len(similar_incidents) > 0, "Should find similar historical incident"
        assert similar_incidents[0]['score'] > 0.5, "Should be reasonably similar"
        
        # 5. Memory metadata should help with resolution
        top_match = similar_incidents[0]
        assert 'root_cause' in top_match
        assert 'resolution' in top_match
        
        # This proves: Past incidents inform future responses


class TestMultiServiceMonitoring:
    """Test monitoring multiple services simultaneously"""
    
    @pytest.fixture
    def reliability_core(self):
        return ReliabilityCore()
    
    @pytest.mark.asyncio
    async def test_monitor_multiple_services_in_parallel(self, reliability_core):
        """Test that ARF can monitor multiple services concurrently"""
        # Create events from different services
        api_event = SystemEvent(
            service_name="api",
            timestamp=datetime.now(),
            latency_ms=300.0,
            error_rate=0.10,
            cpu_percent=70.0,
            memory_percent=65.0,
            throughput=800
        )
        
        worker_event = SystemEvent(
            service_name="worker",
            timestamp=datetime.now(),
            latency_ms=150.0,
            error_rate=0.05,
            cpu_percent=85.0,
            memory_percent=90.0,  # High memory
            throughput=500
        )
        
        db_event = SystemEvent(
            service_name="database",
            timestamp=datetime.now(),
            latency_ms=50.0,
            error_rate=0.02,
            cpu_percent=60.0,
            memory_percent=70.0,
            throughput=2000
        )
        
        # Analyze all services in parallel
        results = await asyncio.gather(
            reliability_core.analyze_event(api_event),
            reliability_core.analyze_event(worker_event),
            reliability_core.analyze_event(db_event)
        )
        
        # All should complete
        assert len(results) == 3
        assert all(isinstance(r, AgentSynthesis) for r in results)
        
        # Worker should have highest concern (high memory)
        worker_result = results[1]
        assert worker_result.consensus_confidence > 0.5, \
            "Worker with high memory should be flagged"


class TestPolicyEngineCooldown:
    """Test that PolicyEngine respects cooldown periods"""
    
    @pytest.fixture
    def policy_engine(self):
        return PolicyEngine()
    
    @pytest.mark.asyncio
    async def test_policy_cooldown_prevents_rapid_execution(self, policy_engine):
        """Test that policies don't execute repeatedly within cooldown"""
        reliability_core = ReliabilityCore()
        
        # Create event that triggers high latency policy
        high_latency_event = SystemEvent(
            service_name="api",
            timestamp=datetime.now(),
            latency_ms=600.0,  # Above threshold
            error_rate=0.05,
            cpu_percent=60.0,
            memory_percent=65.0,
            throughput=700
        )
        
        # First execution - should trigger
        synthesis1 = await reliability_core.analyze_event(high_latency_event)
        actions1 = policy_engine.evaluate_and_execute(high_latency_event, synthesis1)
        
        assert len(actions1) > 0, "First execution should trigger policy"
        
        # Immediate second execution - should be blocked by cooldown
        synthesis2 = await reliability_core.analyze_event(high_latency_event)
        actions2 = policy_engine.evaluate_and_execute(high_latency_event, synthesis2)
        
        assert len(actions2) == 0, "Second execution should be blocked by cooldown"


class TestAgentSynthesisConsensus:
    """Test agent consensus and synthesis logic"""
    
    @pytest.fixture
    def reliability_core(self):
        return ReliabilityCore()
    
    @pytest.mark.asyncio
    async def test_consensus_confidence_calculation(self, reliability_core):
        """Test that consensus confidence is calculated correctly"""
        event = SystemEvent(
            service_name="api",
            timestamp=datetime.now(),
            latency_ms=400.0,
            error_rate=0.15,
            cpu_percent=75.0,
            memory_percent=70.0,
            throughput=600
        )
        
        synthesis = await reliability_core.analyze_event(event)
        
        # Consensus confidence should be average of agent confidences
        assert 0.0 <= synthesis.consensus_confidence <= 1.0
        assert synthesis.consensus_confidence > 0.3, "Should detect moderate issue"
    
    @pytest.mark.asyncio
    async def test_severity_classification(self, reliability_core):
        """Test that severity is classified correctly"""
        # Critical event
        critical_event = SystemEvent(
            service_name="api",
            timestamp=datetime.now(),
            latency_ms=900.0,
            error_rate=0.40,
            cpu_percent=95.0,
            memory_percent=94.0,
            throughput=150
        )
        
        synthesis = await reliability_core.analyze_event(critical_event)
        
        assert synthesis.severity in ["critical", "high"], \
            "Severe event should be classified as critical or high"


class TestBusinessImpactCalculation:
    """Test business impact calculation logic"""
    
    @pytest.fixture
    def reliability_core(self):
        return ReliabilityCore()
    
    @pytest.mark.asyncio
    async def test_business_impact_increases_with_severity(self, reliability_core):
        """Test that business impact correlates with severity"""
        # Moderate event
        moderate_event = SystemEvent(
            service_name="api",
            timestamp=datetime.now(),
            latency_ms=250.0,
            error_rate=0.08,
            cpu_percent=65.0,
            memory_percent=60.0,
            throughput=800
        )
        
        # Critical event
        critical_event = SystemEvent(
            service_name="api",
            timestamp=datetime.now(),
            latency_ms=800.0,
            error_rate=0.35,
            cpu_percent=95.0,
            memory_percent=92.0,
            throughput=200
        )
        
        moderate_synthesis = await reliability_core.analyze_event(moderate_event)
        critical_synthesis = await reliability_core.analyze_event(critical_event)
        
        # Critical should have higher business impact
        assert critical_synthesis.business_impact > moderate_synthesis.business_impact, \
            "Critical incident should have higher business impact"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

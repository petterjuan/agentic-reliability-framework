"""
Test suite for the three core reliability agents
Tests anomaly detection, diagnosis, and prediction capabilities

Save this as: tests/test_agents.py
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from models import SystemEvent, AgentAnalysis
from app import DetectiveAgent, DiagnosticianAgent, PredictiveAgent


# ============================================================================
# DETECTIVE AGENT TESTS (Anomaly Detection)
# ============================================================================

class TestDetectiveAgent:
    """Test anomaly detection capabilities"""
    
    @pytest.fixture
    def detective(self):
        """Create detective agent instance"""
        return DetectiveAgent()
    
    @pytest.fixture
    def normal_event(self):
        """Normal baseline event"""
        return SystemEvent(
            service_name="api",
            timestamp=datetime.now(),
            latency_ms=100.0,
            error_rate=0.01,
            cpu_percent=45.0,
            memory_percent=60.0,
            throughput=1000
        )
    
    @pytest.mark.asyncio
    async def test_detective_normal_event_low_confidence(self, detective, normal_event):
        """Detective should have low confidence for normal events"""
        result = await detective.analyze(normal_event)
        
        assert isinstance(result, AgentAnalysis)
        assert result.agent_type == "detective"
        assert result.confidence < 0.5, "Normal event should have low anomaly confidence"
        assert len(result.reasoning) > 0
    
    @pytest.mark.asyncio
    async def test_detective_high_latency_spike(self, detective):
        """Detective should detect latency spikes"""
        spike_event = SystemEvent(
            service_name="api",
            timestamp=datetime.now(),
            latency_ms=800.0,  # 8x normal (100ms baseline)
            error_rate=0.01,
            cpu_percent=45.0,
            memory_percent=60.0,
            throughput=1000
        )
        
        result = await detective.analyze(spike_event)
        
        assert result.confidence > 0.7, "Should detect significant latency spike"
        assert "latency" in result.reasoning.lower()
    
    @pytest.mark.asyncio
    async def test_detective_error_rate_spike(self, detective):
        """Detective should detect error rate increases"""
        error_event = SystemEvent(
            service_name="api",
            timestamp=datetime.now(),
            latency_ms=100.0,
            error_rate=0.35,  # 35% errors (35x baseline of 1%)
            cpu_percent=45.0,
            memory_percent=60.0,
            throughput=1000
        )
        
        result = await detective.analyze(error_event)
        
        assert result.confidence > 0.8, "Should detect severe error rate spike"
        assert "error" in result.reasoning.lower()
    
    @pytest.mark.asyncio
    async def test_detective_resource_exhaustion(self, detective):
        """Detective should detect CPU/memory exhaustion"""
        resource_event = SystemEvent(
            service_name="api",
            timestamp=datetime.now(),
            latency_ms=100.0,
            error_rate=0.01,
            cpu_percent=96.0,  # Near max
            memory_percent=94.0,  # Near max
            throughput=1000
        )
        
        result = await detective.analyze(resource_event)
        
        assert result.confidence > 0.7, "Should detect resource exhaustion"
        assert any(word in result.reasoning.lower() for word in ["cpu", "memory", "resource"])
    
    @pytest.mark.asyncio
    async def test_detective_combined_anomalies(self, detective):
        """Detective should have high confidence for multiple anomalies"""
        multi_event = SystemEvent(
            service_name="api",
            timestamp=datetime.now(),
            latency_ms=600.0,  # High
            error_rate=0.25,  # High
            cpu_percent=92.0,  # High
            memory_percent=88.0,  # High
            throughput=200  # Low (dropped)
        )
        
        result = await detective.analyze(multi_event)
        
        assert result.confidence > 0.85, "Multiple anomalies should have very high confidence"


# ============================================================================
# DIAGNOSTICIAN AGENT TESTS (Root Cause Analysis)
# ============================================================================

class TestDiagnosticianAgent:
    """Test root cause analysis capabilities"""
    
    @pytest.fixture
    def diagnostician(self):
        """Create diagnostician agent instance"""
        return DiagnosticianAgent()
    
    @pytest.mark.asyncio
    async def test_diagnostician_db_pool_exhaustion(self, diagnostician):
        """Diagnostician should identify database connection pool issues"""
        db_event = SystemEvent(
            service_name="api",
            timestamp=datetime.now(),
            latency_ms=450.0,
            error_rate=0.28,
            error_message="connection pool exhausted, waiting for available connection",
            cpu_percent=60.0,
            memory_percent=70.0,
            throughput=800
        )
        
        result = await diagnostician.analyze(db_event)
        
        assert result.confidence > 0.7, "Should identify DB pool exhaustion"
        assert any(word in result.reasoning.lower() for word in ["database", "connection", "pool"])
    
    @pytest.mark.asyncio
    async def test_diagnostician_dependency_timeout(self, diagnostician):
        """Diagnostician should identify downstream dependency timeouts"""
        timeout_event = SystemEvent(
            service_name="api",
            timestamp=datetime.now(),
            latency_ms=5100.0,  # Just over timeout threshold
            error_rate=0.18,
            error_message="timeout calling downstream service: payment-service",
            cpu_percent=45.0,
            memory_percent=60.0,
            throughput=600
        )
        
        result = await diagnostician.analyze(timeout_event)
        
        assert result.confidence > 0.6, "Should identify dependency timeout"
        assert any(word in result.reasoning.lower() for word in ["timeout", "dependency", "downstream"])
    
    @pytest.mark.asyncio
    async def test_diagnostician_memory_leak(self, diagnostician):
        """Diagnostician should identify memory leak patterns"""
        memory_event = SystemEvent(
            service_name="worker",
            timestamp=datetime.now(),
            latency_ms=200.0,
            error_rate=0.05,
            cpu_percent=55.0,
            memory_percent=94.0,  # Very high
            throughput=500
        )
        
        result = await diagnostician.analyze(memory_event)
        
        assert result.confidence > 0.5, "Should identify memory issue"
        assert "memory" in result.reasoning.lower()


# ============================================================================
# PREDICTIVE AGENT TESTS (Forecasting)
# ============================================================================

class TestPredictiveAgent:
    """Test forecasting and prediction capabilities"""
    
    @pytest.fixture
    def predictive(self):
        """Create predictive agent instance"""
        return PredictiveAgent()
    
    def create_trend_events(self, start_value, end_value, metric="latency_ms", count=10):
        """Helper: Create events showing a linear trend"""
        events = []
        now = datetime.now()
        
        for i in range(count):
            value = start_value + (end_value - start_value) * (i / count)
            
            event = SystemEvent(
                service_name="api",
                timestamp=now - timedelta(minutes=(count - i) * 5),
                latency_ms=value if metric == "latency_ms" else 100.0,
                error_rate=value if metric == "error_rate" else 0.01,
                cpu_percent=value if metric == "cpu_percent" else 50.0,
                memory_percent=value if metric == "memory_percent" else 60.0,
                throughput=int(1000 - value) if metric == "throughput" else 1000
            )
            events.append(event)
        
        return events
    
    @pytest.mark.asyncio
    async def test_predictive_increasing_latency_trend(self, predictive):
        """Predictive should forecast increasing latency"""
        # Latency climbing from 100ms to 400ms
        events = self.create_trend_events(100.0, 400.0, metric="latency_ms")
        
        result = await predictive.analyze(events)
        
        assert result.confidence > 0.6, "Should detect increasing latency trend"
        assert "trend" in result.reasoning.lower() or "increas" in result.reasoning.lower()
    
    @pytest.mark.asyncio
    async def test_predictive_memory_leak_forecast(self, predictive):
        """Predictive should forecast memory exhaustion"""
        # Memory climbing from 60% to 92%
        events = self.create_trend_events(60.0, 92.0, metric="memory_percent")
        
        result = await predictive.analyze(events)
        
        assert result.confidence > 0.7, "Should predict memory exhaustion"
        assert "memory" in result.reasoning.lower()
    
    @pytest.mark.asyncio
    async def test_predictive_stable_metrics_low_risk(self, predictive):
        """Predictive should report low risk for stable metrics"""
        # Stable latency around 100ms
        events = self.create_trend_events(95.0, 105.0, metric="latency_ms")
        
        result = await predictive.analyze(events)
        
        assert result.confidence < 0.5, "Stable metrics should have low predicted risk"


# ============================================================================
# AGENT PARALLEL EXECUTION TEST
# ============================================================================

class TestAgentParallelExecution:
    """Test that agents can run in parallel"""
    
    @pytest.mark.asyncio
    async def test_agents_execute_in_parallel(self):
        """All three agents should execute concurrently without blocking"""
        detective = DetectiveAgent()
        diagnostician = DiagnosticianAgent()
        predictive = PredictiveAgent()
        
        event = SystemEvent(
            service_name="api",
            timestamp=datetime.now(),
            latency_ms=300.0,
            error_rate=0.10,
            cpu_percent=70.0,
            memory_percent=75.0,
            throughput=700
        )
        
        trend_events = [event] * 5
        
        # Execute all agents in parallel
        start_time = datetime.now()
        
        results = await asyncio.gather(
            detective.analyze(event),
            diagnostician.analyze(event),
            predictive.analyze(trend_events)
        )
        
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        # All should complete
        assert len(results) == 3
        assert all(isinstance(r, AgentAnalysis) for r in results)
        
        # Should be fast (parallel execution)
        assert execution_time < 5.0, "Parallel execution should complete quickly"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

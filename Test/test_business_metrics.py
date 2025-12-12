"""
Tests for BusinessMetricsTracker
"""

import pytest


@pytest.mark.unit
def test_business_metrics_initialization():
    """Test BusinessMetricsTracker initialization"""
    from agentic_reliability_framework.app import BusinessMetricsTracker
    
    tracker = BusinessMetricsTracker()
    
    assert tracker.total_incidents == 0
    assert tracker.incidents_auto_healed == 0
    assert tracker.total_revenue_saved == 0.0
    assert tracker.total_revenue_at_risk == 0.0
    assert tracker.detection_times == []


@pytest.mark.unit
def test_record_incident():
    """Test recording an incident"""
    from agentic_reliability_framework.app import BusinessMetricsTracker
    
    tracker = BusinessMetricsTracker()
    
    tracker.record_incident(
        severity="HIGH",
        auto_healed=True,
        revenue_loss=100.0,
        detection_time_seconds=120.0
    )
    
    assert tracker.total_incidents == 1
    assert tracker.incidents_auto_healed == 1
    assert tracker.total_revenue_saved > 0
    assert len(tracker.detection_times) == 1


@pytest.mark.unit
def test_get_metrics():
    """Test getting metrics"""
    from agentic_reliability_framework.app import BusinessMetricsTracker
    
    tracker = BusinessMetricsTracker()
    
    # Record some incidents
    tracker.record_incident("HIGH", True, 100.0, 120.0)
    tracker.record_incident("MEDIUM", False, 50.0, 180.0)
    
    metrics = tracker.get_metrics()
    
    assert metrics["total_incidents"] == 2
    assert metrics["incidents_auto_healed"] == 1
    assert metrics["auto_heal_rate"] == 50.0  # 1 out of 2 auto-healed
    assert "total_revenue_saved" in metrics
    assert "avg_detection_time_seconds" in metrics


@pytest.mark.unit
def test_reset_metrics():
    """Test resetting metrics"""
    from agentic_reliability_framework.app import BusinessMetricsTracker
    
    tracker = BusinessMetricsTracker()
    
    tracker.record_incident("HIGH", True, 100.0, 120.0)
    assert tracker.total_incidents == 1
    
    tracker.reset()
    
    assert tracker.total_incidents == 0
    assert tracker.incidents_auto_healed == 0
    assert tracker.total_revenue_saved == 0.0
    assert tracker.total_revenue_at_risk == 0.0
    assert tracker.detection_times == []


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "unit"])

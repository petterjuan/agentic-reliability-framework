# Test/test_business_metrics.py
import pytest

# Fix the import path - BusinessMetricsTracker is in engine.business, not app
from agentic_reliability_framework.engine.business import BusinessMetricsTracker


def test_business_metrics_initialization():
    """Test that business metrics tracker initializes correctly"""
    tracker = BusinessMetricsTracker()
    assert tracker is not None
    assert tracker.total_incidents == 0
    assert tracker.incidents_auto_healed == 0
    assert tracker.total_revenue_saved == 0.0
    assert tracker.total_revenue_at_risk == 0.0
    assert tracker.detection_times == []


def test_record_incident():
    """Test recording incidents"""
    tracker = BusinessMetricsTracker()
    
    # Record first incident
    tracker.record_incident(
        severity="HIGH",
        auto_healed=True,
        revenue_loss=100.0,
        detection_time_seconds=60.0
    )
    
    assert tracker.total_incidents == 1
    assert tracker.incidents_auto_healed == 1
    # Revenue calculations should be positive
    assert tracker.total_revenue_saved > 0
    assert tracker.total_revenue_at_risk > 0
    assert len(tracker.detection_times) == 1
    assert tracker.detection_times[0] == 60.0
    
    # Record second incident (not auto-healed)
    tracker.record_incident(
        severity="MEDIUM",
        auto_healed=False,
        revenue_loss=50.0,
        detection_time_seconds=90.0
    )
    
    assert tracker.total_incidents == 2
    assert tracker.incidents_auto_healed == 1  # Still 1
    assert len(tracker.detection_times) == 2
    assert tracker.detection_times[1] == 90.0


def test_get_metrics():
    """Test getting metrics"""
    tracker = BusinessMetricsTracker()
    
    # Add some incidents
    tracker.record_incident("HIGH", True, 100.0, 60.0)
    tracker.record_incident("MEDIUM", False, 50.0, 90.0)
    tracker.record_incident("LOW", True, 25.0, 120.0)
    
    metrics = tracker.get_metrics()
    
    assert metrics["total_incidents"] == 3
    assert metrics["incidents_auto_healed"] == 2
    assert metrics["auto_heal_rate"] == pytest.approx(66.67, rel=0.1)
    assert metrics["total_revenue_saved"] > 0
    assert metrics["total_revenue_at_risk"] > 0
    assert metrics["avg_detection_time_seconds"] == pytest.approx(90.0, rel=0.1)
    assert metrics["avg_detection_time_minutes"] == pytest.approx(1.5, rel=0.1)
    assert metrics["time_improvement"] > 0  # Should be positive improvement


def test_reset_metrics():
    """Test resetting metrics"""
    tracker = BusinessMetricsTracker()
    
    # Add data
    tracker.record_incident("HIGH", True, 100.0, 60.0)
    tracker.record_incident("MEDIUM", False, 50.0, 90.0)
    
    # Verify data exists
    assert tracker.total_incidents == 2
    assert tracker.total_revenue_saved > 0
    
    # Reset
    tracker.reset()
    
    # Verify reset
    assert tracker.total_incidents == 0
    assert tracker.incidents_auto_healed == 0
    assert tracker.total_revenue_saved == 0.0
    assert tracker.total_revenue_at_risk == 0.0
    assert tracker.detection_times == []


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

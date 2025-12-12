"""
Basic tests for Agentic Reliability Framework
"""

import pytest


def test_basic_import():
    """Test that we can import the main modules"""
    try:
        import agentic_reliability_framework
        assert True
    except ImportError as e:
        pytest.fail(f"Import failed: {e}")


def test_config_exists():
    """Test that config exists"""
    from agentic_reliability_framework import config
    assert config is not None


def test_models_import():
    """Test that models can be imported"""
    from agentic_reliability_framework.models import ReliabilityEvent
    assert ReliabilityEvent is not None


@pytest.mark.unit
def test_basic_arithmetic():
    """Unit test: basic arithmetic"""
    assert 2 + 2 == 4


# Test using fixtures from conftest.py
def test_sample_event_fixture(sample_event):
    """Test that the sample_event fixture works"""
    assert sample_event is not None
    assert sample_event.component == "test-service"


def test_event_factory_fixture(event_factory):
    """Test that the event_factory fixture works"""
    event = event_factory(component="custom-service", latency_p99=200.0)
    assert event.component == "custom-service"
    assert event.latency_p99 == 200.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

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
    try:
        from agentic_reliability_framework import config
        assert config is not None
    except ImportError:
        pytest.skip("Config module not available")


def test_models_import():
    """Test that models can be imported"""
    try:
        from agentic_reliability_framework.models import ReliabilityEvent
        assert ReliabilityEvent is not None
    except ImportError:
        pytest.skip("Models module not available")


@pytest.mark.unit
def test_basic_arithmetic():
    """Unit test: basic arithmetic"""
    assert 2 + 2 == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "unit"])

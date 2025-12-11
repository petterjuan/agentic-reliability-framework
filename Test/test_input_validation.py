"""
Unit tests for input validation functions
"""

import pytest
from agentic_reliability_framework.app import validate_inputs, validate_component_id


class TestComponentIDValidation:
    """Test component ID validation"""
    
    def test_valid_component_ids(self):
        """Test that valid component IDs pass validation"""
        valid_ids = [
            "api-service",
            "auth-service",
            "payment-service-v2",
            "db-01",
            "cache",
            "a",  # Single character
            "api-gateway-prod-001",
        ]
        
        for component_id in valid_ids:
            is_valid, msg = validate_component_id(component_id)
            assert is_valid is True, f"'{component_id}' should be valid but got: {msg}"
            assert msg == ""
    
    def test_invalid_uppercase(self):
        """Test that uppercase letters are rejected"""
        invalid_ids = ["API-SERVICE", "Auth-Service", "PaymentService"]
        
        for component_id in invalid_ids:
            is_valid, msg = validate_component_id(component_id)
            assert is_valid is False
            assert "lowercase" in msg.lower()
    
    def test_invalid_underscore(self):
        """Test that underscores are rejected"""
        is_valid, msg = validate_component_id("api_service")
        assert is_valid is False
        assert "lowercase" in msg.lower() or "hyphen" in msg.lower()
    
    def test_invalid_special_characters(self):
        """Test that special characters are rejected"""
        invalid_ids = [
            "api@service",
            "api.service",
            "api service",  # Space
            "api/service",
            "api&service",
        ]
        
        for component_id in invalid_ids:
            is_valid, msg = validate_component_id(component_id)
            assert is_valid is False, f"'{component_id}' should be invalid"
    
    def test_empty_string(self):
        """Test that empty string is rejected"""
        is_valid, msg = validate_component_id("")
        assert is_valid is False
        assert "1-255" in msg or "character" in msg.lower()
    
    def test_too_long(self):
        """Test that component IDs longer than 255 chars are rejected"""
        long_id = "a" * 256
        is_valid, msg = validate_component_id(long_id)
        assert is_valid is False
        assert "255" in msg
    
    def test_non_string_type(self):
        """Test that non-string types are rejected"""
        is_valid, msg = validate_component_id(123)
        assert is_valid is False
        assert "string" in msg.lower()


class TestNumericInputValidation:
    """Test numeric input validation"""
    
    def test_valid_inputs(self):
        """Test that valid inputs pass validation"""
        is_valid, msg = validate_inputs(
            latency=150.0,
            error_rate=0.05,
            throughput=1000.0,
            cpu_util=0.7,
            memory_util=0.6
        )
        
        assert is_valid is True
        assert msg == ""
    
    def test_valid_inputs_with_none_optionals(self):
        """Test that None is valid for optional fields"""
        is_valid, msg = validate_inputs(
            latency=150.0,
            error_rate=0.05,
            throughput=1000.0,
            cpu_util=None,
            memory_util=None
        )
        
        assert is_valid is True
        assert msg == ""


class TestLatencyValidation:
    """Test latency validation"""
    
    def test_valid_latency(self):
        """Test valid latency values"""
        valid_values = [0, 1, 100, 500, 1000, 9999]
        
        for latency in valid_values:
            is_valid, msg = validate_inputs(latency, 0.05, 1000, None, None)
            assert is_valid is True, f"Latency {latency} should be valid"
    
    def test_negative_latency(self):
        """Test that negative latency is rejected"""
        is_valid, msg = validate_inputs(-10, 0.05, 1000, None, None)
        assert is_valid is False
        assert "latency" in msg.lower()
    
    def test_excessive_latency(self):
        """Test that excessive latency is rejected"""
        is_valid, msg = validate_inputs(20000, 0.05, 1000, None, None)
        assert is_valid is False
        assert "latency" in msg.lower()
    
    def test_non_numeric_latency(self):
        """Test that non-numeric latency is rejected"""
        is_valid, msg = validate_inputs("invalid", 0.05, 1000, None, None)
        assert is_valid is False
        assert "latency" in msg.lower()


class TestErrorRateValidation:
    """Test error rate validation"""
    
    def test_valid_error_rates(self):
        """Test valid error rate values"""
        valid_values = [0, 0.01, 0.05, 0.5, 0.99, 1.0]
        
        for error_rate in valid_values:
            is_valid, msg = validate_inputs(100, error_rate, 1000, None, None)
            assert is_valid is True, f"Error rate {error_rate} should be valid"
    
    def test_negative_error_rate(self):
        """Test that negative error rate is rejected"""
        is_valid, msg = validate_inputs(100, -0.1, 1000, None, None)
        assert is_valid is False
        assert "error rate" in msg.lower()
    
    def test_error_rate_exceeds_one(self):
        """Test that error rate > 1 is rejected"""
        is_valid, msg = validate_inputs(100, 1.5, 1000, None, None)
        assert is_valid is False
        assert "error rate" in msg.lower()
    
    def test_non_numeric_error_rate(self):
        """Test that non-numeric error rate is rejected"""
        is_valid, msg = validate_inputs(100, "high", 1000, None, None)
        assert is_valid is False
        assert "error rate" in msg.lower()


class TestThroughputValidation:
    """Test throughput validation"""
    
    def test_valid_throughput(self):
        """Test valid throughput values"""
        valid_values = [0, 1, 100, 1000, 10000]
        
        for throughput in valid_values:
            is_valid, msg = validate_inputs(100, 0.05, throughput, None, None)
            assert is_valid is True, f"Throughput {throughput} should be valid"
    
    def test_negative_throughput(self):
        """Test that negative throughput is rejected"""
        is_valid, msg = validate_inputs(100, 0.05, -500, None, None)
        assert is_valid is False
        assert "throughput" in msg.lower()
    
    def test_non_numeric_throughput(self):
        """Test that non-numeric throughput is rejected"""
        is_valid, msg = validate_inputs(100, 0.05, "many", None, None)
        assert is_valid is False
        assert "throughput" in msg.lower()


class TestCPUUtilizationValidation:
    """Test CPU utilization validation"""
    
    def test_valid_cpu_util(self):
        """Test valid CPU utilization values"""
        valid_values = [0, 0.1, 0.5, 0.85, 1.0]
        
        for cpu_util in valid_values:
            is_valid, msg = validate_inputs(100, 0.05, 1000, cpu_util, None)
            assert is_valid is True, f"CPU util {cpu_util} should be valid"
    
    def test_negative_cpu_util(self):
        """Test that negative CPU utilization is rejected"""
        is_valid, msg = validate_inputs(100, 0.05, 1000, -0.1, None)
        assert is_valid is False
        assert "cpu" in msg.lower()
    
    def test_cpu_util_exceeds_one(self):
        """Test that CPU utilization > 1 is rejected"""
        is_valid, msg = validate_inputs(100, 0.05, 1000, 1.5, None)
        assert is_valid is False
        assert "cpu" in msg.lower()
    
    def test_non_numeric_cpu_util(self):
        """Test that non-numeric CPU utilization is rejected"""
        is_valid, msg = validate_inputs(100, 0.05, 1000, "high", None)
        assert is_valid is False
        assert "cpu" in msg.lower()


class TestMemoryUtilizationValidation:
    """Test memory utilization validation"""
    
    def test_valid_memory_util(self):
        """Test valid memory utilization values"""
        valid_values = [0, 0.1, 0.5, 0.85, 1.0]
        
        for memory_util in valid_values:
            is_valid, msg = validate_inputs(100, 0.05, 1000, None, memory_util)
            assert is_valid is True, f"Memory util {memory_util} should be valid"
    
    def test_negative_memory_util(self):
        """Test that negative memory utilization is rejected"""
        is_valid, msg = validate_inputs(100, 0.05, 1000, None, -0.1)
        assert is_valid is False
        assert "memory" in msg.lower()
    
    def test_memory_util_exceeds_one(self):
        """Test that memory utilization > 1 is rejected"""
        is_valid, msg = validate_inputs(100, 0.05, 1000, None, 1.5)
        assert is_valid is False
        assert "memory" in msg.lower()
    
    def test_non_numeric_memory_util(self):
        """Test that non-numeric memory utilization is rejected"""
        is_valid, msg = validate_inputs(100, 0.05, 1000, None, "full")
        assert is_valid is False
        assert "memory" in msg.lower()


class TestEdgeCases:
    """Test edge cases and boundary conditions"""
    
    def test_zero_values(self):
        """Test that zero is valid for all metrics"""
        is_valid, msg = validate_inputs(0, 0, 0, 0, 0)
        assert is_valid is True
    
    def test_maximum_values(self):
        """Test maximum boundary values"""
        is_valid, msg = validate_inputs(10000, 1.0, 999999, 1.0, 1.0)
        assert is_valid is True
    
    def test_float_precision(self):
        """Test that high-precision floats are handled"""
        is_valid, msg = validate_inputs(
            latency=123.456789,
            error_rate=0.123456,
            throughput=1234.56,
            cpu_util=0.87654321,
            memory_util=0.76543210
        )
        assert is_valid is True
    
    def test_integer_inputs(self):
        """Test that integer inputs are accepted"""
        is_valid, msg = validate_inputs(100, 0, 1000, 1, 1)
        assert is_valid is True
    
    def test_string_numbers(self):
        """Test that string numbers are converted"""
        is_valid, msg = validate_inputs("100", "0.05", "1000", "0.7", "0.6")
        assert is_valid is True


class TestErrorMessages:
    """Test that error messages are helpful"""
    
    def test_error_message_contains_field_name(self):
        """Test that error messages identify the problematic field"""
        # Latency error
        is_valid, msg = validate_inputs(-10, 0.05, 1000, None, None)
        assert "latency" in msg.lower()
        
        # Error rate error
        is_valid, msg = validate_inputs(100, 2.0, 1000, None, None)
        assert "error rate" in msg.lower()
        
        # Throughput error
        is_valid, msg = validate_inputs(100, 0.05, -100, None, None)
        assert "throughput" in msg.lower()
    
    def test_error_message_has_emoji(self):
        """Test that error messages include emoji for visibility"""
        is_valid, msg = validate_inputs(-10, 0.05, 1000, None, None)
        assert "❌" in msg
    
    def test_error_message_provides_guidance(self):
        """Test that error messages provide guidance"""
        is_valid, msg = validate_inputs(-10, 0.05, 1000, None, None)
        assert "between" in msg.lower() or "range" in msg.lower() or "0-10000" in msg


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
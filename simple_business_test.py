#!/usr/bin/env python3
"""
Simple test of business metrics
"""
from agentic_reliability_framework.app import get_business_metrics

print("Testing get_business_metrics...")
try:
    metrics = get_business_metrics()
    print(f"Got metrics: {metrics}")
    print(f"Type: {type(metrics)}")
    
    # Check if it has record_incident method
    if hasattr(metrics, 'record_incident'):
        print("✅ Has record_incident method")
    else:
        print("❌ No record_incident method")
        print(f"Methods: {[m for m in dir(metrics) if not m.startswith('_')]}")
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

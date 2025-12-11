import sys
print("Testing BusinessMetricsTracker...")

# First check if we can import it
try:
    from agentic_reliability_framework import get_business_metrics
    print("✅ get_business_metrics imported")
    
    # Try to get the instance
    metrics = get_business_metrics()
    print(f"✅ BusinessMetricsTracker instance: {metrics}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

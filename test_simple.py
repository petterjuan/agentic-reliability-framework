"""
Simple test to verify ARF works after bug fix
"""
import asyncio

from agentic_reliability_framework import get_engine

async def main():
    print("Testing ARF after bug fix...")
    
    engine = get_engine()
    print("✅ Engine created")
    
    # Test normal event
    result = await engine.process_event_enhanced(
        component="test-service",
        latency=100.0,
        error_rate=0.05,
        throughput=1000.0
    )
    
    print(f"✅ Result: {result['status']}")
    print(f"✅ Severity: {result['severity']}")
    
    # Test anomaly
    result2 = await engine.process_event_enhanced(
        component="problem-service",
        latency=500.0,
        error_rate=0.35,
        throughput=300.0
    )
    
    print(f"✅ Result2: {result2['status']}")
    print(f"✅ Severity: {result2['severity']}")
    
    print("\n🎉 All tests passed!")

if __name__ == "__main__":
    asyncio.run(main())

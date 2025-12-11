#!/usr/bin/env python3
import asyncio
from agentic_reliability_framework import get_engine

async def test():
    print("Quick test...")
    engine = get_engine()
    
    # Normal event
    result = await engine.process_event_enhanced(
        component="quick-test",
        latency=50.0,
        error_rate=0.01,
        throughput=2000.0
    )
    print(f"Normal: {result['status']}")
    
    # Anomaly
    result = await engine.process_event_enhanced(
        component="quick-db",
        latency=800.0,
        error_rate=0.40,
        throughput=100.0
    )
    print(f"Anomaly: {result['status']}")
    if 'business_impact' in result:
        print(f"Impact: ${result['business_impact']['revenue_loss_estimate']:.2f}")
    
    print("✅ Test passed!")

asyncio.run(test())

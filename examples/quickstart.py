"""
ARF QuickStart Example
Zero to incident detection in 2 minutes
"""

import asyncio
from arf import EnhancedReliabilityEngine

async def main():
    print("=" * 60)
    print("🚀 ARF QuickStart: Detecting Production Incident")
    print("=" * 60)
    
    # Initialize engine
    engine = EnhancedReliabilityEngine()
    
    # Simulate critical database incident
    print("\n📊 Simulating: Database Latency Spike + High Error Rate")
    
    result = await engine.process_event_enhanced(
        component="database-primary",
        latency=850,
        error_rate=0.35,
        throughput=450,
        cpu_util=0.78,
        memory_util=0.98
    )
    
    print(f"\n🔍 Analysis Result:")
    print(f"  Status: {result['status']}")
    print(f"  Severity: {result['severity']}")
    
    print("\n" + "=" * 60)
    print("✅ QuickStart Complete!")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())

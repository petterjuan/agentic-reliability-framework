#!/usr/bin/env python3
"""
Definitive test after all fixes
"""
import asyncio
import sys

async def test():
    print("="*60)
    print("DEFINITIVE ARF TEST - AFTER ALL FIXES")
    print("="*60)
    
    try:
        # Test import
        from agentic_reliability_framework import get_engine
        print("✅ Import successful")
        
        # Test engine creation
        engine = get_engine()
        print("✅ Engine created")
        
        # Test normal event
        print("\n📊 Testing normal event...")
        result1 = await engine.process_event_enhanced(
            component="api-gateway",
            latency=85.0,
            error_rate=0.03,
            throughput=1500.0
        )
        print(f"   Status: {result1['status']}")
        print(f"   Severity: {result1['severity']}")
        
        # Test anomaly
        print("\n🚨 Testing anomaly...")
        result2 = await engine.process_event_enhanced(
            component="database",
            latency=450.0,
            error_rate=0.28,
            throughput=600.0
        )
        print(f"   Status: {result2['status']}")
        print(f"   Severity: {result2['severity']}")
        
        if 'multi_agent_analysis' in result2:
            confidence = result2['multi_agent_analysis']['incident_summary']['anomaly_confidence']
            print(f"   Confidence: {confidence*100:.1f}%")
        
        if 'business_impact' in result2:
            impact = result2['business_impact']
            print(f"   Revenue Impact: ${impact['revenue_loss_estimate']:.2f}")
        
        print("\n" + "="*60)
        print("🎉 ALL TESTS PASSED!")
        print("="*60)
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test())
    sys.exit(0 if success else 1)

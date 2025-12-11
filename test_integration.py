"""
Integration test for ARF package
"""
import asyncio
import sys
from agentic_reliability_framework import get_engine

async def test_basic_functionality():
    print("🧪 Testing ARF basic functionality...")
    
    try:
        # Test 1: Engine creation
        engine = get_engine()
        print("✅ Engine created successfully")
        
        # Test 2: Process event
        result = await engine.process_event_enhanced(
            component="test-component",
            latency=150.0,
            error_rate=0.05,
            throughput=1000.0,
            cpu_util=0.45,
            memory_util=0.35
        )
        
        print(f"✅ Event processed: {result['status']}")
        
        # Test 3: Check structure
        required_keys = ['status', 'severity', 'timestamp']
        for key in required_keys:
            if key in result:
                print(f"✅ Result contains {key}")
            else:
                print(f"❌ Result missing {key}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_multiple_events():
    print("\n🧪 Testing multiple events...")
    
    try:
        engine = get_engine()
        
        events = [
            {"latency": 50.0, "error_rate": 0.01, "throughput": 1500.0},
            {"latency": 250.0, "error_rate": 0.15, "throughput": 800.0},
            {"latency": 500.0, "error_rate": 0.35, "throughput": 300.0},
        ]
        
        for i, event in enumerate(events, 1):
            result = await engine.process_event_enhanced(
                component=f"test-service-{i}",
                **event
            )
            print(f"✅ Event {i}: {result['status']} ({result['severity']})")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    print("="*60)
    print("ARF Integration Test")
    print("="*60)
    
    tests = [
        ("Basic Functionality", test_basic_functionality()),
        ("Multiple Events", test_multiple_events()),
    ]
    
    results = []
    for test_name, test_coro in tests:
        print(f"\n📋 {test_name}:")
        try:
            success = await test_coro
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    all_passed = True
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} - {test_name}")
        if not success:
            all_passed = False
    
    print("\n" + "="*60)
    if all_passed:
        print("🎉 ALL INTEGRATION TESTS PASSED!")
    else:
        print("⚠️  SOME TESTS FAILED")
    
    return all_passed

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)

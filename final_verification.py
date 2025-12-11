#!/usr/bin/env python3
"""
Final verification before commit
"""
import asyncio
import subprocess
import sys

def test_cli():
    print("1. Testing CLI...")
    tests = [
        ("arf --version", "2.0.0"),
        ("arf doctor", "All dependencies OK"),
    ]
    
    for cmd, expected in tests:
        try:
            result = subprocess.run(cmd.split(), capture_output=True, text=True)
            if expected in result.stdout or expected in result.stderr:
                print(f"   ✅ {cmd}")
            else:
                print(f"   ❌ {cmd}")
                return False
        except:
            print(f"   ❌ {cmd}")
            return False
    return True

async def test_core():
    print("\n2. Testing core engine...")
    try:
        from agentic_reliability_framework import get_engine
        
        engine = get_engine()
        
        # Normal
        result = await engine.process_event_enhanced(
            component="final-test",
            latency=80.0,
            error_rate=0.03,
            throughput=1600.0
        )
        if result['status'] == 'NORMAL':
            print("   ✅ Normal event processing")
        else:
            print(f"   ❌ Expected NORMAL, got {result['status']}")
            return False
        
        # Anomaly
        result = await engine.process_event_enhanced(
            component="final-db",
            latency=750.0,
            error_rate=0.38,
            throughput=280.0
        )
        if result['status'] == 'ANOMALY':
            print("   ✅ Anomaly detection")
        else:
            print(f"   ❌ Expected ANOMALY, got {result['status']}")
            return False
        
        return True
        
    except Exception as e:
        print(f"   ❌ Core test failed: {e}")
        return False

async def main():
    print("="*60)
    print("FINAL VERIFICATION - ARF v2.0.0")
    print("="*60)
    
    cli_ok = test_cli()
    core_ok = await test_core()
    
    print("\n" + "="*60)
    if cli_ok and core_ok:
        print("✅ ALL VERIFICATIONS PASSED")
        print("ARF v2.0.0 is ready for production!")
        return True
    else:
        print("❌ VERIFICATION FAILED")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)

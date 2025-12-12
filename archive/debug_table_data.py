#!/usr/bin/env python3
import sys
sys.path.insert(0, '.')
import asyncio

async def test():
    from app import enhanced_engine  # Removed: business_metrics
    
    print("=== Testing Table Data Issue ===")
    
    print(f"1. Current event store count: {enhanced_engine.event_store.count()}")
    
    # Simulate what the submit function does
    print("\n2. Simulating table data building...")
    
    # First, add some events manually
    from models import ReliabilityEvent, EventSeverity
    import datetime
    
    for i in range(3):
        event = ReliabilityEvent(
            component=f'test-service-{i}',
            latency_p99=100 + i*100,
            error_rate=0.05 + i*0.05,
            throughput=1000 + i*200,
            severity=EventSeverity.HIGH if i > 1 else EventSeverity.MEDIUM,
            timestamp=datetime.datetime.now(datetime.timezone.utc)
        )
        enhanced_engine.event_store.add(event)
    
    print("   Added 3 test events")  # Removed: f prefix
    print(f"   Total events now: {enhanced_engine.event_store.count()}")
    
    # Now simulate the table building
    table_data = []
    events = enhanced_engine.event_store.get_recent(15)
    print(f"\n3. Retrieved {len(events)} events with get_recent(15)")
    
    for event in events:
        print(f"   Processing: {event.component} at {event.timestamp}")
        table_data.append([
            event.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            event.component,
            f"{event.latency_p99:.0f}ms",
            f"{event.error_rate:.3f}",
            f"{event.throughput:.0f}",
            event.severity.value.upper(),
            "Test analysis"
        ])
    
    print(f"\n4. Table data built: {len(table_data)} rows")
    if table_data:
        print("   First row:")
        for i, cell in enumerate(table_data[0]):
            print(f"     [{i}] {cell}")
    else:
        print("   ERROR: table_data is empty!")
        
    # Check if there's an issue with timestamp formatting
    print("\n5. Checking event attributes...")
    if events:
        event = events[0]
        print(f"   Event timestamp type: {type(event.timestamp)}")
        print(f"   Event timestamp: {event.timestamp}")
        print(f"   Has strftime? {hasattr(event.timestamp, 'strftime')}")
        try:
            formatted = event.timestamp.strftime("%Y-%m-%d %H:%M:%S")
            print(f"   Formatted: {formatted}")
        except Exception as e:
            print(f"   Format error: {e}")

if __name__ == "__main__":
    asyncio.run(test())

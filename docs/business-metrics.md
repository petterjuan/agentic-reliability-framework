# Business Impact Calculation

**Translating Technical Metrics to Revenue Impact**

---

## Table of Contents

- [Why Business Metrics Matter](#why-business-metrics-matter)
- [The Revenue Impact Formula](#the-revenue-impact-formula)
- [ARF's Business Metrics System](#arfs-business-metrics-system)
- [Calculating Impact](#calculating-impact)
- [Real-World Examples](#real-world-examples)
- [ROI Calculation](#roi-calculation)
- [Reporting & Dashboards](#reporting--dashboards)
- [Best Practices](#best-practices)

---

## Why Business Metrics Matter

### The Problem: Technical vs Business Language

**Technical team says:**
> "Latency increased to 450ms, error rate is 15%"

**Business team hears:**
> "Some numbers went up. Is this bad?"

**Gap:** No shared understanding of impact.

---

### The Solution: Speak Money

**ARF says:**
> "Latency spike is costing $2,500/minute in lost revenue  
> Estimated total impact: $75,000 over 30 minutes  
> 1,250 users affected  
> 187 failed transactions"

**Business team:** "Fix it NOW!" ✅

---

### Why This Matters

**For Engineering:**
- ✅ Justifies infrastructure investment
- ✅ Prioritizes incident response
- ✅ Proves ROI of reliability work

**For Business:**
- ✅ Understands technical issues in business terms
- ✅ Makes informed trade-off decisions
- ✅ Allocates budget appropriately

**For Everyone:**
- ✅ Shared language
- ✅ Aligned incentives
- ✅ Better outcomes

---

## The Revenue Impact Formula

### Basic Formula

```python
revenue_loss = downtime_minutes × revenue_per_minute × impact_percentage
```

**Components:**

1. **downtime_minutes:** Duration of incident
2. **revenue_per_minute:** Baseline revenue rate
3. **impact_percentage:** % of revenue stream affected

---

### Detailed Formula

ARF uses a more sophisticated model:

```python
def calculate_revenue_loss(event: ReliabilityEvent) -> float:
    # Base revenue rate
    base_revenue = config.BASE_REVENUE_PER_MINUTE
    
    # Severity multiplier
    severity_multiplier = {
        "CRITICAL": 1.0,  # 100% impact
        "HIGH": 0.5,      # 50% impact
        "MEDIUM": 0.2,    # 20% impact
        "LOW": 0.05       # 5% impact
    }[event.severity]
    
    # Throughput impact
    throughput_factor = calculate_throughput_impact(event)
    
    # Error rate impact
    error_factor = min(event.error_rate * 2, 1.0)
    
    # Time duration
    duration_minutes = (datetime.utcnow() - event.timestamp).seconds / 60
    
    # Total loss
    revenue_loss = (
        base_revenue * 
        severity_multiplier * 
        throughput_factor * 
        error_factor *
        duration_minutes
    )
    
    return revenue_loss
```

---

## ARF's Business Metrics System

### Architecture

```
Reliability Event
     ↓
Business Impact Calculator
     ↓
- Revenue Loss
- Users Affected
- Transactions Failed
- Recovery Time Estimate
     ↓
Business Metrics Tracker
     ↓
- Cumulative Totals
- MTTR
- Availability %
- Revenue Saved
```

---

### BusinessImpactCalculator

```python
class BusinessImpactCalculator:
    """Calculates financial impact of incidents"""
    
    def __init__(self, config: AppConfig):
        self.base_revenue = config.base_revenue_per_minute
        self.base_users = config.base_users
    
    def calculate_impact(self, event: ReliabilityEvent) -> Dict[str, Any]:
        """
        Calculate complete business impact
        
        Returns:
            {
                "revenue_loss_estimate": float,      # $ lost
                "users_affected": int,               # User count
                "transactions_failed": int,          # Transaction count
                "recovery_time_minutes": float,      # Estimated MTTR
                "total_cost": float                  # Total $ impact
            }
        """
        return {
            "revenue_loss_estimate": self._calculate_revenue_loss(event),
            "users_affected": self._estimate_users_affected(event),
            "transactions_failed": self._estimate_failed_transactions(event),
            "recovery_time_minutes": self._estimate_recovery_time(event),
            "total_cost": self._calculate_total_cost(event)
        }
```

---

### BusinessMetricsTracker

```python
class BusinessMetricsTracker:
    """Tracks cumulative business metrics"""
    
    def __init__(self):
        self.total_incidents = 0
        self.total_downtime_minutes = 0.0
        self.total_revenue_saved = 0.0
        self.incident_timestamps = []
    
    def record_incident(
        self,
        downtime_minutes: float,
        revenue_lost: float,
        severity: Severity
    ) -> None:
        """Record incident metrics"""
        self.total_incidents += 1
        self.total_downtime_minutes += downtime_minutes
        
        # Revenue saved = What would have been lost without ARF
        traditional_mttr = 30  # minutes (industry average)
        arf_mttr = downtime_minutes
        
        saved = revenue_lost * (traditional_mttr / arf_mttr - 1)
        self.total_revenue_saved += saved
    
    def get_totals(self) -> Dict[str, Any]:
        """Get cumulative metrics"""
        return {
            "total_incidents": self.total_incidents,
            "total_downtime_minutes": self.total_downtime_minutes,
            "total_revenue_saved": self.total_revenue_saved,
            "mttr_minutes": self._calculate_mttr(),
            "availability_percentage": self._calculate_availability()
        }
```

---

## Calculating Impact

### 1. Revenue Loss

#### **Method 1: Direct Calculation**

```python
def calculate_revenue_loss(event: ReliabilityEvent) -> float:
    """
    Calculate revenue loss from event
    
    Formula:
        Loss = Revenue/min × Impact% × Duration
    """
    base_revenue = 100.0  # $/min
    
    # Impact based on severity
    if event.severity == "CRITICAL":
        impact_pct = 1.0  # 100% of traffic affected
    elif event.severity == "HIGH":
        impact_pct = 0.5  # 50% affected
    elif event.severity == "MEDIUM":
        impact_pct = 0.2  # 20% affected
    else:
        impact_pct = 0.05  # 5% affected
    
    # Duration
    duration = (datetime.utcnow() - event.timestamp).seconds / 60
    
    # Calculate loss
    loss = base_revenue * impact_pct * duration
    
    return loss
```

---

#### **Method 2: Throughput-Based**

```python
def calculate_throughput_loss(event: ReliabilityEvent) -> float:
    """
    Calculate loss based on throughput degradation
    
    Formula:
        Loss = (Normal_RPS - Current_RPS) × Revenue_Per_Request × Duration
    """
    normal_throughput = 2000  # req/sec (baseline)
    current_throughput = event.throughput
    
    revenue_per_request = 0.05  # $0.05 per request
    
    # Lost requests per second
    lost_rps = max(0, normal_throughput - current_throughput)
    
    # Duration in seconds
    duration_sec = (datetime.utcnow() - event.timestamp).seconds
    
    # Total loss
    loss = lost_rps * revenue_per_request * duration_sec
    
    return loss
```

---

#### **Method 3: Error-Based**

```python
def calculate_error_loss(event: ReliabilityEvent) -> float:
    """
    Calculate loss from failed transactions
    
    Formula:
        Loss = Failed_Transactions × Average_Transaction_Value
    """
    total_requests = event.throughput * 60  # per minute
    failed_requests = total_requests * event.error_rate
    
    avg_transaction_value = 25.0  # $25 per transaction
    
    # Loss from failed transactions
    loss = failed_requests * avg_transaction_value
    
    return loss
```

---

### 2. Users Affected

```python
def estimate_users_affected(event: ReliabilityEvent) -> int:
    """
    Estimate number of users impacted
    
    Assumptions:
    - Base users × severity factor
    - Error rate influences count
    """
    base_users = 10000  # Active users
    
    # Severity factor
    severity_factors = {
        "CRITICAL": 1.0,   # All users
        "HIGH": 0.5,       # Half of users
        "MEDIUM": 0.2,     # 20% of users
        "LOW": 0.05        # 5% of users
    }
    
    severity_factor = severity_factors.get(event.severity, 0.1)
    
    # Error rate influence
    error_factor = min(event.error_rate * 2, 1.0)
    
    # Estimate
    affected = int(base_users * severity_factor * error_factor)
    
    return affected
```

---

### 3. Transactions Failed

```python
def estimate_failed_transactions(event: ReliabilityEvent) -> int:
    """
    Estimate failed transaction count
    
    Formula:
        Failed = Throughput × Error_Rate × Duration
    """
    duration_minutes = (datetime.utcnow() - event.timestamp).seconds / 60
    
    # Requests per minute
    requests_per_min = event.throughput * 60
    
    # Failed transactions
    failed = int(requests_per_min * event.error_rate * duration_minutes)
    
    return failed
```

---

### 4. Recovery Time Estimate

```python
def estimate_recovery_time(event: ReliabilityEvent) -> float:
    """
    Estimate time to recovery (MTTR)
    
    Based on:
    - Severity
    - Historical data
    - Complexity
    """
    # Base recovery times (minutes)
    base_times = {
        "CRITICAL": 15,
        "HIGH": 10,
        "MEDIUM": 5,
        "LOW": 2
    }
    
    base_time = base_times.get(event.severity, 10)
    
    # Adjust based on complexity
    if event.cpu_util > 0.9 and event.memory_util > 0.85:
        # Complex: resource exhaustion
        base_time *= 1.5
    
    if event.error_rate > 0.5:
        # Complex: widespread errors
        base_time *= 1.3
    
    return base_time
```

---

## Real-World Examples

### Example 1: E-commerce Checkout Failure

**Scenario:**
- Component: `payment-service`
- Latency: 850ms (normally 100ms)
- Error rate: 25%
- Duration: 45 minutes
- Base revenue: $5,000/min

**Calculation:**

```python
# Revenue loss
impact_pct = 0.75  # 75% of traffic affected
duration = 45  # minutes
loss = 5000 * 0.75 * 45
     = $168,750

# Users affected
base_users = 50000
affected = 50000 * 0.75
         = 37,500 users

# Transactions failed
throughput = 2000  # req/sec
failed = 2000 * 60 * 0.25 * 45
       = 1,350,000 failed requests
```

**Business Impact:**
```
💰 Revenue Lost: $168,750
👥 Users Affected: 37,500
❌ Failed Transactions: 1.35M
⏱️ Downtime: 45 minutes
```

---

### Example 2: API Rate Limit Exceeded

**Scenario:**
- Component: `api-gateway`
- Latency: Normal
- Error rate: 100% (for some requests)
- Throughput: 60% of normal
- Duration: 15 minutes

**Calculation:**

```python
# Throughput loss
normal_throughput = 5000  # req/sec
current_throughput = 3000  # 60% of normal
lost_rps = 5000 - 3000 = 2000

revenue_per_request = 0.10  # $0.10
duration_sec = 15 * 60 = 900

loss = 2000 * 0.10 * 900
     = $180,000
```

**Business Impact:**
```
💰 Revenue Lost: $180,000
👥 Users Affected: 25,000
❌ Failed Requests: 1.8M
⏱️ Downtime: 15 minutes
```

---

### Example 3: Database Slowdown

**Scenario:**
- Component: `database`
- Latency: 3x normal
- Error rate: 5%
- All services affected
- Duration: 2 hours

**Calculation:**

```python
# Multi-service impact
services = ["api", "web", "mobile"]
base_revenue = 2000  # $/min per service

# Severity: HIGH (not critical, but widespread)
impact_pct = 0.5

total_loss = 0
for service in services:
    loss = 2000 * 0.5 * 120  # 2 hours = 120 min
    total_loss += loss

total_loss = $360,000
```

**Business Impact:**
```
💰 Revenue Lost: $360,000
👥 Users Affected: 100,000+
⏱️ Downtime: 2 hours
📉 3 services impacted
```

---

## ROI Calculation

### Traditional vs ARF

**Without ARF (Manual Response):**

```python
# Typical incident
mttr_manual = 45  # minutes
incidents_per_month = 20

# Revenue loss per incident
avg_loss_per_incident = 100 * 45  # $100/min × 45 min
                      = $4,500

# Monthly cost
monthly_cost = 20 * 4500
             = $90,000

# Annual cost
annual_cost = 90000 * 12
            = $1,080,000
```

---

**With ARF (Automated Response):**

```python
# ARF recovery
mttr_arf = 2  # minutes (95% faster)
incidents_per_month = 20

# Revenue loss per incident
avg_loss_per_incident = 100 * 2  # $100/min × 2 min
                      = $200

# Monthly cost
monthly_cost = 20 * 200
             = $4,000

# Annual cost
annual_cost = 4000 * 12
            = $48,000
```

---

**ROI Calculation:**

```python
# Savings
annual_savings = 1,080,000 - 48,000
               = $1,032,000

# ARF cost
arf_implementation = 47,500  # One-time
arf_maintenance = 12,500 * 12  # Monthly × 12
                = 150,000  # Annual

# Net benefit
net_benefit = 1,032,000 - 47,500 - 150,000
            = $834,500

# ROI
roi = (834,500 / (47,500 + 150,000)) * 100
    = 422% ROI

# Payback period
payback = (47,500 + 150,000) / (1,032,000 / 12)
        = 2.3 months
```

**Summary:**
```
💰 Annual Savings: $1,032,000
💵 Investment: $197,500
📈 ROI: 422%
⏱️ Payback: 2.3 months
```

---

## Reporting & Dashboards

### Daily Incident Report

```
═══════════════════════════════════════
DAILY RELIABILITY REPORT
Date: December 9, 2025
═══════════════════════════════════════

INCIDENTS TODAY: 3

Incident #1: Critical Latency Spike
  Time: 02:47 AM
  Component: payment-service
  Duration: 2 minutes (ARF auto-healed)
  Impact: $200 lost
  Users Affected: 125

Incident #2: Database Connection Pool
  Time: 10:15 AM
  Component: database
  Duration: 30 seconds (ARF auto-healed)
  Impact: $50 lost
  Users Affected: 42

Incident #3: API Rate Limit
  Time: 3:22 PM
  Component: api-gateway
  Duration: 5 minutes (ARF auto-healed)
  Impact: $500 lost
  Users Affected: 312

───────────────────────────────────────
TOTALS:
  Revenue Lost: $750
  Revenue Saved: $44,250 (vs manual)
  Users Protected: 479
  Avg MTTR: 2.5 minutes
  Availability: 99.98%
═══════════════════════════════════════
```

---

### Monthly Executive Summary

```
═══════════════════════════════════════
MONTHLY EXECUTIVE SUMMARY
December 2025
═══════════════════════════════════════

RELIABILITY METRICS:
  Incidents: 62
  Avg MTTR: 2.3 minutes (95% improvement)
  Availability: 99.96%
  Uptime: 99.99%

BUSINESS IMPACT:
  💰 Revenue Protected: $1,240,000
  💵 Revenue Lost: $32,000
  👥 Users Affected: 15,200
  🎯 Recovery Success Rate: 98.4%

COST SAVINGS:
  Traditional Cost: $1,272,000
  Actual Cost: $32,000
  Net Savings: $1,240,000

ROI THIS MONTH:
  Investment: $12,500 (monthly fee)
  Return: $1,240,000
  ROI: 9,820%

TOP INCIDENTS:
  1. Database outage (Dec 5): $12,000
  2. Payment gateway (Dec 12): $8,500
  3. API overload (Dec 18): $6,200

ACTIONS TAKEN:
  Restarts: 28
  Scale-ups: 18
  Circuit breakers: 12
  Alerts: 4
═══════════════════════════════════════
```

---

## Best Practices

### 1. Configure Realistic Base Revenue

```python
# Calculate your actual revenue/minute
annual_revenue = 50_000_000  # $50M
minutes_per_year = 365 * 24 * 60

revenue_per_minute = annual_revenue / minutes_per_year
                   = 50_000_000 / 525_600
                   = $95.13 per minute

# Set in config
BASE_REVENUE_PER_MINUTE = 95.13
```

---

### 2. Adjust for Business Hours

```python
# Not all hours are equal
def get_revenue_multiplier():
    hour = datetime.now().hour
    
    if 9 <= hour <= 17:
        return 2.0  # Peak business hours
    elif 17 <= hour <= 22:
        return 1.5  # Evening
    else:
        return 0.5  # Night (lower traffic)
```

---

### 3. Track by Service

```python
# Different services = different revenue
service_revenue = {
    "payment-service": 1000,  # $/min
    "search-service": 50,     # $/min
    "recommendation": 200     # $/min
}

def get_service_revenue(component: str) -> float:
    return service_revenue.get(component, 100)  # default
```

---

### 4. Include Indirect Costs

```python
# Not just lost revenue
def calculate_total_cost(revenue_loss: float) -> float:
    # Direct revenue loss
    total = revenue_loss
    
    # Customer churn (future revenue)
    churn_cost = revenue_loss * 0.2
    total += churn_cost
    
    # Engineering time
    eng_time_cost = 5 * 150  # 5 people × $150/hour
    total += eng_time_cost
    
    # Reputation damage
    reputation_cost = revenue_loss * 0.1
    total += reputation_cost
    
    return total
```

---

### 5. Report to Stakeholders

```python
# Monthly email to executives
def send_monthly_report():
    metrics = tracker.get_totals()
    
    email_body = f"""
    Reliability Performance - {current_month}
    
    Revenue Protected: ${metrics['revenue_saved']:,.2f}
    Incidents Resolved: {metrics['total_incidents']}
    Avg Recovery Time: {metrics['mttr_minutes']:.1f} min
    System Availability: {metrics['availability_percentage']:.2f}%
    
    ARF prevented {metrics['total_incidents']} outages,
    saving ${metrics['revenue_saved']:,.2f} in revenue.
    
    ROI this month: {calculate_roi():,.0f}%
    """
    
    send_email(to=executives, subject="Monthly Reliability Report", body=email_body)
```

---

## Further Reading

- [Architecture Overview](./architecture.md) - How metrics fit in
- [Self-Healing Patterns](./self-healing.md) - Automated recovery
- [API Reference](./api.md) - Business metrics APIs

---

**Built by [LGCY Labs](https://lgcylabs.vercel.app/) | [Back to README](../README.md)**

#!/usr/bin/env python3
"""
Rebuild the BusinessMetricsTracker section completely
"""
import re

with open('agentic_reliability_framework/app.py', 'r') as f:
    content = f.read()

# Find the section from "class BusinessMetricsTracker:" to "class RateLimiter:"
pattern = r'(class BusinessMetricsTracker:.*?)(?=^class RateLimiter:)'

match = re.search(pattern, content, re.DOTALL | re.MULTILINE)
if not match:
    print("❌ Could not find BusinessMetricsTracker section")
    exit(1)

old_section = match.group(1)
print(f"Old section length: {len(old_section)} chars")

# Create the correct section
correct_section = '''class BusinessMetricsTracker:
    """Track cumulative business metrics for ROI dashboard"""
    
    def __init__(self):
        self.total_incidents = 0
        self.incidents_auto_healed = 0
        self.total_revenue_saved = 0.0
        self.total_revenue_at_risk = 0.0
        self.detection_times = []
        self._lock = threading.RLock()
        logger.info("Initialized BusinessMetricsTracker")
    
    def record_incident(
        self,
        severity: str,
        auto_healed: bool,
        revenue_loss: float,
        detection_time_seconds: float = 120.0  # 2 minutes default
    ):
        """Record an incident and update metrics"""
        with self._lock:
            self.total_incidents += 1
            
            if auto_healed:
                self.incidents_auto_healed += 1
            
            # Calculate what revenue would have been lost (industry average: 14 min response)
            # vs what we actually lost (ARF average: 2 min response)
            industry_avg_response_minutes = 14
            arf_response_minutes = detection_time_seconds / 60
            
            # Revenue at risk if using traditional monitoring
            revenue_per_minute = revenue_loss / max(1, arf_response_minutes)
            traditional_loss = revenue_per_minute * industry_avg_response_minutes
            
            self.total_revenue_at_risk += traditional_loss
            self.total_revenue_saved += (traditional_loss - revenue_loss)
            
            self.detection_times.append(detection_time_seconds)
            
            logger.info(
                f"Recorded incident: auto_healed={auto_healed}, "
                f"saved=${traditional_loss - revenue_loss:.2f}"
            )
    
    def get_metrics(self) -> dict:
        """Get current cumulative metrics"""
        with self._lock:
            auto_heal_rate = (
                (self.incidents_auto_healed / self.total_incidents * 100)
                if self.total_incidents > 0 else 0
            )
            
            avg_detection_time = (
                sum(self.detection_times) / len(self.detection_times)
                if self.detection_times else 120.0
            )
            
            return {
                "total_incidents": self.total_incidents,
                "incidents_auto_healed": self.incidents_auto_healed,
                "auto_heal_rate": auto_heal_rate,
                "total_revenue_saved": self.total_revenue_saved,
                "total_revenue_at_risk": self.total_revenue_at_risk,
                "avg_detection_time_seconds": avg_detection_time,
                "avg_detection_time_minutes": avg_detection_time / 60,
                "time_improvement": (
                    (14 - (avg_detection_time / 60)) / 14 * 100
                )  # vs industry 14 min
            }
    
    def reset(self):
        """Reset all metrics (for demo purposes)"""
        with self._lock:
            self.total_incidents = 0
            self.incidents_auto_healed = 0
            self.total_revenue_saved = 0.0
            self.total_revenue_at_risk = 0.0
            self.detection_times = []
            logger.info("Reset BusinessMetricsTracker")


# Global business metrics tracker
business_metrics = BusinessMetricsTracker()

def get_business_metrics():
    """Get the global BusinessMetricsTracker instance"""
    return business_metrics

'''

# Replace the old section with the correct one
new_content = content.replace(old_section, correct_section)

# Write back
with open('agentic_reliability_framework/app.py', 'w') as f:
    f.write(new_content)

print("✅ Rebuilt BusinessMetricsTracker section")

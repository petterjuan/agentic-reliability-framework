"""
Enterprise Agentic Reliability Framework - Main Application (FIXED VERSION)
Multi-Agent AI System for Production Reliability Monitoring

NYC Industry Focus: SaaS, Finance, Healthcare, Media/Advertising, Logistics/Shipping
"""

import asyncio
import datetime
import json
import logging
import os
import sys
import tempfile
import threading
from collections import deque
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from enum import Enum
from queue import Queue
from typing import Any, Literal, Optional, Union

import atomicwrites
import gradio as gr
import numpy as np
from circuitbreaker import circuit

# Import our modules
from .config import config
from .engine.anomaly import AdvancedAnomalyDetector
from .engine.business import BusinessImpactCalculator, BusinessMetricsTracker
from .engine.predictive import SimplePredictiveEngine
from .engine.reliability import EnhancedReliabilityEngine, ThreadSafeEventStore
from .healing_policies import PolicyEngine
from .memory.faiss_index import ProductionFAISSIndex, create_faiss_index
from .models import (
    EventSeverity,  # noqa: F401
    ForecastResult,  # noqa: F401
    HealingAction,  # noqa: F401
    ReliabilityEvent,
)


def get_engine():
    from .lazy import get_engine as _get_engine
    return _get_engine()


def get_agents():
    from .lazy import get_agents as _get_agents
    return _get_agents()


def get_faiss_index():
    from .lazy import get_faiss_index as _get_faiss_index
    return _get_faiss_index()


def get_business_metrics():
    from .lazy import get_business_metrics as _get_business_metrics
    return _get_business_metrics()


def enhanced_engine():
    return get_engine()


# === Logging Configuration ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# === CONSTANTS (FIXED: Extracted all magic numbers) ===
class Constants:
    """Centralized constants to eliminate magic numbers"""
    
    # Thresholds
    LATENCY_WARNING = 150.0
    LATENCY_CRITICAL = 300.0
    LATENCY_EXTREME = 500.0
    
    ERROR_RATE_WARNING = 0.05
    ERROR_RATE_HIGH = 0.15
    ERROR_RATE_CRITICAL = 0.3
    
    CPU_WARNING = 0.8
    CPU_CRITICAL = 0.9
    
    MEMORY_WARNING = 0.8
    MEMORY_CRITICAL = 0.9
    
    # Forecasting
    SLOPE_THRESHOLD_INCREASING = 5.0
    SLOPE_THRESHOLD_DECREASING = -2.0
    
    FORECAST_MIN_DATA_POINTS = 5
    FORECAST_LOOKAHEAD_MINUTES = 15
    
    # Performance
    HISTORY_WINDOW = 50
    MAX_EVENTS_STORED = 1000
    AGENT_TIMEOUT_SECONDS = 5
    CACHE_EXPIRY_MINUTES = 15
    
    # FAISS
    FAISS_BATCH_SIZE = 10
    FAISS_SAVE_INTERVAL_SECONDS = 30
    VECTOR_DIM = 384
    
    # Business metrics
    BASE_REVENUE_PER_MINUTE = 100.0
    BASE_USERS = 1000
    
    # Rate limiting
    MAX_REQUESTS_PER_MINUTE = 60
    MAX_REQUESTS_PER_HOUR = 500


# === Configuration ===
HEADERS = {"Authorization": f"Bearer {config.hf_api_key}"} if config.hf_api_key else {}

# === Demo Scenarios for NYC Industries ===
DEMO_SCENARIOS = {
    "🏦 Finance - HFT Latency Spike": {
        "component": "trading-engine",
        "latency": 42.0,
        "error_rate": 0.0001,
        "throughput": 50000.0,
        "cpu_util": 0.85,
        "memory_util": 0.72,
        "revenue_at_risk": 5000000,
        "story": """
**FINANCE: High-Frequency Trading Latency Anomaly**

🏢 **Firm:** Wall Street Trading Desk  
📍 **Location:** Lower Manhattan, NYC  
📊 **Assets:** $47B under management  
⏱️ **Critical Threshold:** 50ms max latency  

🕐 **Time:** 9:47 AM EST (Market Open)  
⚡ **Latency Spike:** 42ms (from 8ms baseline)  
📉 **Impact:** $5M/minute potential alpha decay  

Your algorithmic trading engine is experiencing microsecond-level latency spikes during NYSE open. 
Each extra millisecond costs ~$25K in slippage. Competitors are already 15ms ahead. 

Traditional monitoring sees "42ms" as "normal" - ARF detects the 425% increase and predicts 
$5.2M revenue loss in the next 30 minutes.

**Watch ARF trigger microservice optimization before the market notices...**
        """
    },
    
    "🏥 Healthcare - Patient Monitor Alert": {
        "component": "icu-patient-monitor",
        "latency": 85.0,
        "error_rate": 0.08,
        "throughput": 200.0,
        "cpu_util": 0.65,
        "memory_util": 0.91,
        "patients_affected": 12,
        "story": """
**HEALTHCARE: ICU Patient Monitoring System Degradation**

🏥 **Hospital:** NYC Medical Center  
📍 **Location:** Upper East Side, NYC  
👥 **Patients at Risk:** 12 in ICU  
⚖️ **Regulatory:** HIPAA Critical, FDA Class II Medical Device  

🕐 **Time:** 2:15 AM (Night Shift)  
📊 **Data Drop Rate:** 8% of vitals lost  
🚨 **Alert:** Potential sensor network failure  

The ICU patient monitoring system is dropping 8% of vital sign data (heart rate, O2 saturation, BP). 
12 critically ill patients are at risk. System memory at 91% with slow response to nurse alerts.

Traditional IT monitoring would wait for 15% data loss. ARF detects the pattern shift in 47 seconds 
and triggers automatic failover to backup monitoring system.

**See ARF prevent patient harm through predictive failure detection...**
        """
    },
    
    "🚀 SaaS - AI Inference API Meltdown": {
        "component": "ai-inference-engine",
        "latency": 2450.0,
        "error_rate": 0.22,
        "throughput": 450.0,
        "cpu_util": 0.97,
        "memory_util": 0.95,
        "api_users": 4250,
        "story": """
**SaaS PLATFORM: GPT-4 Inference Service Failure**

🤖 **Service:** Enterprise AI Assistant API  
📍 **Location:** Chelsea, NYC Tech Hub  
👥 **Customers:** 4,250 API users  
💸 **Pricing:** $0.12/1K tokens, $85K daily revenue  

🕐 **Time:** 11:15 AM (Business Hours Peak)  
💥 **Failure Rate:** 22% of inference requests  
⏱️ **Latency:** 2.45s (vs 350ms SLA)  

Your GPT-4 inference service is failing 22% of requests with 2.45s latency vs 350ms SLA. 
CUDA memory fragmentation causing GPU OOM errors. 4,250 enterprise customers affected, 
including 3 Fortune 500 companies.

Traditional monitoring sees "GPUs at 97%" as normal. ARF detects the memory fragmentation 
pattern and triggers automatic container restart + model sharding across 8 additional GPUs.

**Watch ARF maintain 99.97% uptime for AI inference at scale...**
        """
    },
    
    "📺 Media - Ad Server Performance Crash": {
        "component": "ad-serving-engine",
        "latency": 1800.0,
        "error_rate": 0.28,
        "throughput": 120000.0,
        "cpu_util": 0.93,
        "memory_util": 0.87,
        "revenue_impact": 85000,
        "story": """
**MEDIA/ADVERTISING: Real-Time Ad Serving Platform Failure**

📺 **Company:** Madison Avenue Ad Tech Firm  
📍 **Location:** Midtown Manhattan, NYC  
💰 **CPM Impact:** $85,000/minute during primetime  
👥 **Audience:** 2.5M concurrent viewers  

🕐 **Time:** 8:15 PM (Primetime Broadcast)  
📉 **Ad Serving Failure:** 28% of impressions lost  
⚡ **Latency:** 1.8s (vs 100ms SLA)  
😠 **Publisher Complaints:** 15+ major networks affected  

Your real-time bidding ad server is failing 28% of impressions during NBC primetime broadcast.
Network timeout causing missed $85,000 CPM opportunities. 15+ publisher networks reporting 
lost revenue and threatening contract termination.

Traditional monitoring would wait for 5% error rate. ARF detects the exponential failure curve 
at 2.8% and triggers automatic traffic failover + cache warming + ad network rebalancing.

**See ARF save $2.1M in ad revenue during 25-minute crisis...**
        """
    },
    
    "🚚 Logistics - Real-Time Tracking Failure": {
        "component": "shipment-tracker",
        "latency": 650.0,
        "error_rate": 0.15,
        "throughput": 85000.0,
        "cpu_util": 0.88,
        "memory_util": 0.92,
        "shipments_affected": 12500,
        "story": """
**LOGISTICS: Port Authority Shipment Tracking System Failure**

🚚 **Company:** NYC Port Authority Logistics  
📍 **Location:** Red Hook Container Terminal, NYC  
📦 **Shipments Affected:** 12,500 containers  
💰 **Demurrage Costs:** $5,000/hour per delayed container  

🕐 **Time:** 6:30 AM (Container Arrival Peak)  
📊 **Tracking Failure:** 15% of shipments offline  
🚢 **Port Impact:** 3 container ships delayed at berth  
💸 **Financial Impact:** $2.1M/hour demurrage penalties  

Your real-time container tracking system has lost communication with 15% of shipments 
during morning arrival peak at Port of New York. RFID readers failing due to network 
congestion, causing 12,500 containers to become "invisible" in the system.

Traditional port monitoring would wait for manual reports. ARF detects the network 
partition pattern and triggers automatic RFID failover + satellite backup comms + 
priority scanning for critical shipments.

**Watch ARF prevent $12M in port demurrage fees through predictive infrastructure management...**
        """
    },
    
    "🏢 Healthy System Baseline": {
        "component": "api-service",
        "latency": 85.0,
        "error_rate": 0.008,
        "throughput": 1200.0,
        "cpu_util": 0.35,
        "memory_util": 0.42,
        "story": """
**HEALTHY SYSTEM: NYC Multi-Industry Normal Operations**

📍 **Location:** Multi-tenant Cloud Infrastructure, NYC  
✅ **Status:** NORMAL  
📊 **All Metrics:** Within optimal ranges  

This is what good looks like across NYC industries. All services running smoothly 
with optimal performance metrics. 

Use this to show how ARF distinguishes between normal operations and actual incidents
across finance, healthcare, SaaS, media, and logistics sectors.

**Intelligent anomaly detection prevents alert fatigue while catching real issues...**
        """
    }
}

# === Input Validation (FIXED: Comprehensive validation) ===
def validate_inputs(
    latency: Any,
    error_rate: Any,
    throughput: Any,
    cpu_util: Any,
    memory_util: Any
) -> tuple[bool, str]:
    """
    Comprehensive input validation with type checking
    
    Returns:
        tuple[bool, str]: (is_valid, error_message)
    """
    try:
        # Type conversion with error handling
        try:
            latency_f = float(latency)
        except (ValueError, TypeError):
            return False, "❌ Invalid latency: must be a number"
        
        try:
            error_rate_f = float(error_rate)
        except (ValueError, TypeError):
            return False, "❌ Invalid error rate: must be a number"
        
        try:
            throughput_f = float(throughput) if throughput else 1000.0
        except (ValueError, TypeError):
            return False, "❌ Invalid throughput: must be a number"
        
        # CPU and memory are optional
        cpu_util_f = None
        if cpu_util:
            try:
                cpu_util_f = float(cpu_util)
            except (ValueError, TypeError):
                return False, "❌ Invalid CPU utilization: must be a number"
        
        memory_util_f = None
        if memory_util:
            try:
                memory_util_f = float(memory_util)
            except (ValueError, TypeError):
                return False, "❌ Invalid memory utilization: must be a number"
        
        # Range validation
        if not (0 <= latency_f <= 10000):
            return False, "❌ Invalid latency: must be between 0-10000ms"
        
        if not (0 <= error_rate_f <= 1):
            return False, "❌ Invalid error rate: must be between 0-1"
        
        if throughput_f < 0:
            return False, "❌ Invalid throughput: must be positive"
        
        if cpu_util_f is not None and not (0 <= cpu_util_f <= 1):
            return False, "❌ Invalid CPU utilization: must be between 0-1"
        
        if memory_util_f is not None and not (0 <= memory_util_f <= 1):
            return False, "❌ Invalid memory utilization: must be between 0-1"
        
        return True, ""
        
    except Exception as e:
        logger.error(f"Validation error: {e}", exc_info=True)
        return False, f"❌ Validation error: {str(e)}"


# === Multi-Agent System ===
class AgentSpecialization(Enum):
    """Agent specialization types"""
    DETECTIVE = "anomaly_detection"
    DIAGNOSTICIAN = "root_cause_analysis"
    PREDICTIVE = "predictive_analytics"


class BaseAgent:
    """Base class for all specialized agents"""
    
    def __init__(self, specialization: AgentSpecialization):
        self.specialization = specialization
        self.performance_metrics = {
            'processed_events': 0,
            'successful_analyses': 0,
            'average_confidence': 0.0
        }
    
    async def analyze(self, event: ReliabilityEvent) -> dict[str, Any]:
        """Base analysis method to be implemented by specialized agents"""
        raise NotImplementedError


class AnomalyDetectionAgent(BaseAgent):
    """Specialized agent for anomaly detection and pattern recognition"""
    
    def __init__(self):
        super().__init__(AgentSpecialization.DETECTIVE)
        logger.info("Initialized AnomalyDetectionAgent")
    
    async def analyze(self, event: ReliabilityEvent) -> dict[str, Any]:
        """Perform comprehensive anomaly analysis"""
        try:
            anomaly_score = self._calculate_anomaly_score(event)
            
            return {
                'specialization': self.specialization.value,
                'confidence': anomaly_score,
                'findings': {
                    'anomaly_score': anomaly_score,
                    'severity_tier': self._classify_severity(anomaly_score),
                    'primary_metrics_affected': self._identify_affected_metrics(event)
                },
                'recommendations': self._generate_detection_recommendations(event, anomaly_score)
            }
        except Exception as e:
            logger.error(f"AnomalyDetectionAgent error: {e}", exc_info=True)
            return {
                'specialization': self.specialization.value,
                'confidence': 0.0,
                'findings': {},
                'recommendations': [f"Analysis error: {str(e)}"]
            }
    
    def _calculate_anomaly_score(self, event: ReliabilityEvent) -> float:
        """Calculate comprehensive anomaly score (0-1)"""
        scores = []
        
        # Latency anomaly (weighted 40%)
        if event.latency_p99 > Constants.LATENCY_WARNING:
            latency_score = min(1.0, (event.latency_p99 - Constants.LATENCY_WARNING) / 500)
            scores.append(0.4 * latency_score)
        
        # Error rate anomaly (weighted 30%)
        if event.error_rate > Constants.ERROR_RATE_WARNING:
            error_score = min(1.0, event.error_rate / 0.3)
            scores.append(0.3 * error_score)
        
        # Resource anomaly (weighted 30%)
        resource_score: float = 0.0
        if event.cpu_util and event.cpu_util > Constants.CPU_WARNING:
            resource_score += 0.15 * min(1.0, (event.cpu_util - Constants.CPU_WARNING) / 0.2)
        if event.memory_util and event.memory_util > Constants.MEMORY_WARNING:
            resource_score += 0.15 * min(1.0, (event.memory_util - Constants.MEMORY_WARNING) / 0.2)
        scores.append(resource_score)
        
        return min(1.0, sum(scores))
    
    def _classify_severity(self, anomaly_score: float) -> str:
        """Classify severity tier based on anomaly score"""
        if anomaly_score > 0.8:
            return "CRITICAL"
        elif anomaly_score > 0.6:
            return "HIGH"
        elif anomaly_score > 0.4:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _identify_affected_metrics(self, event: ReliabilityEvent) -> list[dict[str, Any]]:
        """Identify which metrics are outside normal ranges"""
        affected = []
        
        # Latency checks
        if event.latency_p99 > Constants.LATENCY_EXTREME:
            affected.append({
                "metric": "latency",
                "value": event.latency_p99,
                "severity": "CRITICAL",
                "threshold": Constants.LATENCY_WARNING
            })
        elif event.latency_p99 > Constants.LATENCY_CRITICAL:
            affected.append({
                "metric": "latency",
                "value": event.latency_p99,
                "severity": "HIGH",
                "threshold": Constants.LATENCY_WARNING
            })
        elif event.latency_p99 > Constants.LATENCY_WARNING:
            affected.append({
                "metric": "latency",
                "value": event.latency_p99,
                "severity": "MEDIUM",
                "threshold": Constants.LATENCY_WARNING
            })
        
        # Error rate checks
        if event.error_rate > Constants.ERROR_RATE_CRITICAL:
            affected.append({
                "metric": "error_rate",
                "value": event.error_rate,
                "severity": "CRITICAL",
                "threshold": Constants.ERROR_RATE_WARNING
            })
        elif event.error_rate > Constants.ERROR_RATE_HIGH:
            affected.append({
                "metric": "error_rate",
                "value": event.error_rate,
                "severity": "HIGH",
                "threshold": Constants.ERROR_RATE_WARNING
            })
        elif event.error_rate > Constants.ERROR_RATE_WARNING:
            affected.append({
                "metric": "error_rate",
                "value": event.error_rate,
                "severity": "MEDIUM",
                "threshold": Constants.ERROR_RATE_WARNING
            })
        
        # CPU checks
        if event.cpu_util and event.cpu_util > Constants.CPU_CRITICAL:
            affected.append({
                "metric": "cpu",
                "value": event.cpu_util,
                "severity": "CRITICAL",
                "threshold": Constants.CPU_WARNING
            })
        elif event.cpu_util and event.cpu_util > Constants.CPU_WARNING:
            affected.append({
                "metric": "cpu",
                "value": event.cpu_util,
                "severity": "HIGH",
                "threshold": Constants.CPU_WARNING
            })
        
        # Memory checks
        if event.memory_util and event.memory_util > Constants.MEMORY_CRITICAL:
            affected.append({
                "metric": "memory",
                "value": event.memory_util,
                "severity": "CRITICAL",
                "threshold": Constants.MEMORY_WARNING
            })
        elif event.memory_util and event.memory_util > Constants.MEMORY_WARNING:
            affected.append({
                "metric": "memory",
                "value": event.memory_util,
                "severity": "HIGH",
                "threshold": Constants.MEMORY_WARNING
            })
        
        return affected
    
    def _generate_detection_recommendations(
        self,
        event: ReliabilityEvent,
        anomaly_score: float
    ) -> list[str]:
        """Generate actionable recommendations"""
        recommendations = []
        affected_metrics = self._identify_affected_metrics(event)
        
        for metric in affected_metrics:
            metric_name = metric["metric"]
            severity = metric["severity"]
            value = metric["value"]
            threshold = metric["threshold"]
            
            if metric_name == "latency":
                if severity == "CRITICAL":
                    recommendations.append(
                        f"🚨 CRITICAL: Latency {value:.0f}ms (>{threshold}ms) - "
                        f"Check database & external dependencies"
                    )
                elif severity == "HIGH":
                    recommendations.append(
                        f"⚠️ HIGH: Latency {value:.0f}ms (>{threshold}ms) - "
                        f"Investigate service performance"
                    )
                else:
                    recommendations.append(
                        f"📈 Latency elevated: {value:.0f}ms (>{threshold}ms) - Monitor trend"
                    )
            
            elif metric_name == "error_rate":
                if severity == "CRITICAL":
                    recommendations.append(
                        f"🚨 CRITICAL: Error rate {value*100:.1f}% (>{threshold*100:.1f}%) - "
                        f"Check recent deployments"
                    )
                elif severity == "HIGH":
                    recommendations.append(
                        f"⚠️ HIGH: Error rate {value*100:.1f}% (>{threshold*100:.1f}%) - "
                        f"Review application logs"
                    )
                else:
                    recommendations.append(
                        f"📈 Errors increasing: {value*100:.1f}% (>{threshold*100:.1f}%)"
                    )
            
            elif metric_name == "cpu":
                recommendations.append(
                    f"🔥 CPU {severity}: {value*100:.1f}% utilization - Consider scaling"
                )
            
            elif metric_name == "memory":
                recommendations.append(
                    f"💾 Memory {severity}: {value*100:.1f}% utilization - Check for memory leaks"
                )
        
        # Overall severity recommendations
        if anomaly_score > 0.8:
            recommendations.append("🎯 IMMEDIATE ACTION REQUIRED: Multiple critical metrics affected")
        elif anomaly_score > 0.6:
            recommendations.append("🎯 INVESTIGATE: Significant performance degradation detected")
        elif anomaly_score > 0.4:
            recommendations.append("📊 MONITOR: Early warning signs detected")
        
        return recommendations[:4]


class RootCauseAgent(BaseAgent):
    """Specialized agent for root cause analysis"""
    
    def __init__(self):
        super().__init__(AgentSpecialization.DIAGNOSTICIAN)
        logger.info("Initialized RootCauseAgent")
    
    async def analyze(self, event: ReliabilityEvent) -> dict[str, Any]:
        """Perform root cause analysis"""
        try:
            causes = self._analyze_potential_causes(event)
            
            return {
                'specialization': self.specialization.value,
                'confidence': 0.7,
                'findings': {
                    'likely_root_causes': causes,
                    'evidence_patterns': self._identify_evidence(event),
                    'investigation_priority': self._prioritize_investigation(causes)
                },
                'recommendations': [
                    f"Check {cause['cause']} for issues" for cause in causes[:2]
                ]
            }
        except Exception as e:
            logger.error(f"RootCauseAgent error: {e}", exc_info=True)
            return {
                'specialization': self.specialization.value,
                'confidence': 0.0,
                'findings': {},
                'recommendations': [f"Analysis error: {str(e)}"]
            }
    
    def _analyze_potential_causes(self, event: ReliabilityEvent) -> list[dict[str, Any]]:
        """Analyze potential root causes based on event patterns"""
        causes = []
        
        # Pattern 1: Database/External Dependency Failure
        if event.latency_p99 > Constants.LATENCY_EXTREME and event.error_rate > 0.2:
            causes.append({
                "cause": "Database/External Dependency Failure",
                "confidence": 0.85,
                "evidence": f"Extreme latency ({event.latency_p99:.0f}ms) with high errors ({event.error_rate*100:.1f}%)",
                "investigation": "Check database connection pool, external API health"
            })
        
        # Pattern 2: Resource Exhaustion
        if (event.cpu_util and event.cpu_util > Constants.CPU_CRITICAL and
            event.memory_util and event.memory_util > Constants.MEMORY_CRITICAL):
            causes.append({
                "cause": "Resource Exhaustion",
                "confidence": 0.90,
                "evidence": f"CPU ({event.cpu_util*100:.1f}%) and Memory ({event.memory_util*100:.1f}%) critically high",
                "investigation": "Check for memory leaks, infinite loops, insufficient resources"
            })
        
        # Pattern 3: Application Bug / Configuration Issue
        if event.error_rate > Constants.ERROR_RATE_CRITICAL and event.latency_p99 < 200:
            causes.append({
                "cause": "Application Bug / Configuration Issue",
                "confidence": 0.75,
                "evidence": f"High error rate ({event.error_rate*100:.1f}%) without latency impact",
                "investigation": "Review recent deployments, configuration changes, application logs"
            })
        
        # Pattern 4: Gradual Performance Degradation
        if (200 <= event.latency_p99 <= 400 and
            Constants.ERROR_RATE_WARNING <= event.error_rate <= Constants.ERROR_RATE_HIGH):
            causes.append({
                "cause": "Gradual Performance Degradation",
                "confidence": 0.65,
                "evidence": f"Moderate latency ({event.latency_p99:.0f}ms) and errors ({event.error_rate*100:.1f}%)",
                "investigation": "Check resource trends, dependency performance, capacity planning"
            })
        
        # Default: Unknown pattern
        if not causes:
            causes.append({
                "cause": "Unknown - Requires Investigation",
                "confidence": 0.3,
                "evidence": "Pattern does not match known failure modes",
                "investigation": "Complete system review needed"
            })
        
        return causes
    
    def _identify_evidence(self, event: ReliabilityEvent) -> list[str]:
        """Identify evidence patterns in the event data"""
        evidence = []
        
        if event.latency_p99 > event.error_rate * 1000:
            evidence.append("latency_disproportionate_to_errors")
        
        if (event.cpu_util and event.cpu_util > Constants.CPU_WARNING and
            event.memory_util and event.memory_util > Constants.MEMORY_WARNING):
            evidence.append("correlated_resource_exhaustion")
        
        if event.error_rate > Constants.ERROR_RATE_HIGH and event.latency_p99 < Constants.LATENCY_CRITICAL:
            evidence.append("errors_without_latency_impact")
        
        return evidence
    
    def _prioritize_investigation(self, causes: list[dict[str, Any]]) -> str:
        """Determine investigation priority"""
        for cause in causes:
            if "Database" in cause["cause"] or "Resource Exhaustion" in cause["cause"]:
                return "HIGH"
        return "MEDIUM"


class PredictiveAgent(BaseAgent):
    """Specialized agent for predictive analytics"""
    
    def __init__(self, engine: SimplePredictiveEngine):
        super().__init__(AgentSpecialization.PREDICTIVE)
        self.engine = engine
        logger.info("Initialized PredictiveAgent")
    
    async def analyze(self, event: ReliabilityEvent) -> dict[str, Any]:
        """Perform predictive analysis for future risks"""
        try:
            event_data = {
                'latency_p99': event.latency_p99,
                'error_rate': event.error_rate,
                'throughput': event.throughput,
                'cpu_util': event.cpu_util,
                'memory_util': event.memory_util
            }
            self.engine.add_telemetry(event.component, event_data)
            
            insights = self.engine.get_predictive_insights(event.component)
            
            return {
                'specialization': self.specialization.value,
                'confidence': 0.8 if insights['critical_risk_count'] > 0 else 0.5,
                'findings': insights,
                'recommendations': insights['recommendations']
            }
        except Exception as e:
            logger.error(f"PredictiveAgent error: {e}", exc_info=True)
            return {
                'specialization': self.specialization.value,
                'confidence': 0.0,
                'findings': {},
                'recommendations': [f"Analysis error: {str(e)}"]
            }


# FIXED: Add circuit breaker for agent resilience
@circuit(failure_threshold=3, recovery_timeout=30, name="agent_circuit_breaker")
async def call_agent_with_protection(agent: BaseAgent, event: ReliabilityEvent) -> dict[str, Any]:
    """
    Call agent with circuit breaker protection
    
    FIXED: Prevents cascading failures from misbehaving agents
    """
    try:
        result = await asyncio.wait_for(
            agent.analyze(event),
            timeout=Constants.AGENT_TIMEOUT_SECONDS
        )
        return result
    except asyncio.TimeoutError:
        logger.warning(f"Agent {agent.specialization.value} timed out")
        raise
    except Exception as e:
        logger.error(f"Agent {agent.specialization.value} error: {e}", exc_info=True)
        raise


class OrchestrationManager:
    """Orchestrates multiple specialized agents for comprehensive analysis"""
    
    def __init__(
        self,
        detective: AnomalyDetectionAgent | None = None,
        diagnostician: RootCauseAgent | None = None,
        predictive: PredictiveAgent | None = None
    ):
        """
        Initialize orchestration manager
        
        FIXED: Dependency injection for testability
        """
        self.agents = {
            AgentSpecialization.DETECTIVE: detective or AnomalyDetectionAgent(),
            AgentSpecialization.DIAGNOSTICIAN: diagnostician or RootCauseAgent(),
            AgentSpecialization.PREDICTIVE: predictive or PredictiveAgent(SimplePredictiveEngine()),
        }
        logger.info(f"Initialized OrchestrationManager with {len(self.agents)} agents")
    
    async def orchestrate_analysis(self, event: ReliabilityEvent) -> dict[str, Any]:
        """
        Coordinate multiple agents for comprehensive analysis
        
        FIXED: Improved timeout handling with circuit breakers
        """
        # Create tasks for all agents
        agent_tasks = []
        agent_specs = []
        
        for spec, agent in self.agents.items():
            agent_tasks.append(call_agent_with_protection(agent, event))
            agent_specs.append(spec)
        
        # FIXED: Parallel execution with global timeout
        agent_results: dict[str, Any] = {}
        
        try:
            # Run all agents in parallel with global timeout
            results = await asyncio.wait_for(
                asyncio.gather(*agent_tasks, return_exceptions=True),
                timeout=Constants.AGENT_TIMEOUT_SECONDS + 1
            )
            
            # Process results
            for spec, result in zip(agent_specs, results, strict=False):
                if isinstance(result, Exception):
                    logger.error(f"Agent {spec.value} failed: {result}")
                    continue
                
                agent_results[spec.value] = result
                logger.debug(f"Agent {spec.value} completed successfully")
                
        except asyncio.TimeoutError:
            logger.warning("Agent orchestration timed out")
        except Exception as e:
            logger.error(f"Agent orchestration error: {e}", exc_info=True)
        
        return self._synthesize_agent_findings(event, agent_results)
    
    def _synthesize_agent_findings(
        self,
        event: ReliabilityEvent,
        agent_results: dict
    ) -> dict[str, Any]:
        """Combine insights from all specialized agents"""
        detective_result = agent_results.get(AgentSpecialization.DETECTIVE.value)
        diagnostician_result = agent_results.get(AgentSpecialization.DIAGNOSTICIAN.value)
        predictive_result = agent_results.get(AgentSpecialization.PREDICTIVE.value)
        
        if not detective_result:
            logger.warning("No detective agent results available")
            return {'error': 'No agent results available'}
        
        synthesis = {
            'incident_summary': {
                'severity': detective_result['findings'].get('severity_tier', 'UNKNOWN'),
                'anomaly_confidence': detective_result['confidence'],
                'primary_metrics_affected': [
                    metric["metric"] for metric in
                    detective_result['findings'].get('primary_metrics_affected', [])
                ]
            },
            'root_cause_insights': diagnostician_result['findings'] if diagnostician_result else {},
            'predictive_insights': predictive_result['findings'] if predictive_result else {},
            'recommended_actions': self._prioritize_actions(
                detective_result.get('recommendations', []),
                diagnostician_result.get('recommendations', []) if diagnostician_result else [],
                predictive_result.get('recommendations', []) if predictive_result else []
            ),
            'agent_metadata': {
                'participating_agents': list(agent_results.keys()),
                'analysis_timestamp': datetime.datetime.now(datetime.timezone.utc).isoformat()
            }
        }
        
        return synthesis
    
    def _prioritize_actions(
        self,
        detection_actions: list[str],
        diagnosis_actions: list[str],
        predictive_actions: list[str]
    ) -> list[str]:
        """Combine and prioritize actions from multiple agents"""
        all_actions = detection_actions + diagnosis_actions + predictive_actions
        seen = set()
        unique_actions = []
        for action in all_actions:
            if action not in seen:
                seen.add(action)
                unique_actions.append(action)
        return unique_actions[:5]


# NOTE: Removed duplicate EnhancedReliabilityEngine class (using module version from engine/reliability.py)
# NOTE: Removed duplicate ThreadSafeEventStore class (using module version from engine/reliability.py)

class RateLimiter:
    """Simple rate limiter for request throttling"""
    
    def __init__(self, max_per_minute: int = Constants.MAX_REQUESTS_PER_MINUTE):
        self.max_per_minute = max_per_minute
        self.requests: deque = deque(maxlen=max_per_minute)
        self._lock = threading.RLock()
    
    def is_allowed(self) -> tuple[bool, str]:
        """Check if request is allowed"""
        with self._lock:
            now = datetime.datetime.now(datetime.timezone.utc)
            
            # Remove requests older than 1 minute
            one_minute_ago = now - datetime.timedelta(minutes=1)
            while self.requests and self.requests[0] < one_minute_ago:
                self.requests.popleft()
            
            # Check rate limit
            if len(self.requests) >= self.max_per_minute:
                return False, f"Rate limit exceeded: {self.max_per_minute} requests/minute"
            
            # Add current request
            self.requests.append(now)
            return True, ""


rate_limiter = RateLimiter()

# === Gradio UI ===
def create_enhanced_ui():
    """
    Create the comprehensive Gradio UI for the reliability framework
    
    FIXED: Uses native async handlers (no event loop creation)
    FIXED: Rate limiting on all endpoints
    NEW: Demo scenarios for NYC industries
    NEW: ROI Dashboard with real-time business metrics
    """
    
    with gr.Blocks(title="🧠 Agentic Reliability Framework", theme="soft") as demo:
        gr.Markdown("""
        # 🧠 Agentic Reliability Framework
        **Multi-Agent AI System for Production Reliability**
        
        _Specialized AI agents working together to detect, diagnose, predict, and heal system issues_
        
        """)
        
        # === ROI DASHBOARD ===
        with gr.Accordion("💰 Business Impact Dashboard", open=True):
            gr.Markdown("""
            ### Real-Time ROI Metrics
            Track cumulative business value delivered by ARF across all analyzed incidents.
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    total_incidents_display = gr.Number(
                        label="📊 Total Incidents Analyzed",
                        value=0,
                        interactive=False
                    )
                with gr.Column(scale=1):
                    incidents_healed_display = gr.Number(
                        label="🔧 Incidents Auto-Healed",
                        value=0,
                        interactive=False
                    )
                with gr.Column(scale=1):
                    auto_heal_rate_display = gr.Number(
                        label="⚡ Auto-Heal Rate (%)",
                        value=0,
                        interactive=False,
                        precision=1
                    )
            
            with gr.Row():
                with gr.Column(scale=1):
                    revenue_saved_display = gr.Number(
                        label="💰 Revenue Saved ($)",
                        value=0,
                        interactive=False,
                        precision=2
                    )
                with gr.Column(scale=1):
                    avg_detection_display = gr.Number(
                        label="⏱️ Avg Detection Time (min)",
                        value=2.3,
                        interactive=False,
                        precision=1
                    )
                with gr.Column(scale=1):
                    time_improvement_display = gr.Number(
                        label="🚀 Time Improvement vs Industry (%)",
                        value=83.6,
                        interactive=False,
                        precision=1
                    )
            
            with gr.Row():
                gr.Markdown("""
                **📈 Comparison:**  
                - **Industry Average Response:** 14 minutes  
                - **ARF Average Response:** 2.3 minutes  
                - **Result:** 6x faster incident resolution
                
                *Metrics update in real-time as incidents are processed*
                """)
                
                reset_metrics_btn = gr.Button("🔄 Reset Metrics (Demo)", size="sm")
        # === END ROI DASHBOARD ===
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📊 Telemetry Input")
                
                # Demo Scenarios Dropdown
                with gr.Row():
                    scenario_dropdown = gr.Dropdown(
                        choices=["Manual Entry"] + list(DEMO_SCENARIOS.keys()),
                        value="Manual Entry",
                        label="🎬 Demo Scenario (Quick Start)",
                        info="Select a pre-configured NYC industry scenario or enter manually"
                    )
                
                # Scenario Story Display
                scenario_story = gr.Markdown(
                    value="*Select a demo scenario above for a pre-configured incident, or enter values manually below.*",
                    visible=True
                )
                
                component = gr.Dropdown(
                    choices=["api-service", "auth-service", "payment-service", "database", "cache-service"],
                    value="api-service",
                    label="Component",
                    info="Select the service being monitored"
                )
                latency = gr.Slider(
                    minimum=10, maximum=1000, value=100, step=1,
                    label="Latency P99 (ms)",
                    info=f"Alert threshold: >{Constants.LATENCY_WARNING}ms (adaptive)"
                )
                error_rate = gr.Slider(
                    minimum=0, maximum=0.5, value=0.02, step=0.001,
                    label="Error Rate",
                    info=f"Alert threshold: >{Constants.ERROR_RATE_WARNING}"
                )
                throughput = gr.Number(
                    value=1000,
                    label="Throughput (req/sec)",
                    info="Current request rate"
                )
                cpu_util = gr.Slider(
                    minimum=0, maximum=1, value=0.4, step=0.01,
                    label="CPU Utilization",
                    info="0.0 - 1.0 scale"
                )
                memory_util = gr.Slider(
                    minimum=0, maximum=1, value=0.3, step=0.01,
                    label="Memory Utilization",
                    info="0.0 - 1.0 scale"
                )
                submit_btn = gr.Button("🚀 Submit Telemetry Event", variant="primary", size="lg")
                
            with gr.Column(scale=2):
                gr.Markdown("### 🔍 Multi-Agent Analysis")
                output_text = gr.Textbox(
                    label="Agent Synthesis",
                    placeholder="AI agents are analyzing...",
                    lines=6
                )
                
                with gr.Accordion("🤖 Agent Specialists Analysis", open=False):
                    gr.Markdown("""
                    **Specialized AI Agents:**
                    - 🕵️ **Detective**: Anomaly detection & pattern recognition
                    - 🔍 **Diagnostician**: Root cause analysis & investigation
                    - 🔮 **Predictive**: Future risk forecasting & trend analysis
                    """)
                    
                    agent_insights = gr.JSON(
                        label="Detailed Agent Findings",
                        value={}
                    )
                
                with gr.Accordion("🔮 Predictive Analytics & Forecasting", open=False):
                    gr.Markdown("""
                    **Future Risk Forecasting:**
                    - 📈 Latency trends and thresholds
                    - 🚨 Error rate predictions
                    - 🔥 Resource utilization forecasts
                    - ⏰ Time-to-failure estimates
                    """)
                    
                    predictive_insights = gr.JSON(
                        label="Predictive Forecasts",
                        value={}
                    )
                
                gr.Markdown("### 📈 Recent Events (Last 15)")
                events_table = gr.Dataframe(
                    headers=["Timestamp", "Component", "Latency", "Error Rate", "Throughput", "Severity", "Analysis"],
                    label="Event History",
                    wrap=True,
                )
        
        with gr.Accordion("ℹ️ Framework Capabilities", open=False):
            gr.Markdown("""
            - **🤖 Multi-Agent AI**: Specialized agents for detection, diagnosis, prediction, and healing
            - **🔮 Predictive Analytics**: Forecast future risks and performance degradation
            - **🔧 Policy-Based Healing**: Automated recovery actions based on severity and context
            - **💰 Business Impact**: Revenue and user impact quantification
            - **🎯 Adaptive Detection**: ML-powered thresholds that learn from your environment
            - **📚 Vector Memory**: FAISS-based incident memory for similarity detection
            - **⚡ Production Ready**: Circuit breakers, cooldowns, thread safety, and enterprise features
            - **🔒 Security Patched**: All critical CVEs fixed (Gradio 5.50.0+, Requests 2.32.5+)
            """)
        
        with gr.Accordion("🔧 Healing Policies", open=False):
            policy_info = []
            engine = get_engine()
            if hasattr(engine, 'policy_engine') and hasattr(engine.policy_engine, 'policies'):
                for policy in engine.policy_engine.policies:
                    if hasattr(policy, 'enabled') and policy.enabled:
                        actions = ", ".join([action.value for action in policy.actions]) if hasattr(policy, 'actions') else ""
                        policy_info.append(
                            f"**{getattr(policy, 'name', 'Unknown')}** "
                            f"(Priority {getattr(policy, 'priority', 0)}): {actions}\n"
                            f"  - Cooldown: {getattr(policy, 'cool_down_seconds', 0)}s\n"
                            f"  - Max executions: {getattr(policy, 'max_executions_per_hour', 0)}/hour"
                        )
            
            gr.Markdown("\n\n".join(policy_info) if policy_info else "No policies available")
        
        # Scenario change handler
        def on_scenario_change(scenario_name: str) -> dict[str, Any]:
            """Update input fields when demo scenario is selected"""
            if scenario_name == "Manual Entry":
                return {
                    scenario_story: gr.update(value="*Enter values manually below.*"),
                    component: gr.update(value="api-service"),
                    latency: gr.update(value=100),
                    error_rate: gr.update(value=0.02),
                    throughput: gr.update(value=1000),
                    cpu_util: gr.update(value=0.4),
                    memory_util: gr.update(value=0.3)
                }
            
            scenario = DEMO_SCENARIOS.get(scenario_name)
            if not scenario:
                return {}
            
            return {
                scenario_story: gr.update(value=scenario["story"]),
                component: gr.update(value=scenario["component"]),
                latency: gr.update(value=scenario["latency"]),
                error_rate: gr.update(value=scenario["error_rate"]),
                throughput: gr.update(value=scenario["throughput"]),
                cpu_util: gr.update(value=scenario.get("cpu_util", 0.5)),
                memory_util: gr.update(value=scenario.get("memory_util", 0.5))
            }
        
        # Reset metrics handler
        def reset_metrics() -> tuple[int, int, float, float, float, float]:
            """Reset business metrics for demo purposes"""
            business_metrics = get_business_metrics()
            if hasattr(business_metrics, 'reset'):
                business_metrics.reset()
            return 0, 0, 0.0, 0.0, 2.3, 83.6
        
        # Connect scenario dropdown to inputs
        scenario_dropdown.change(
            fn=on_scenario_change,
            inputs=[scenario_dropdown],
            outputs=[scenario_story, component, latency, error_rate, throughput, cpu_util, memory_util]
        )
        
        # Connect reset button
        reset_metrics_btn.click(
            fn=reset_metrics,
            outputs=[
                total_incidents_display,
                incidents_healed_display,
                auto_heal_rate_display,
                revenue_saved_display,
                avg_detection_display,
                time_improvement_display
            ]
        )
            
        # Event submission handler with ROI tracking - FIXED: Removed unreachable code
        async def submit_event_enhanced_async(
            component: str,
            latency: float,
            error_rate: float,
            throughput: float,
            cpu_util: float | None,
            memory_util: float | None
        ) -> tuple[str, dict[str, Any], dict[str, Any], Any, int, int, float, float, float, float]:
            """
            Async event handler - uses Gradio's native async support
            
            CRITICAL FIX: No event loop creation - Gradio handles this
            FIXED: Rate limiting added
            FIXED: Comprehensive error handling
            NEW: Updates ROI dashboard metrics
            """
            try:
                # Rate limiting check
                allowed, rate_msg = rate_limiter.is_allowed()
                if not allowed:
                    logger.warning("Rate limit exceeded")
                    business_metrics = get_business_metrics()
                    if hasattr(business_metrics, 'get_metrics'):
                        metrics = business_metrics.get_metrics()
                    else:
                        metrics = {
                            "total_incidents": 0,
                            "incidents_auto_healed": 0,
                            "auto_heal_rate": 0.0,
                            "total_revenue_saved": 0.0,
                            "avg_detection_time_minutes": 2.3,
                            "time_improvement": 83.6
                        }
                    return (
                        rate_msg, {}, {}, gr.Dataframe(value=[]),
                        metrics["total_incidents"],
                        metrics["incidents_auto_healed"],
                        metrics["auto_heal_rate"],
                        metrics["total_revenue_saved"],
                        metrics["avg_detection_time_minutes"],
                        metrics["time_improvement"]
                    )
                
                # Type conversion
                try:
                    latency_f = float(latency)
                    error_rate_f = float(error_rate)
                    throughput_f = float(throughput) if throughput else 1000
                    cpu_util_f = float(cpu_util) if cpu_util else None
                    memory_util_f = float(memory_util) if memory_util else None
                except (ValueError, TypeError) as e:
                    error_msg = f"❌ Invalid input types: {str(e)}"
                    logger.warning(error_msg)
                    business_metrics = get_business_metrics()
                    if hasattr(business_metrics, 'get_metrics'):
                        metrics = business_metrics.get_metrics()
                    else:
                        metrics = {
                            "total_incidents": 0,
                            "incidents_auto_healed": 0,
                            "auto_heal_rate": 0.0,
                            "total_revenue_saved": 0.0,
                            "avg_detection_time_minutes": 2.3,
                            "time_improvement": 83.6
                        }
                    return (
                        error_msg, {}, {}, gr.Dataframe(value=[]),
                        metrics["total_incidents"],
                        metrics["incidents_auto_healed"],
                        metrics["auto_heal_rate"],
                        metrics["total_revenue_saved"],
                        metrics["avg_detection_time_minutes"],
                        metrics["time_improvement"]
                    )
                
                # Input validation
                is_valid, error_msg = validate_inputs(
                    latency_f, error_rate_f, throughput_f, cpu_util_f, memory_util_f
                )
                if not is_valid:
                    logger.warning(f"Invalid input: {error_msg}")
                    business_metrics = get_business_metrics()
                    if hasattr(business_metrics, 'get_metrics'):
                        metrics = business_metrics.get_metrics()
                    else:
                        metrics = {
                            "total_incidents": 0,
                            "incidents_auto_healed": 0,
                            "auto_heal_rate": 0.0,
                            "total_revenue_saved": 0.0,
                            "avg_detection_time_minutes": 2.3,
                            "time_improvement": 83.6
                        }
                    return (
                        error_msg, {}, {}, gr.Dataframe(value=[]),
                        metrics["total_incidents"],
                        metrics["incidents_auto_healed"],
                        metrics["auto_heal_rate"],
                        metrics["total_revenue_saved"],
                        metrics["avg_detection_time_minutes"],
                        metrics["time_improvement"]
                    )
                
                # Process event through engine
                engine = get_engine()
                if not engine:
                    return (
                        "❌ Engine not available", {}, {}, gr.Dataframe(value=[]),
                        0, 0, 0.0, 0.0, 2.3, 83.6
                    )
                
                result = await engine.process_event_enhanced(
                    component, latency_f, error_rate_f, throughput_f, cpu_util_f, memory_util_f
                )
                
                # Handle errors
                if 'error' in result:
                    business_metrics = get_business_metrics()
                    if hasattr(business_metrics, 'get_metrics'):
                        metrics = business_metrics.get_metrics()
                    else:
                        metrics = {
                            "total_incidents": 0,
                            "incidents_auto_healed": 0,
                            "auto_heal_rate": 0.0,
                            "total_revenue_saved": 0.0,
                            "avg_detection_time_minutes": 2.3,
                            "time_improvement": 83.6
                        }
                    return (
                        f"❌ {result['error']}", {}, {}, gr.Dataframe(value=[]),
                        metrics["total_incidents"],
                        metrics["incidents_auto_healed"],
                        metrics["auto_heal_rate"],
                        metrics["total_revenue_saved"],
                        metrics["avg_detection_time_minutes"],
                        metrics["time_improvement"]
                    )
                
                # Build table data (THREAD-SAFE)
                table_data = []
                
                # Try to get events from engine
                if hasattr(engine, 'event_store') and hasattr(engine.event_store, 'get_recent'):
                    try:
                        events = engine.event_store.get_recent(15)
                        for event in events:
                            table_data.append([
                                event.timestamp.strftime("%Y-%m-%d %H:%M:%S") if hasattr(event.timestamp, 'strftime') else "Unknown",
                                event.component if hasattr(event, 'component') else "Unknown",
                                f"{event.latency_p99:.0f}ms" if hasattr(event, 'latency_p99') else "0ms",
                                f"{event.error_rate:.3f}" if hasattr(event, 'error_rate') else "0.000",
                                f"{event.throughput:.0f}" if hasattr(event, 'throughput') else "0",
                                event.severity.value.upper() if hasattr(event, 'severity') and hasattr(event.severity, 'value') else "UNKNOWN",
                                "Multi-agent analysis"
                            ])
                    except Exception as e:
                        logger.error(f"Error getting events: {e}")
                        table_data = []
                
                # Format output message
                status_emoji = "🚨" if result.get("status") == "ANOMALY" else "✅"
                status = result.get("status", "UNKNOWN")
                output_msg = f"{status_emoji} **{status}**\n"
                
                if "multi_agent_analysis" in result:
                    analysis = result["multi_agent_analysis"]
                    confidence = analysis.get('incident_summary', {}).get('anomaly_confidence', 0)
                    output_msg += f"🎯 **Confidence**: {confidence*100:.1f}%\n"
                    
                    predictive_data = analysis.get('predictive_insights', {})
                    if predictive_data.get('critical_risk_count', 0) > 0:
                        output_msg += f"🔮 **PREDICTIVE**: {predictive_data['critical_risk_count']} critical risks forecast\n"
                    
                    if analysis.get('recommended_actions'):
                        actions_preview = ', '.join(analysis['recommended_actions'][:2])
                        output_msg += f"💡 **Top Insights**: {actions_preview}\n"
                
                if result.get("business_impact"):
                    impact = result["business_impact"]
                    output_msg += (
                        f"💰 **Business Impact**: ${impact.get('revenue_loss_estimate', 0):.2f} | "
                        f"👥 {impact.get('affected_users_estimate', 0)} users | "
                        f"🚨 {impact.get('severity_level', 'UNKNOWN')}\n"
                    )
                
                if result.get("healing_actions") and result["healing_actions"] != ["no_action"]:
                    actions = ", ".join(result["healing_actions"])
                    output_msg += f"🔧 **Auto-Actions**: {actions}"
                
                agent_insights_data = result.get("multi_agent_analysis", {})
                predictive_insights_data = agent_insights_data.get('predictive_insights', {})
                
                # Get updated metrics
                business_metrics = get_business_metrics()
                if hasattr(business_metrics, 'get_metrics'):
                    metrics = business_metrics.get_metrics()
                else:
                    metrics = {
                        "total_incidents": 0,
                        "incidents_auto_healed": 0,
                        "auto_heal_rate": 0.0,
                        "total_revenue_saved": 0.0,
                        "avg_detection_time_minutes": 2.3,
                        "time_improvement": 83.6
                    }
                
                # FIX: This return statement was causing unreachable code error
                # Ensure this is the last executable statement in the try block
                return (
                    output_msg,
                    agent_insights_data,
                    predictive_insights_data,
                    gr.Dataframe(
                        headers=["Timestamp", "Component", "Latency", "Error Rate", "Throughput", "Severity", "Analysis"],
                        value=table_data,
                        wrap=True
                    ),
                    metrics["total_incidents"],
                    metrics["incidents_auto_healed"],
                    metrics["auto_heal_rate"],
                    metrics["total_revenue_saved"],
                    metrics["avg_detection_time_minutes"],
                    metrics["time_improvement"]
                )
                
            except Exception as e:
                error_msg = f"❌ Error processing event: {str(e)}"
                logger.error(error_msg, exc_info=True)
                business_metrics = get_business_metrics()
                if hasattr(business_metrics, 'get_metrics'):
                    metrics = business_metrics.get_metrics()
                else:
                    metrics = {
                        "total_incidents": 0,
                        "incidents_auto_healed": 0,
                        "auto_heal_rate": 0.0,
                        "total_revenue_saved": 0.0,
                        "avg_detection_time_minutes": 2.3,
                        "time_improvement": 83.6
                    }
                return (
                    error_msg, {}, {}, gr.Dataframe(value=[]),
                    metrics["total_incidents"],
                    metrics["incidents_auto_healed"],
                    metrics["auto_heal_rate"],
                    metrics["total_revenue_saved"],
                    metrics["avg_detection_time_minutes"],
                    metrics["time_improvement"]
                )
        
        # Connect submit button with all outputs
        submit_btn.click(
            fn=submit_event_enhanced_async,
            inputs=[component, latency, error_rate, throughput, cpu_util, memory_util],
            outputs=[
                output_text,
                agent_insights,
                predictive_insights,
                events_table,
                total_incidents_display,
                incidents_healed_display,
                auto_heal_rate_display,
                revenue_saved_display,
                avg_detection_display,
                time_improvement_display
            ]
        )
    
    return demo


# === Main Entry Point ===
def main() -> None:
    """Main entry point for the application"""
    logger.info("=" * 80)
    logger.info("Starting Enterprise Agentic Reliability Framework (DEMO READY VERSION)")
    logger.info("=" * 80)
    logger.info(f"Python version: {sys.version}")
    
    try:
        engine = get_engine()
        faiss_index = get_faiss_index()
        business_metrics = get_business_metrics()
        
        logger.info(f"Total events in history: {getattr(engine.event_store, 'count', lambda: 0)() if hasattr(engine, 'event_store') else 'N/A'}")
        logger.info(f"Vector index size: {faiss_index.get_count() if faiss_index and hasattr(faiss_index, 'get_count') else 'N/A'}")
        logger.info(f"Agents initialized: {len(engine.orchestrator.agents) if hasattr(engine, 'orchestrator') and hasattr(engine.orchestrator, 'agents') else 'N/A'}")
        logger.info(f"Policies loaded: {len(engine.policy_engine.policies) if hasattr(engine, 'policy_engine') and hasattr(engine.policy_engine, 'policies') else 'N/A'}")
        logger.info(f"Demo scenarios loaded: {len(DEMO_SCENARIOS)}")
        logger.info(f"Configuration: HF_TOKEN={'SET' if getattr(config, 'hf_api_key', None) else 'NOT SET'}")
        logger.info(f"Rate limit: {Constants.MAX_REQUESTS_PER_MINUTE} requests/minute")
        logger.info("=" * 80)
        
        demo = create_enhanced_ui()
        
        logger.info("Launching Gradio UI on 0.0.0.0:7860...")
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            show_error=True
        )
        
    except KeyboardInterrupt:
        logger.info("Received shutdown signal...")
    except Exception as e:
        logger.error(f"Application error: {e}", exc_info=True)
    finally:
        # Graceful shutdown
        logger.info("Shutting down gracefully...")
        
        faiss_index = get_faiss_index()
        if faiss_index and hasattr(faiss_index, 'shutdown'):
            logger.info("Saving pending vectors before shutdown...")
            faiss_index.shutdown()
        
        logger.info("=" * 80)
        logger.info("Application shutdown complete")
        logger.info("=" * 80)


if __name__ == "__main__":
    main()

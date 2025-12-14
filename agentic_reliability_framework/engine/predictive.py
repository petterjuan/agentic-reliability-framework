"""
Predictive Engine for forecasting and trend analysis
Extracted from app.py for modularity
"""

import numpy as np
import datetime
import threading
import logging
from typing import Dict, List, Optional, Tuple, Literal, Any
from collections import deque

from ..models import ForecastResult
from ..config import config

logger = logging.getLogger(__name__)


class SimplePredictiveEngine:
    """
    Lightweight forecasting engine with proper constant usage
    
    Extracted from app.py
    """
    
    def __init__(self, history_window: int = 50):
        self.history_window = history_window
        self.service_history: Dict[str, deque] = {}
        self.prediction_cache: Dict[str, Tuple[ForecastResult, datetime.datetime]] = {}
        self.max_cache_age = datetime.timedelta(minutes=config.cache_expiry_minutes)
        self._lock = threading.RLock()
        logger.info(f"Initialized SimplePredictiveEngine with history_window={history_window}")
    
    def add_telemetry(self, service: str, event_data: Dict) -> None:
        """Add telemetry data to service history"""
        with self._lock:
            if service not in self.service_history:
                self.service_history[service] = deque(maxlen=self.history_window)
            
            telemetry_point = {
                'timestamp': datetime.datetime.now(datetime.timezone.utc),
                'latency': event_data.get('latency_p99', 0),
                'error_rate': event_data.get('error_rate', 0),
                'throughput': event_data.get('throughput', 0),
                'cpu_util': event_data.get('cpu_util'),
                'memory_util': event_data.get('memory_util')
            }
            
            self.service_history[service].append(telemetry_point)
            self._clean_cache()
    
    def _clean_cache(self) -> None:
        """Remove expired entries from prediction cache"""
        now = datetime.datetime.now(datetime.timezone.utc)
        expired = [k for k, (_, ts) in self.prediction_cache.items() 
                   if now - ts > self.max_cache_age]
        for k in expired:
            del self.prediction_cache[k]
        
        if expired:
            logger.debug(f"Cleaned {len(expired)} expired cache entries")
    
    def forecast_service_health(
        self,
        service: str,
        lookahead_minutes: int = config.forecast_lookahead_minutes
    ) -> List[ForecastResult]:
        """Forecast service health metrics"""
        with self._lock:
            if service not in self.service_history or \
               len(self.service_history[service]) < config.forecast_min_data_points:
                return []
            
            history = list(self.service_history[service])
        
        forecasts = []
        
        # Forecast latency
        latency_forecast = self._forecast_latency(history, lookahead_minutes)
        if latency_forecast:
            forecasts.append(latency_forecast)
        
        # Forecast error rate
        error_forecast = self._forecast_error_rate(history, lookahead_minutes)
        if error_forecast:
            forecasts.append(error_forecast)
        
        # Forecast resource utilization
        resource_forecasts = self._forecast_resources(history, lookahead_minutes)
        forecasts.extend(resource_forecasts)
        
        # Cache results
        with self._lock:
            for forecast in forecasts:
                cache_key = f"{service}_{forecast.metric}"
                self.prediction_cache[cache_key] = (forecast, datetime.datetime.now(datetime.timezone.utc))
        
        return forecasts
    
    def _get_trend_literal(self, slope: float) -> Literal["increasing", "decreasing", "stable"]:
        """Safely convert slope to trend literal"""
        if slope > config.slope_threshold_increasing:
            return "increasing"
        elif slope < config.slope_threshold_decreasing:
            return "decreasing"
        return "stable"
    
    def _get_risk_literal(self, metric: str, value: float, trend: str) -> Literal["low", "medium", "high", "critical"]:
        """Safely determine risk level based on metric and value"""
        if metric == "latency":
            if value > config.latency_extreme:
                return "critical"
            elif value > config.latency_critical:
                return "high" if trend == "increasing" else "medium"
            elif value > config.latency_warning:
                return "medium"
            return "low"
        elif metric == "error_rate":
            if value > config.error_rate_critical:
                return "critical"
            elif value > config.error_rate_high:
                return "high" if trend == "increasing" else "medium"
            elif value > config.error_rate_warning:
                return "medium"
            return "low"
        elif metric in ["cpu_util", "memory_util"]:
            if value > 0.9:
                return "critical"
            elif value > 0.8:
                return "high" if trend == "increasing" else "medium"
            elif value > 0.7:
                return "medium"
            return "low"
        return "low"
    
    def _forecast_latency(
        self,
        history: List,
        lookahead_minutes: int
    ) -> Optional[ForecastResult]:
        """Forecast latency using linear regression"""
        try:
            latencies = [point['latency'] for point in history[-20:]]
            
            if len(latencies) < config.forecast_min_data_points:
                return None
            
            # Linear trend
            x = np.arange(len(latencies))
            slope, intercept = np.polyfit(x, latencies, 1)
            
            # Predict next value
            next_x = len(latencies)
            predicted_latency = slope * next_x + intercept
            
            # Calculate confidence
            residuals = latencies - (slope * x + intercept)
            confidence = max(0, 1 - (np.std(residuals) / max(1, np.mean(latencies))))
            
            # Determine trend and risk using helper methods
            trend = self._get_trend_literal(slope)
            risk = self._get_risk_literal("latency", predicted_latency, trend)
            
            # Calculate time to reach critical threshold
            time_to_critical = None
            if slope > 0 and predicted_latency < config.latency_extreme:
                denominator = predicted_latency - latencies[-1]
                if abs(denominator) > 0.1:
                    minutes_to_critical = lookahead_minutes * \
                        (config.latency_extreme - predicted_latency) / denominator
                    if minutes_to_critical > 0:
                        time_to_critical = minutes_to_critical
            
            return ForecastResult(
                metric="latency",
                predicted_value=predicted_latency,
                confidence=confidence,
                trend=trend,
                time_to_threshold=time_to_critical,
                risk_level=risk
            )
            
        except Exception as e:
            logger.error(f"Latency forecast error: {e}", exc_info=True)
            return None
    
    def _forecast_error_rate(
        self,
        history: List,
        lookahead_minutes: int
    ) -> Optional[ForecastResult]:
        """Forecast error rate using exponential smoothing"""
        try:
            error_rates = [point['error_rate'] for point in history[-15:]]
            
            if len(error_rates) < config.forecast_min_data_points:
                return None
            
            # Exponential smoothing
            alpha = 0.3
            forecast = error_rates[0]
            for rate in error_rates[1:]:
                forecast = alpha * rate + (1 - alpha) * forecast
            
            predicted_rate = forecast
            
            # Trend analysis
            recent_trend = np.mean(error_rates[-3:]) - np.mean(error_rates[-6:-3])
            
            # Determine trend and risk using helper methods
            if recent_trend > 0.02:
                trend: Literal["increasing", "decreasing", "stable"] = "increasing"
            elif recent_trend < -0.01:
                trend = "decreasing"
            else:
                trend = "stable"
            
            risk = self._get_risk_literal("error_rate", predicted_rate, trend)
            
            # Confidence based on volatility
            confidence = max(0, 1 - (np.std(error_rates) / max(0.01, np.mean(error_rates))))
            
            return ForecastResult(
                metric="error_rate",
                predicted_value=predicted_rate,
                confidence=confidence,
                trend=trend,
                risk_level=risk
            )
            
        except Exception as e:
            logger.error(f"Error rate forecast error: {e}", exc_info=True)
            return None
    
    def _forecast_resources(
        self,
        history: List,
        lookahead_minutes: int
    ) -> List[ForecastResult]:
        """Forecast CPU and memory utilization"""
        forecasts = []
        
        # CPU forecast
        cpu_values = [point['cpu_util'] for point in history if point.get('cpu_util') is not None]
        if len(cpu_values) >= config.forecast_min_data_points:
            try:
                predicted_cpu = np.mean(cpu_values[-5:])
                
                # Determine trend
                if len(cpu_values) >= 10:
                    cpu_trend: Literal["increasing", "decreasing", "stable"] = (
                        "increasing" if cpu_values[-1] > np.mean(cpu_values[-10:-5]) 
                        else "stable"
                    )
                else:
                    cpu_trend = "stable"
                
                risk = self._get_risk_literal("cpu_util", predicted_cpu, cpu_trend)
                
                forecasts.append(ForecastResult(
                    metric="cpu_util",
                    predicted_value=predicted_cpu,
                    confidence=0.7,
                    trend=cpu_trend,
                    risk_level=risk
                ))
            except Exception as e:
                logger.error(f"CPU forecast error: {e}", exc_info=True)
        
        # Memory forecast
        memory_values = [point['memory_util'] for point in history if point.get('memory_util') is not None]
        if len(memory_values) >= config.forecast_min_data_points:
            try:
                predicted_memory = np.mean(memory_values[-5:])
                
                # Determine trend
                if len(memory_values) >= 10:
                    memory_trend: Literal["increasing", "decreasing", "stable"] = (
                        "increasing" if memory_values[-1] > np.mean(memory_values[-10:-5]) 
                        else "stable"
                    )
                else:
                    memory_trend = "stable"
                
                risk = self._get_risk_literal("memory_util", predicted_memory, memory_trend)
                
                forecasts.append(ForecastResult(
                    metric="memory_util",
                    predicted_value=predicted_memory,
                    confidence=0.7,
                    trend=memory_trend,
                    risk_level=risk
                ))
            except Exception as e:
                logger.error(f"Memory forecast error: {e}", exc_info=True)
        
        return forecasts
    
    def get_predictive_insights(self, service: str) -> Dict[str, Any]:
        """Generate actionable insights from forecasts"""
        forecasts = self.forecast_service_health(service)
        
        critical_risks = [f for f in forecasts if f.risk_level in ["high", "critical"]]
        warnings = []
        recommendations = []
        
        for forecast in critical_risks:
            if forecast.metric == "latency" and forecast.risk_level in ["high", "critical"]:
                warnings.append(f"📈 Latency expected to reach {forecast.predicted_value:.0f}ms")
                if forecast.time_to_threshold:
                    minutes = int(forecast.time_to_threshold)
                    recommendations.append(f"⏰ Critical latency (~{config.latency_extreme}ms) in ~{minutes} minutes")
                recommendations.append("🔧 Consider scaling or optimizing dependencies")
            
            elif forecast.metric == "error_rate" and forecast.risk_level in ["high", "critical"]:
                warnings.append(f"🚨 Errors expected to reach {forecast.predicted_value*100:.1f}%")
                recommendations.append("🐛 Investigate recent deployments or dependency issues")
            
            elif forecast.metric == "cpu_util" and forecast.risk_level in ["high", "critical"]:
                warnings.append(f"🔥 CPU expected at {forecast.predicted_value*100:.1f}%")
                recommendations.append("⚡ Consider scaling compute resources")
            
            elif forecast.metric == "memory_util" and forecast.risk_level in ["high", "critical"]:
                warnings.append(f"💾 Memory expected at {forecast.predicted_value*100:.1f}%")
                recommendations.append("🧹 Check for memory leaks or optimize usage")
        
        return {
            'service': service,
            'forecasts': [
                {
                    'metric': f.metric,
                    'predicted_value': f.predicted_value,
                    'confidence': f.confidence,
                    'trend': f.trend,
                    'risk_level': f.risk_level,
                    'time_to_threshold': f.time_to_threshold
                }
                for f in forecasts
            ],
            'warnings': warnings[:3],
            'recommendations': list(dict.fromkeys(recommendations))[:3],
            'critical_risk_count': len(critical_risks),
            'forecast_timestamp': datetime.datetime.now(datetime.timezone.utc).isoformat()
        }

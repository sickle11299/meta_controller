"""
Hazard Rate Calculator for DRL Meta-Controller.

Implements hazard rate computation and numerical integration
as specified in Algorithm 1, Line 8.
"""

import logging
from typing import List, Optional
import numpy as np

from ..metrics.types import SystemMetrics

logger = logging.getLogger(__name__)


class HazardRateCalculator:
    """
    风险率计算器 (Hazard Rate Calculator)
    
    Computes hazard rate λk(t) and its integral over time windows.
    
    Implements Cox proportional hazards model:
    λk(t) = λ0 * exp(βᵀx(t))
    
    where x(t) = [CPU load, memory usage, queue length, response time]
    
    Corresponds to Algorithm 1, Line 8: "compute hazard integral H = ∫λ(t)dt"
    """
    
    def __init__(self,
                 baseline_hazard: float = 0.1,
                 coefficients: Optional[np.ndarray] = None,
                 dt: float = 1.0,
                 model: str = 'cox'):
        """
        Initialize hazard rate calculator.
        
        :param baseline_hazard: Baseline hazard rate λ0
        :param coefficients: Cox model coefficients β for [CPU, Mem, Queue, RTT]
        :param dt: Time step for numerical integration
        :param model: Model type: 'cox', 'weibull', or 'exponential'
        """
        self.baseline_hazard = baseline_hazard
        self.coefficients = coefficients if coefficients is not None else np.array([0.5, 0.3, 0.2, 0.1])
        self.dt = dt
        self.model = model
        
        assert len(self.coefficients) == 4, "Expected 4 coefficients for [CPU, Mem, Queue, RTT]"
        
        logger.info(f"HazardRateCalculator initialized: model={model}, "
                   f"baseline={baseline_hazard}, dt={dt}")
    
    def compute_hazard_rate(self, metrics: SystemMetrics) -> float:
        """
        Compute instantaneous hazard rate at current state.
        
        :param metrics: Current system metrics
        :return: Hazard rate λk(t)
        """
        if self.model == 'cox':
            return self._cox_hazard(metrics)
        elif self.model == 'weibull':
            return self._weibull_hazard(metrics)
        elif self.model == 'exponential':
            return self._exponential_hazard(metrics)
        else:
            raise ValueError(f"Unknown hazard model: {self.model}")
    
    def _cox_hazard(self, metrics: SystemMetrics) -> float:
        """
        Cox proportional hazards model.
        
        λk(t) = λ0 * exp(βᵀx(t))
        
        Covariates:
        - x[0]: CPU load (normalized)
        - x[1]: Memory usage (normalized)
        - x[2]: Queue length (normalized)
        - x[3]: Response time (normalized)
        """
        # Extract and normalize covariates
        x = np.array([
            metrics.cpu_load,                    # Already in [0,1]
            metrics.memory_usage,                # Already in [0,1]
            min(metrics.queue_length / 100.0, 1.0),     # Normalize to [0,1]
            min(metrics.avg_response_time / 1000.0, 1.0) # Normalize to [0,1]
        ])
        
        # Compute hazard rate
        hazard = self.baseline_hazard * np.exp(np.dot(self.coefficients, x))
        
        return float(hazard)
    
    def _weibull_hazard(self, metrics: SystemMetrics) -> float:
        """
        Weibull hazard model.
        
        λ(t) = (k/λ) * (t/λ)^{k-1}
        
        Uses system stress to modulate shape parameter k.
        """
        # Base time (use queue length as proxy)
        t = max(metrics.queue_length, 1)
        
        # Shape parameter (depends on system stress)
        stress = (metrics.cpu_load + metrics.memory_usage) / 2
        k = 1.5 + stress  # Shape parameter > 1 (increasing failure rate)
        
        # Scale parameter
        lambda_scale = 100.0 / (1.0 + stress)
        
        # Weibull hazard
        hazard = (k / lambda_scale) * ((t / lambda_scale) ** (k - 1))
        
        return float(hazard)
    
    def _exponential_hazard(self, metrics: SystemMetrics) -> float:
        """
        Exponential hazard model (constant failure rate).
        
        λ(t) = λ0 * exp(stress_factor)
        """
        stress = (metrics.cpu_load + metrics.memory_usage) / 2
        hazard = self.baseline_hazard * np.exp(2.0 * stress)
        
        return float(hazard)
    
    def compute_integral(self, 
                        hazard_rates: List[float],
                        timestamps: Optional[List[float]] = None,
                        method: str = 'trapezoidal') -> float:
        """
        Compute numerical integral of hazard rate over time.
        
        ∫λk(u)du ≈ Σ λ(ti) * Δt
        
        :param hazard_rates: Sequence of hazard rates [λ(t0), λ(t1), ...]
        :param timestamps: Optional timestamps for non-uniform integration
        :param method: Integration method: 'trapezoidal' or 'rectangle'
        :return: Integral value
        """
        if len(hazard_rates) < 2:
            return 0.0
        
        if method == 'trapezoidal':
            return self._trapezoidal_integrate(hazard_rates, timestamps)
        elif method == 'rectangle':
            return self._rectangle_integrate(hazard_rates, timestamps)
        else:
            raise ValueError(f"Unknown integration method: {method}")
    
    def _trapezoidal_integrate(self, 
                              hazard_rates: List[float],
                              timestamps: Optional[List[float]] = None) -> float:
        """
        Trapezoidal rule for numerical integration.
        
        ∫f(x)dx ≈ Σ (f(xi) + f(xi+1)) * Δxi / 2
        """
        integral = 0.0
        n = len(hazard_rates)
        
        if timestamps is not None:
            # Non-uniform time steps
            for i in range(n - 1):
                dt = timestamps[i+1] - timestamps[i]
                integral += (hazard_rates[i] + hazard_rates[i+1]) * dt / 2
        else:
            # Uniform time steps
            for i in range(n - 1):
                integral += (hazard_rates[i] + hazard_rates[i+1]) * self.dt / 2
        
        return float(integral)
    
    def _rectangle_integrate(self,
                            hazard_rates: List[float],
                            timestamps: Optional[List[float]] = None) -> float:
        """
        Rectangle rule (left Riemann sum) for numerical integration.
        
        ∫f(x)dx ≈ Σ f(xi) * Δxi
        """
        if timestamps is not None:
            # Non-uniform time steps
            integral = 0.0
            for i in range(len(timestamps) - 1):
                dt = timestamps[i+1] - timestamps[i]
                integral += hazard_rates[i] * dt
            return float(integral)
        else:
            # Uniform time steps
            return float(sum(hazard_rates[:-1]) * self.dt)
    
    def compute_sequence(self, 
                        history: List[SystemMetrics]) -> List[float]:
        """
        Compute hazard rate sequence for a history of metrics.
        
        :param history: List of SystemMetrics snapshots
        :return: List of hazard rates
        """
        hazard_rates = []
        
        for metrics in history:
            hazard = self.compute_hazard_rate(metrics)
            hazard_rates.append(hazard)
        
        return hazard_rates
    
    def compute_window_integral(self, 
                               history: List[SystemMetrics]) -> float:
        """
        Compute hazard integral over a history window.
        
        Combines compute_sequence and compute_integral.
        
        :param history: List of SystemMetrics snapshots
        :return: Integral of hazard rate over window
        """
        hazard_rates = self.compute_sequence(history)
        timestamps = [m.timestamp for m in history]
        
        integral = self.compute_integral(hazard_rates, timestamps)
        
        logger.debug(f"Computed hazard integral: {integral:.4f} "
                    f"over {len(history)} samples")
        
        return integral

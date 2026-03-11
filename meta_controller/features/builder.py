"""
Feature Builder for DRL Meta-Controller.

Computes gradient features from historical metrics data.
Implements Algorithm 1, Line 2: compute gradients ∇T, ∇R
"""

import logging
from typing import List, Optional
import numpy as np

from ..metrics.types import SystemMetrics, GradientFeatures

logger = logging.getLogger(__name__)


class FeatureBuilder:
    """
    特征构建器 (Feature Builder)
    
    Responsible for:
    - Computing gradient features from metric history
    - Supporting first-order and second-order differences
    - Filtering outliers and noise
    
    Corresponds to Algorithm 1, Line 2:
    - ∇T = ΔTcpu / ΔLoad (CPU load gradient)
    - ∇R = ΔRSSI / Δt (RSSI gradient)
    """
    
    def __init__(self, 
                 delta_t: float = 1.0,
                 order: int = 1,
                 use_center_diff: bool = True,
                 outlier_threshold: float = 5.0):
        """
        Initialize feature builder.
        
        :param delta_t: Time step for gradient computation (seconds)
        :param order: Order of differentiation (1 or 2)
        :param use_center_diff: Use central difference for better accuracy
        :param outlier_threshold: Threshold for outlier detection (std devs)
        """
        self.delta_t = delta_t
        self.order = order
        self.use_center_diff = use_center_diff
        self.outlier_threshold = outlier_threshold
        
        logger.debug(f"FeatureBuilder initialized: delta_t={delta_t}, order={order}")
    
    def compute_gradients(self, 
                         history: List[SystemMetrics]) -> GradientFeatures:
        """
        Compute gradient features from metric history.
        
        :param history: List of SystemMetrics snapshots (oldest to newest)
        :return: GradientFeatures with computed gradients
        """
        if len(history) < 2:
            logger.warning("Insufficient history for gradient computation")
            return GradientFeatures.zeros()
        
        if len(history) < self.order + 1:
            logger.warning(f"Need at least {self.order + 1} samples for order-{self.order} gradients")
            # Fall back to first-order
            actual_order = 1
        else:
            actual_order = self.order
        
        # Extract time series
        timestamps = np.array([m.timestamp for m in history])
        cpu_loads = np.array([m.cpu_load for m in history])
        memories = np.array([m.memory_usage for m in history])
        rssis = np.array([m.rssi for m in history])
        queues = np.array([float(m.queue_length) for m in history])
        
        # Compute time deltas
        dt = np.mean(np.diff(timestamps))
        if dt <= 0:
            dt = self.delta_t
        
        # Compute gradients using finite differences
        grad_cpu = self._compute_gradient(cpu_loads, dt, actual_order)
        grad_mem = self._compute_gradient(memories, dt, actual_order)
        grad_rssi = self._compute_gradient(rssis, dt, actual_order)
        grad_queue = self._compute_gradient(queues, dt, actual_order)
        
        # Apply outlier filtering
        grad_cpu = self._filter_outlier(grad_cpu, cpu_loads.std())
        grad_mem = self._filter_outlier(grad_mem, memories.std())
        grad_rssi = self._filter_outlier(grad_rssi, rssis.std())
        grad_queue = self._filter_outlier(grad_queue, queues.std())
        
        gradients = GradientFeatures(
            grad_cpu_load=float(grad_cpu),
            grad_memory=float(grad_mem),
            grad_rssi=float(grad_rssi),
            grad_queue=float(grad_queue)
        )
        
        logger.debug(f"Computed gradients: ∇T={grad_cpu:.4f}, ∇Mem={grad_mem:.4f}, "
                    f"∇R={grad_rssi:.4f}, ∇Queue={grad_queue:.4f}")
        
        return gradients
    
    def _compute_gradient(self, 
                         values: np.ndarray, 
                         dt: float, 
                         order: int = 1) -> float:
        """
        Compute gradient using finite differences.
        
        :param values: Time series values
        :param dt: Time step
        :param order: Order of differentiation
        :return: Gradient value (latest)
        """
        if len(values) < 2:
            return 0.0
        
        if self.use_center_diff and len(values) >= 3:
            # Central difference: f'(x) ≈ (f(x+h) - f(x-h)) / (2h)
            # More accurate than forward difference
            if order == 1:
                gradient = (values[-1] - values[-3]) / (2 * dt)
            elif order == 2:
                # Second derivative: f''(x) ≈ (f(x+h) - 2f(x) + f(x-h)) / h²
                gradient = (values[-1] - 2*values[-2] + values[-3]) / (dt ** 2)
            else:
                raise ValueError(f"Unsupported order: {order}")
        else:
            # Forward difference: f'(x) ≈ (f(x+h) - f(x)) / h
            if order == 1:
                gradient = np.diff(values)[-1] / dt
            elif order == 2:
                # Second order from first differences
                first_diff = np.diff(values)
                if len(first_diff) < 2:
                    return 0.0
                gradient = np.diff(first_diff)[-1] / dt
            else:
                raise ValueError(f"Unsupported order: {order}")
        
        return gradient
    
    def _filter_outlier(self, value: float, std: float) -> float:
        """
        Filter outliers based on standard deviation.
        
        :param value: Gradient value
        :param std: Standard deviation of the underlying metric
        :return: Filtered gradient value
        """
        if std < 1e-6:
            return value
        
        threshold = self.outlier_threshold * std / self.delta_t
        
        if abs(value) > threshold:
            # Clip to threshold
            return np.sign(value) * threshold
        
        return value
    
    @staticmethod
    def compute_moving_average(history: List[SystemMetrics], 
                               window: int = 3) -> SystemMetrics:
        """
        Compute moving average of recent metrics.
        
        :param history: List of SystemMetrics
        :param window: Window size for averaging
        :return: Averaged SystemMetrics
        """
        if len(history) == 0:
            return SystemMetrics()
        
        # Use most recent 'window' samples
        recent = history[-window:] if len(history) >= window else history
        n = len(recent)
        
        # Average all numeric fields
        avg_metrics = SystemMetrics(
            timestamp=recent[-1].timestamp,
            cpu_load=sum(m.cpu_load for m in recent) / n,
            memory_usage=sum(m.memory_usage for m in recent) / n,
            rssi=sum(m.rssi for m in recent) / n,
            task_success_rate=sum(m.task_success_rate for m in recent) / n,
            queue_length=int(sum(m.queue_length for m in recent) / n),
            avg_response_time=sum(m.avg_response_time for m in recent) / n,
            active_nodes=int(sum(m.active_nodes for m in recent) / n)
        )
        
        return avg_metrics

"""
Mock UTAA Environment for testing and validation.

Simulates the behavior of the UTAA scheduler, including:
- System metrics evolution with realistic noise and trends
- Task execution with success/failure outcomes
- Response to hyperparameter changes (w4, epsilon_risk)
"""

import logging
import numpy as np
from typing import Tuple, List, Dict, Any, Optional
from dataclasses import dataclass
from ..metrics.types import SystemMetrics

logger = logging.getLogger(__name__)


@dataclass
class TaskRecord:
    """Record of a scheduled task."""
    timestamp: float
    success: bool
    response_time: float
    queue_length: int


class MockUTAAEnvironment:
    """
    模拟 UTAA 调度器环境 (Mock UTAA Scheduler Environment)
    
    This class simulates the behavior of the UTAA scheduler for offline
    testing and validation of the DRL meta-controller.
    
    Features:
    - Realistic metric dynamics with diurnal patterns
    - Task execution with parameter-dependent success rates
    - Configurable noise levels and system characteristics
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize mock UTAA environment.
        
        :param config: Configuration dictionary with optional keys:
            - noise_level: Standard deviation of measurement noise (default: 0.05)
            - base_success_rate: Baseline task success probability (default: 0.85)
            - w4_sensitivity: How much w4 affects success (default: 0.1)
            - risk_sensitivity: How much epsilon_risk affects success (default: 0.2)
            - diurnal_amplitude: Amplitude of daily patterns (default: 0.3)
        """
        self.config = config or {}
        
        # Hyperparameters (controlled by meta-controller)
        self.w4 = 0.5  # Default weight parameter
        self.epsilon_risk = 0.1  # Default risk threshold
        
        # Simulation state
        self.current_time = 0.0
        self.task_history: List[TaskRecord] = []
        
        # Initial metrics
        self._cpu_load = 0.5
        self._memory_usage = 0.6
        self._rssi = -70.0
        self._queue_length = 10
        self._response_time = 50.0
        self._active_nodes = 45
        
        # Configuration parameters
        self.noise_level = self.config.get('noise_level', 0.05)
        self.base_success_rate = self.config.get('base_success_rate', 0.85)
        self.w4_sensitivity = self.config.get('w4_sensitivity', 0.1)
        self.risk_sensitivity = self.config.get('risk_sensitivity', 0.2)
        self.diurnal_amplitude = self.config.get('diurnal_amplitude', 0.3)
        
        logger.info("MockUTAAEnvironment initialized",
                   extra={'w4': self.w4, 'epsilon_risk': self.epsilon_risk})
    
    def update_params(self, w4: float, epsilon_risk: float) -> None:
        """
        Update scheduler hyperparameters from meta-controller.
        
        :param w4: Weight parameter (should be positive)
        :param epsilon_risk: Risk threshold (should be in [0, 1])
        """
        # Validate and clip parameters
        self.w4 = max(0.01, w4)  # Ensure w4 > 0
        self.epsilon_risk = np.clip(epsilon_risk, 0.0, 1.0)
        
        logger.debug("UTAA parameters updated",
                    extra={'w4': self.w4, 'epsilon_risk': self.epsilon_risk})
    
    def step(self, delta_t: float = 1.0) -> SystemMetrics:
        """
        Advance simulation by delta_t seconds and return new metrics.
        
        :param delta_t: Time step in seconds
        :return: SystemMetrics snapshot
        """
        self.current_time += delta_t
        
        # Update metrics with diurnal patterns and noise
        self._update_metrics(delta_t)
        
        # Simulate task execution
        self._simulate_tasks()
        
        # Create metrics snapshot
        metrics = SystemMetrics(
            timestamp=self.current_time,
            cpu_load=self._cpu_load,
            memory_usage=self._memory_usage,
            rssi=self._rssi,
            task_success_rate=self._compute_recent_success_rate(),
            queue_length=self._queue_length,
            avg_response_time=self._response_time,
            active_nodes=self._active_nodes
        )
        
        return metrics
    
    def _update_metrics(self, delta_t: float) -> None:
        """Update internal metric states with realistic dynamics."""
        # Diurnal pattern (period = 3600 seconds = 1 hour for faster simulation)
        phase = self.current_time / 3600.0 * 2 * np.pi
        
        # CPU load: diurnal pattern + noise + dependency on w4
        cpu_trend = 0.5 + self.diurnal_amplitude * np.sin(phase)
        cpu_noise = np.random.randn() * self.noise_level
        # Higher w4 -> slightly better load balancing -> lower peak load
        w4_effect = -(self.w4 - 0.5) * 0.1
        self._cpu_load = np.clip(cpu_trend + cpu_noise + w4_effect, 0.0, 1.0)
        
        # Memory usage: slower dynamics + diurnal pattern
        mem_trend = 0.6 + self.diurnal_amplitude * 0.5 * np.sin(phase - np.pi/4)
        mem_noise = np.random.randn() * self.noise_level * 0.5
        self._memory_usage = np.clip(mem_trend + mem_noise, 0.0, 1.0)
        
        # RSSI: random walk with mean reversion
        rssi_mean = -70.0
        rssi_noise = np.random.randn() * 3.0
        self._rssi += 0.1 * (rssi_mean - self._rssi) + rssi_noise
        self._rssi = np.clip(self._rssi, -100.0, -30.0)
        
        # Queue length: depends on load and w4
        queue_base = self._cpu_load * 50
        queue_noise = np.random.randn() * 5
        # Higher w4 -> better scheduling -> shorter queues
        w4_queue_effect = -(self.w4 - 0.5) * 10
        self._queue_length = max(0, int(queue_base + queue_noise + w4_queue_effect))
        
        # Response time: depends on queue and memory pressure
        rt_base = 50.0 + self._queue_length * 2.0 + self._memory_usage * 20.0
        rt_noise = np.random.randn() * 5.0
        self._response_time = max(10.0, rt_base + rt_noise)
        
        # Active nodes: varies with load
        node_target = 40 + self._cpu_load * 20
        node_change = np.random.randint(-2, 3)
        self._active_nodes = int(np.clip(node_target + node_change, 30, 60))
    
    def _simulate_tasks(self) -> None:
        """Simulate task execution during current time step."""
        # Number of tasks depends on queue length
        n_tasks = max(1, self._queue_length // 2 + np.random.randint(1, 5))
        
        for _ in range(n_tasks):
            # Success probability depends on hyperparameters
            # Higher w4 -> better resource allocation -> higher success
            # Lower epsilon_risk -> more conservative -> higher success but slower
            base_prob = self.base_success_rate
            w4_adjustment = (self.w4 - 0.5) * self.w4_sensitivity
            risk_adjustment = -(self.epsilon_risk - 0.1) * self.risk_sensitivity
            
            success_prob = np.clip(base_prob + w4_adjustment + risk_adjustment, 0.5, 0.95)
            success = np.random.rand() < success_prob
            
            # Response time for this task
            response_time = self._response_time * (0.8 + np.random.rand() * 0.4)
            if not success:
                response_time *= 2.0  # Failed tasks take longer
            
            self.task_history.append(TaskRecord(
                timestamp=self.current_time,
                success=success,
                response_time=response_time,
                queue_length=self._queue_length
            ))
        
        # Prune old task history (keep last 5 minutes)
        cutoff = self.current_time - 300.0
        self.task_history = [t for t in self.task_history if t.timestamp > cutoff]
    
    def _compute_recent_success_rate(self, window: float = 60.0) -> float:
        """
        Compute task success rate over recent window.
        
        :param window: Time window in seconds
        :return: Success rate in [0, 1]
        """
        cutoff = self.current_time - window
        recent_tasks = [t for t in self.task_history if t.timestamp > cutoff]
        
        if len(recent_tasks) == 0:
            return 1.0  # No tasks -> assume perfect
        
        n_success = sum(1 for t in recent_tasks if t.success)
        return n_success / len(recent_tasks)
    
    def get_task_stats(self) -> Tuple[int, int]:
        """
        Get cumulative task statistics.
        
        :return: (n_success, n_total) tuple
        """
        n_total = len(self.task_history)
        n_success = sum(1 for t in self.task_history if t.success)
        return n_success, n_total
    
    def get_current_metrics(self) -> SystemMetrics:
        """Get current metrics without advancing simulation."""
        return SystemMetrics(
            timestamp=self.current_time,
            cpu_load=self._cpu_load,
            memory_usage=self._memory_usage,
            rssi=self._rssi,
            task_success_rate=self._compute_recent_success_rate(),
            queue_length=self._queue_length,
            avg_response_time=self._response_time,
            active_nodes=self._active_nodes
        )
    
    def reset(self, seed: Optional[int] = None) -> None:
        """
        Reset environment to initial state.
        
        :param seed: Random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)
        
        self.current_time = 0.0
        self.task_history = []
        self._cpu_load = 0.5
        self._memory_usage = 0.6
        self._rssi = -70.0
        self._queue_length = 10
        self._response_time = 50.0
        self._active_nodes = 45
        self.w4 = 0.5
        self.epsilon_risk = 0.1
        
        logger.info("MockUTAAEnvironment reset")

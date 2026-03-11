"""
Data types for DRL Meta-Controller.

Defines the core data structures used throughout the meta-controller module:
- SystemMetrics: Raw system metrics snapshot
- GradientFeatures: Computed gradient features
- MetaState: Complete RL state representation
"""

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np


@dataclass
class SystemMetrics:
    """
    系统指标快照 (System Metrics Snapshot)
    
    Contains all raw metrics collected from the UTAA scheduler at a given timestamp.
    These metrics are used to construct the RL state vector.
    """
    timestamp: float = 0.0
    
    # Core metrics (7 dimensions as per paper)
    cpu_load: float = 0.0           # CPU 负载 (0-1)
    memory_usage: float = 0.0       # 内存使用率 (0-1)
    rssi: float = -70.0            # 信号强度 (dBm), typically -100 to -30
    task_success_rate: float = 1.0  # 任务成功率 (0-1)
    queue_length: int = 0          # 队列长度
    avg_response_time: float = 0.0  # 平均响应时间 (ms)
    active_nodes: int = 0          # 活跃节点数
    
    def to_vector(self, normalize: bool = True) -> np.ndarray:
        """
        Convert metrics to normalized vector.
        
        :param normalize: Whether to normalize values to [0,1] range
        :return: numpy array of shape (7,)
        """
        if normalize:
            return np.array([
                self.cpu_load,                    # Already in [0,1]
                self.memory_usage,                # Already in [0,1]
                (self.rssi + 100) / 70.0,        # Map [-100, -30] to [0,1]
                self.task_success_rate,           # Already in [0,1]
                min(self.queue_length / 100.0, 1.0),  # Cap at 100
                min(self.avg_response_time / 1000.0, 1.0),  # Cap at 1000ms
                min(self.active_nodes / 100.0, 1.0)  # Cap at 100 nodes
            ])
        else:
            return np.array([
                self.cpu_load,
                self.memory_usage,
                self.rssi,
                self.task_success_rate,
                float(self.queue_length),
                self.avg_response_time,
                float(self.active_nodes)
            ])
    
    @classmethod
    def from_vector(cls, vector: np.ndarray, normalize: bool = True) -> 'SystemMetrics':
        """
        Create SystemMetrics from vector.
        
        :param vector: numpy array of shape (7,)
        :param normalize: Whether vector is normalized
        :return: SystemMetrics instance
        """
        if normalize:
            # Denormalize
            rssi = vector[2] * 70.0 - 100
            queue_length = int(vector[3] * 100.0)
            response_time = vector[4] * 1000.0
            active_nodes = int(vector[5] * 100.0)
            
            return cls(
                cpu_load=float(np.clip(vector[0], 0, 1)),
                memory_usage=float(np.clip(vector[1], 0, 1)),
                rssi=float(np.clip(rssi, -100, -30)),
                task_success_rate=float(np.clip(vector[3], 0, 1)),
                queue_length=max(0, queue_length),
                avg_response_time=max(0, response_time),
                active_nodes=max(0, active_nodes)
            )
        else:
            return cls(
                cpu_load=float(vector[0]),
                memory_usage=float(vector[1]),
                rssi=float(vector[2]),
                task_success_rate=float(vector[3]),
                queue_length=int(vector[4]),
                avg_response_time=float(vector[5]),
                active_nodes=int(vector[6])
            )


@dataclass
class GradientFeatures:
    """
    梯度特征 (Gradient Features)
    
    Contains computed gradients representing rate of change in key metrics.
    Corresponds to Algorithm 1, Line 2: ∇T = ΔTcpu/ΔLoad, ∇R = ΔRSSI/Δt
    """
    grad_cpu_load: float = 0.0      # ∇T = ΔCPU/Δt
    grad_memory: float = 0.0        # ∇Mem = ΔMem/Δt
    grad_rssi: float = 0.0          # ∇R = ΔRSSI/Δt
    grad_queue: float = 0.0         # ∇Queue = ΔQueue/Δt
    
    def to_vector(self, normalize: bool = True) -> np.ndarray:
        """
        Convert gradients to normalized vector.
        
        :param normalize: Whether to clip values to [-5, 5] range
        :return: numpy array of shape (4,)
        """
        if normalize:
            # Clip extreme gradients for stability
            return np.array([
                np.clip(self.grad_cpu_load, -5.0, 5.0),
                np.clip(self.grad_memory, -5.0, 5.0),
                np.clip(self.grad_rssi / 10.0, -5.0, 5.0),  # RSSI gradient scaled
                np.clip(self.grad_queue / 10.0, -5.0, 5.0)   # Queue gradient scaled
            ])
        else:
            return np.array([
                self.grad_cpu_load,
                self.grad_memory,
                self.grad_rssi,
                self.grad_queue
            ])
    
    @classmethod
    def zeros(cls) -> 'GradientFeatures':
        """Create zero-initialized gradient features."""
        return cls()


@dataclass
class MetaState:
    """
    元控制器状态 (Meta-Controller State)
    
    Complete state representation for the DRL meta-controller.
    Combines current metrics with gradient features.
    
    State vector dimension: 7 (metrics) + 4 (gradients) = 11
    """
    current_metrics: SystemMetrics
    gradients: GradientFeatures
    history_window: List[SystemMetrics] = field(default_factory=list)
    
    def to_vector(self, normalize: bool = True) -> np.ndarray:
        """
        Encode state as neural network input vector.
        
        :param normalize: Whether to normalize/clip values
        :return: numpy array of shape (11,)
        """
        metrics_vec = self.current_metrics.to_vector(normalize=normalize)
        gradients_vec = self.gradients.to_vector(normalize=normalize)
        
        return np.concatenate([metrics_vec, gradients_vec])
    
    @property
    def state_dim(self) -> int:
        """Return state vector dimension."""
        return 11  # 7 metrics + 4 gradients
    
    @classmethod
    def zeros(cls) -> 'MetaState':
        """Create zero-initialized state."""
        return cls(
            current_metrics=SystemMetrics(),
            gradients=GradientFeatures.zeros(),
            history_window=[]
        )
    
    def copy(self) -> 'MetaState':
        """Create a deep copy of the state."""
        return MetaState(
            current_metrics=SystemMetrics(
                timestamp=self.current_metrics.timestamp,
                cpu_load=self.current_metrics.cpu_load,
                memory_usage=self.current_metrics.memory_usage,
                rssi=self.current_metrics.rssi,
                task_success_rate=self.current_metrics.task_success_rate,
                queue_length=self.current_metrics.queue_length,
                avg_response_time=self.current_metrics.avg_response_time,
                active_nodes=self.current_metrics.active_nodes
            ),
            gradients=GradientFeatures(
                grad_cpu_load=self.gradients.grad_cpu_load,
                grad_memory=self.gradients.grad_memory,
                grad_rssi=self.gradients.grad_rssi,
                grad_queue=self.gradients.grad_queue
            ),
            history_window=[
                SystemMetrics(
                    timestamp=m.timestamp,
                    cpu_load=m.cpu_load,
                    memory_usage=m.memory_usage,
                    rssi=m.rssi,
                    task_success_rate=m.task_success_rate,
                    queue_length=m.queue_length,
                    avg_response_time=m.avg_response_time,
                    active_nodes=m.active_nodes
                )
                for m in self.history_window
            ]
        )

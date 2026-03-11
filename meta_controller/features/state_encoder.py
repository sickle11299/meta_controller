"""
State Encoder for DRL Meta-Controller.

Encodes the meta-controller state into a fixed-dimensional vector
suitable for neural network input.
"""

import logging
from typing import Optional, Tuple
import numpy as np

from ..metrics.types import SystemMetrics, GradientFeatures, MetaState

logger = logging.getLogger(__name__)


class MetaStateEncoder:
    """
    状态编码器 (Meta-State Encoder)
    
    Responsible for:
    - Encoding MetaState into fixed-dimensional numpy array
    - Normalizing features using Z-score or Min-Max normalization
    - Clipping extreme values for numerical stability
    
    State vector structure:
    [0:7]   - Current metrics (CPU, Mem, RSSI, Success, Queue, RTT, Nodes)
    [7:11]  - Gradient features (∇CPU, ∇Mem, ∇RSSI, ∇Queue)
    
    Total dimension: 11
    """
    
    def __init__(self, 
                 normalize: bool = True,
                 method: str = 'zscore',
                 clip_range: Tuple[float, float] = (-5.0, 5.0),
                 running_mean: Optional[np.ndarray] = None,
                 running_std: Optional[np.ndarray] = None):
        """
        Initialize state encoder.
        
        :param normalize: Whether to normalize features
        :param method: Normalization method: 'zscore' or 'minmax'
        :param clip_range: Range for clipping normalized values
        :param running_mean: Running mean for z-score normalization
        :param running_std: Running std for z-score normalization
        """
        self.normalize = normalize
        self.method = method
        self.clip_range = clip_range
        
        # Running statistics for online normalization
        self.running_mean = running_mean if running_mean is not None else np.zeros(11)
        self.running_std = running_std if running_std is not None else np.ones(11)
        self.n_samples = 0
        
        logger.info(f"MetaStateEncoder initialized: normalize={normalize}, method={method}")
    
    def encode(self, state: MetaState) -> np.ndarray:
        """
        Encode MetaState into normalized vector.
        
        :param state: MetaState object
        :return: Numpy array of shape (11,)
        """
        # Get raw state vector
        state_vec = state.to_vector(normalize=False)
        
        if self.normalize:
            # Update running statistics
            self._update_statistics(state_vec)
            
            # Normalize
            if self.method == 'zscore':
                normalized = self._zscore_normalize(state_vec)
            elif self.method == 'minmax':
                normalized = self._minmax_normalize(state_vec)
            else:
                raise ValueError(f"Unknown normalization method: {self.method}")
            
            # Clip extreme values
            normalized = np.clip(normalized, self.clip_range[0], self.clip_range[1])
            
            logger.debug(f"Encoded state: mean={normalized.mean():.3f}, "
                        f"std={normalized.std():.3f}")
            
            return normalized
        else:
            return state_vec
    
    def _update_statistics(self, state_vec: np.ndarray) -> None:
        """Update running mean and std for online normalization."""
        self.n_samples += 1
        
        # Welford's online algorithm for computing mean and variance
        delta = state_vec - self.running_mean
        self.running_mean += delta / self.n_samples
        
        if self.n_samples > 1:
            delta2 = state_vec - self.running_mean
            self.running_std = np.sqrt(
                (self.running_std ** 2 * (self.n_samples - 2) + delta * delta2) 
                / (self.n_samples - 1)
            )
        
        # Prevent division by zero
        self.running_std = np.maximum(self.running_std, 1e-6)
    
    def _zscore_normalize(self, state_vec: np.ndarray) -> np.ndarray:
        """
        Z-score normalization: (x - μ) / σ
        
        :param state_vec: Raw state vector
        :return: Normalized vector
        """
        return (state_vec - self.running_mean) / self.running_std
    
    def _minmax_normalize(self, state_vec: np.ndarray) -> np.ndarray:
        """
        Min-Max normalization: (x - min) / (max - min)
        
        Uses running statistics to estimate min/max as μ ± 3σ
        
        :param state_vec: Raw state vector
        :return: Normalized vector in approximately [0, 1]
        """
        min_val = self.running_mean - 3 * self.running_std
        max_val = self.running_mean + 3 * self.running_std
        
        range_val = max_val - min_val
        range_val = np.maximum(range_val, 1e-6)  # Prevent division by zero
        
        normalized = (state_vec - min_val) / range_val
        
        # Scale to [-1, 1] for symmetry
        return 2.0 * normalized - 1.0
    
    def decode(self, normalized_vec: np.ndarray) -> MetaState:
        """
        Decode normalized vector back to MetaState (approximate).
        
        Note: This is an approximate inverse operation, primarily for debugging.
        
        :param normalized_vec: Normalized state vector
        :return: Approximate MetaState
        """
        if not self.normalize:
            return MetaState.from_vector(normalized_vec)
        
        # Inverse normalization
        if self.method == 'zscore':
            raw_vec = normalized_vec * self.running_std + self.running_mean
        elif self.method == 'minmax':
            # Inverse of 2*((x-min)/(max-min)) - 1
            scaled = (normalized_vec + 1.0) / 2.0
            min_val = self.running_mean - 3 * self.running_std
            max_val = self.running_mean + 3 * self.running_std
            raw_vec = scaled * (max_val - min_val) + min_val
        else:
            raw_vec = normalized_vec
        
        # Reconstruct metrics and gradients
        metrics_vec = raw_vec[:7]
        gradients_vec = raw_vec[7:]
        
        current_metrics = SystemMetrics.from_vector(metrics_vec, normalize=False)
        gradients = GradientFeatures.from_vector(gadients_vec, normalize=False)
        
        return MetaState(
            current_metrics=current_metrics,
            gradients=gradients,
            history_window=[]
        )
    
    def get_statistics(self) -> dict:
        """
        Get current normalization statistics.
        
        :return: Dictionary with mean, std, n_samples
        """
        return {
            'mean': self.running_mean.copy(),
            'std': self.running_std.copy(),
            'n_samples': self.n_samples
        }
    
    def reset_statistics(self) -> None:
        """Reset running statistics to initial values."""
        self.running_mean = np.zeros(11)
        self.running_std = np.ones(11)
        self.n_samples = 0
        logger.info("State encoder statistics reset")

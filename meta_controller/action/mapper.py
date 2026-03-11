"""
Action Mapper for DRL Meta-Controller.

Maps raw action vectors from the policy network to scheduler hyperparameters
using the transformations specified in Algorithm 1, Line 5.
"""

import logging
import numpy as np
from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class MetaAction:
    """
    元控制器动作 (Meta-Controller Action)
    
    Raw continuous action vector sampled from the policy network.
    Dimension: 5 (as per paper specification)
    
    Components:
    - a[0]: Used for εrisk mapping
    - a[1:4]: Reserved for future extensions
    - a[4]: Used for w4 mapping
    """
    raw_values: np.ndarray
    
    def __post_init__(self):
        """Validate action dimension."""
        if len(self.raw_values) != 5:
            raise ValueError(f"Expected 5D action vector, got {len(self.raw_values)}")
    
    @property
    def epsilon_component(self) -> float:
        """Get component used for epsilon_risk mapping."""
        return float(self.raw_values[0])
    
    @property
    def w4_component(self) -> float:
        """Get component used for w4 mapping."""
        return float(self.raw_values[4])
    
    @classmethod
    def zeros(cls) -> 'MetaAction':
        """Create zero action."""
        return cls(raw_values=np.zeros(5))
    
    @classmethod
    def from_torch(cls, tensor: torch.Tensor) -> 'MetaAction':
        """Create from PyTorch tensor."""
        if isinstance(tensor, torch.Tensor):
            array = tensor.detach().cpu().numpy()
        else:
            array = tensor
        
        # Ensure 1D array
        if array.ndim > 1:
            array = array.flatten()
        
        return cls(raw_values=array)


@dataclass
class SchedulerParams:
    """
    调度器超参数 (Scheduler Hyperparameters)
    
    Mapped hyperparameters to be sent to UTAA scheduler.
    
    Fields:
    - w4: Weight parameter (positive real number)
    - epsilon_risk: Risk threshold (in [0, 1])
    - additional_params: Dictionary for future extensions
    """
    w4: float = 0.5
    epsilon_risk: float = 0.1
    additional_params: dict = None
    
    def __post_init__(self):
        """Initialize additional_params if None."""
        if self.additional_params is None:
            self.additional_params = {}
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'w4': self.w4,
            'epsilon_risk': self.epsilon_risk,
            **self.additional_params
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'SchedulerParams':
        """Create from dictionary."""
        return cls(
            w4=data.get('w4', 0.5),
            epsilon_risk=data.get('epsilon_risk', 0.1),
            additional_params={k: v for k, v in data.items() 
                             if k not in ['w4', 'epsilon_risk']}
        )


class ActionMapper:
    """
    动作映射器 (Action Mapper)
    
    Maps raw policy actions to scheduler hyperparameters using:
    - w4 <- Softplus(a[4] * β)  (ensures w4 > 0)
    - εrisk <- Clip(εbase + a[0], 0, 1)  (ensures εrisk ∈ [0,1])
    
    Corresponds to Algorithm 1, Line 5.
    """
    
    def __init__(self, 
                 epsilon_base: float = 0.1,
                 beta: float = 1.0,
                 w4_min: float = 0.01,
                 w4_max: float = 10.0):
        """
        Initialize action mapper.
        
        :param epsilon_base: Base risk threshold εbase
        :param beta: Scaling factor β for Softplus transformation
        :param w4_min: Minimum allowed w4 value
        :param w4_max: Maximum allowed w4 value
        """
        self.epsilon_base = epsilon_base
        self.beta = beta
        self.w4_min = w4_min
        self.w4_max = w4_max
        
        logger.info(f"ActionMapper initialized: epsilon_base={epsilon_base}, beta={beta}")
    
    def map_to_params(self, action: MetaAction) -> SchedulerParams:
        """
        Map raw action to scheduler parameters.
        
        Implements Algorithm 1, Line 5:
        - w4 ← Softplus(a₄ * β)
        - εrisk ← Clip(εbase + a₀, 0, 1)
        
        :param action: Raw action from policy
        :return: Mapped scheduler parameters
        """
        # Map w4 using Softplus (ensures positivity)
        # Softplus(x) = log(1 + exp(x))
        w4_raw = F.softplus(torch.tensor(action.w4_component * self.beta)).item()
        w4 = np.clip(w4_raw, self.w4_min, self.w4_max)
        
        # Map epsilon_risk using clipping
        epsilon_risk = self.epsilon_base + action.epsilon_component
        epsilon_risk = np.clip(epsilon_risk, 0.0, 1.0)
        
        params = SchedulerParams(
            w4=float(w4),
            epsilon_risk=float(epsilon_risk)
        )
        
        logger.debug(f"Mapped action to params: w4={w4:.4f}, "
                    f"epsilon_risk={epsilon_risk:.4f}",
                    extra={'w4': w4, 'epsilon_risk': epsilon_risk})
        
        return params
    
    def map_to_params_batch(self, 
                           actions: np.ndarray) -> list:
        """
        Map batch of actions to scheduler parameters.
        
        :param actions: Array of shape (batch_size, 5)
        :return: List of SchedulerParams
        """
        if actions.ndim != 2 or actions.shape[1] != 5:
            raise ValueError(f"Expected actions of shape (batch_size, 5), got {actions.shape}")
        
        params_list = []
        for i in range(actions.shape[0]):
            action = MetaAction(raw_values=actions[i])
            params = self.map_to_params(action)
            params_list.append(params)
        
        return params_list
    
    def inverse_map_w4(self, w4: float) -> float:
        """
        Inverse Softplus mapping for w4 (for debugging/analysis).
        
        :param w4: Mapped w4 value
        :return: Approximate raw action component
        """
        # Inverse of Softplus: x = log(exp(y) - 1)
        # But we need to account for beta scaling
        w4_clipped = np.clip(w4, self.w4_min, self.w4_max)
        raw = np.log(np.exp(w4_clipped) - 1 + 1e-6) / self.beta
        return float(raw)
    
    def inverse_map_epsilon(self, epsilon_risk: float) -> float:
        """
        Inverse clipping mapping for epsilon_risk (for debugging/analysis).
        
        :param epsilon_risk: Mapped epsilon_risk value
        :return: Approximate raw action component
        """
        # Simple inverse: a[0] = epsilon_risk - epsilon_base
        return float(epsilon_risk - self.epsilon_base)

"""
Reward Calculator for DRL Meta-Controller.

Computes multi-objective reward as specified in Algorithm 1, Line 9:
rt = ψ * (Nsucc / Ntot) - ξ * ∫λk(u)du - φ * ||at - at-1||²
"""

import logging
from typing import Optional
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RewardComponents:
    """
    奖励分解 (Reward Components)
    
    Breakdown of reward into interpretable components for analysis.
    
    Fields:
    - success_bonus: Reward from task success rate
    - hazard_penalty: Penalty from hazard integral
    - smoothness_penalty: Penalty from action changes
    - total_reward: Sum of all components
    """
    success_bonus: float = 0.0
    hazard_penalty: float = 0.0
    smoothness_penalty: float = 0.0
    total_reward: float = 0.0
    
    @staticmethod
    def from_components(success_bonus: float,
                       hazard_penalty: float,
                       smoothness_penalty: float) -> 'RewardComponents':
        """
        Create RewardComponents from individual components.
        
        :param success_bonus: Success rate bonus
        :param hazard_penalty: Hazard integral penalty
        :param smoothness_penalty: Action smoothness penalty
        :return: RewardComponents instance
        """
        total = success_bonus - hazard_penalty - smoothness_penalty
        return RewardComponents(
            success_bonus=success_bonus,
            hazard_penalty=hazard_penalty,
            smoothness_penalty=smoothness_penalty,
            total_reward=total
        )
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'success_bonus': self.success_bonus,
            'hazard_penalty': self.hazard_penalty,
            'smoothness_penalty': self.smoothness_penalty,
            'total_reward': self.total_reward
        }
    
    def __str__(self) -> str:
        """String representation."""
        return (f"Reward(total={self.total_reward:.4f}, "
                f"success={self.success_bonus:.4f}, "
                f"hazard={self.hazard_penalty:.4f}, "
                f"smooth={self.smoothness_penalty:.4f})")


class RewardCalculator:
    """
    奖励计算器 (Reward Calculator)
    
    Computes the multi-objective reward function from Algorithm 1, Line 9:
    
    rt = ψ * (Nsucc / Ntot) - ξ * ∫λk(u)du - φ * ||at - at-1||²
    
    where:
    - ψ (psi): Weight for success rate bonus
    - ξ (xi): Weight for hazard penalty
    - φ (phi): Weight for smoothness penalty
    """
    
    def __init__(self,
                 psi: float = 1.0,
                 xi: float = 0.5,
                 phi: float = 0.1):
        """
        Initialize reward calculator.
        
        :param psi: Weight for task success rate bonus
        :param xi: Weight for hazard integral penalty
        :param phi: Weight for action smoothness penalty
        """
        self.psi = psi
        self.xi = xi
        self.phi = phi
        
        logger.info(f"RewardCalculator initialized: psi={psi}, xi={xi}, phi={phi}")
    
    def calculate(self,
                 n_success: int,
                 n_total: int,
                 hazard_integral: float,
                 current_action: np.ndarray,
                 previous_action: Optional[np.ndarray] = None) -> RewardComponents:
        """
        Compute reward with full breakdown.
        
        Implements Algorithm 1, Line 9:
        rt = ψ*(Nsucc/Ntot) - ξ*∫λk(u)du - φ*||at - at-1||²
        
        :param n_success: Number of successful tasks
        :param n_total: Total number of tasks
        :param hazard_integral: Integral of hazard rate over window
        :param current_action: Current action vector at
        :param previous_action: Previous action vector at-1 (optional)
        :return: RewardComponents with breakdown
        """
        # 1. Success rate bonus: ψ * (Nsucc / Ntot)
        if n_total > 0:
            success_rate = n_success / n_total
        else:
            success_rate = 1.0  # No tasks -> assume perfect
        
        success_bonus = self.psi * success_rate
        
        # 2. Hazard penalty: ξ * ∫λk(u)du
        hazard_penalty = self.xi * hazard_integral
        
        # 3. Action smoothness penalty: φ * ||at - at-1||²
        if previous_action is not None:
            action_change = np.linalg.norm(current_action - previous_action) ** 2
            smoothness_penalty = self.phi * action_change
        else:
            smoothness_penalty = 0.0  # No penalty on first step
        
        # 4. Total reward
        reward = RewardComponents.from_components(
            success_bonus=success_bonus,
            hazard_penalty=hazard_penalty,
            smoothness_penalty=smoothness_penalty
        )
        
        logger.debug(f"Computed reward: {reward}",
                    extra={'n_success': n_success, 'n_total': n_total,
                          'hazard_integral': hazard_integral})
        
        return reward
    
    def compute_success_bonus(self, 
                             n_success: int, 
                             n_total: int) -> float:
        """
        Compute success rate bonus component.
        
        :param n_success: Number of successful tasks
        :param n_total: Total number of tasks
        :return: Success bonus value
        """
        if n_total > 0:
            success_rate = n_success / n_total
        else:
            success_rate = 1.0
        
        return self.psi * success_rate
    
    def compute_hazard_penalty(self, 
                              hazard_integral: float) -> float:
        """
        Compute hazard penalty component.
        
        :param hazard_integral: Integral of hazard rate
        :return: Hazard penalty value
        """
        return self.xi * hazard_integral
    
    def compute_smoothness_penalty(self,
                                  current_action: np.ndarray,
                                  previous_action: np.ndarray) -> float:
        """
        Compute action smoothness penalty component.
        
        :param current_action: Current action vector
        :param previous_action: Previous action vector
        :return: Smoothness penalty value
        """
        action_change = np.linalg.norm(current_action - previous_action) ** 2
        return self.phi * action_change
    
    def update_weights(self,
                      psi: Optional[float] = None,
                      xi: Optional[float] = None,
                      phi: Optional[float] = None) -> None:
        """
        Update reward weights.
        
        :param psi: New psi weight (if provided)
        :param xi: New xi weight (if provided)
        :param phi: New phi weight (if provided)
        """
        if psi is not None:
            self.psi = psi
        if xi is not None:
            self.xi = xi
        if phi is not None:
            self.phi = phi
        
        logger.info(f"Reward weights updated: psi={self.psi}, xi={self.xi}, phi={self.phi}")

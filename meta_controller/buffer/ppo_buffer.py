"""
PPO Replay Buffer for DRL Meta-Controller.

Stores transitions with log probabilities required for PPO updates.
Supports batch sampling and GAE advantage computation.
"""

import logging
from typing import List, Tuple, Optional, NamedTuple
from dataclasses import dataclass
import numpy as np
import torch

from ..metrics.types import MetaState
from ..policy.actor_critic import PPOConfig
from ..action.mapper import MetaAction

logger = logging.getLogger(__name__)


@dataclass
class PPOTTransition:
    """
    PPO 转移样本 (PPO Transition)
    
    Extended transition tuple including log probability for PPO.
    
    Fields:
    - state: Pre-action state
    - action: Raw action vector
    - reward: Immediate reward
    - next_state: Post-action state
    - done: Whether episode terminated
    - log_prob: Log probability under behavior policy
    """
    state: MetaState
    action: MetaAction
    reward: float
    next_state: MetaState
    done: bool
    log_prob: float


class PPOBatch(NamedTuple):
    """
    Batch of transitions for training.
    
    Contains tensors ready for PPO update.
    """
    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_states: torch.Tensor
    dones: torch.Tensor
    old_log_probs: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor


class PPOBuffer:
    """
    PPO 回放缓冲区 (PPO Replay Buffer)
    
    Stores transitions collected during environment interaction.
    Unlike DQN replay buffers, PPO uses on-policy learning, so:
    - Buffer is cleared after each update phase
    - All samples are used in each epoch (no random sampling)
    - Transitions must include log probabilities
    
    Supports:
    - Storing transitions with log probs
    - Computing GAE advantages
    - Converting to PyTorch tensors
    """
    
    def __init__(self, 
                 capacity: int = 4096,
                 gamma: float = 0.99,
                 lam: float = 0.95):
        """
        Initialize PPO buffer.
        
        :param capacity: Maximum buffer size
        :param gamma: Discount factor for returns
        :param lam: GAE lambda parameter
        """
        self.capacity = capacity
        self.gamma = gamma
        self.lam = lam
        
        self.buffer: List[PPOTTransition] = []
        self.position = 0
        self.is_full = False
        
        logger.info(f"PPOBuffer initialized: capacity={capacity}, "
                   f"gamma={gamma}, lam={lam}")
    
    def store(self, transition: PPOTTransition) -> None:
        """
        Store a single transition.
        
        :param transition: Transition to store
        """
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            # Overwrite oldest (should not happen in standard PPO)
            self.buffer[self.position] = transition
            self.is_full = True
        
        self.position = (self.position + 1) % self.capacity
        
        if len(self.buffer) % 100 == 0:
            logger.debug(f"Buffer size: {len(self.buffer)}")
    
    def store_batch(self, transitions: List[PPOTTransition]) -> None:
        """
        Store multiple transitions.
        
        :param transitions: List of transitions
        """
        for transition in transitions:
            self.store(transition)
    
    def get(self) -> List[PPOTTransition]:
        """
        Get all stored transitions.
        
        :return: List of transitions
        """
        return self.buffer
    
    def clear(self) -> None:
        """Clear all stored transitions."""
        self.buffer.clear()
        self.position = 0
        self.is_full = False
        logger.debug("PPOBuffer cleared")
    
    def __len__(self) -> int:
        """Return number of stored transitions."""
        return len(self.buffer)
    
    def compute_returns_and_advantages(self, 
                                       values: np.ndarray,
                                       rewards: np.ndarray,
                                       dones: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute returns and advantages using GAE.
        
        Generalized Advantage Estimation (GAE):
        δ_t = r_t + γ*V(s_{t+1}) - V(s_t)
        A_t = δ_t + γ*λ*δ_{t+1} + γ²*λ²*δ_{t+2} + ...
        
        :param values: Value estimates V(s_t) for each timestep
        :param rewards: Rewards r_t for each timestep
        :param dones: Done flags for each timestep
        :return: Tuple of (returns, advantages)
        """
        n_steps = len(rewards)
        
        # Compute TD errors
        deltas = np.zeros(n_steps)
        for t in range(n_steps):
            if t < n_steps - 1:
                next_value = values[t + 1]
            else:
                next_value = 0.0
            
            # δ_t = r_t + γ*V(s_{t+1})*(1-done) - V(s_t)
            deltas[t] = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
        
        # Compute advantages using GAE
        advantages = np.zeros(n_steps)
        gae = 0.0
        for t in reversed(range(n_steps)):
            # A_t = δ_t + γ*λ*A_{t+1}
            gae = deltas[t] + self.gamma * self.lam * (1 - dones[t]) * gae
            advantages[t] = gae
        
        # Returns = advantages + values
        returns = advantages + values
        
        # Normalize advantages (standard practice)
        if advantages.std() > 1e-6:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return returns, advantages
    
    def to_batch(self, 
                device: str = 'cpu',
                compute_gae: bool = True) -> PPOBatch:
        """
        Convert buffer to training batch.
        
        :param device: Target device for tensors
        :param compute_gae: Whether to compute GAE advantages
        :return: PPOBatch with all tensors
        """
        if len(self.buffer) == 0:
            raise ValueError("Buffer is empty")
        
        # Extract arrays
        states = np.array([t.state.to_vector() for t in self.buffer])
        actions = np.array([t.action.raw_values for t in self.buffer])
        rewards = np.array([t.reward for t in self.buffer])
        next_states = np.array([t.next_state.to_vector() for t in self.buffer])
        dones = np.array([t.done for t in self.buffer])
        old_log_probs = np.array([t.log_prob for t in self.buffer])
        
        # Compute values for GAE
        # Note: This requires access to the critic network
        # For now, we'll use zeros as placeholder
        # The actual implementation will compute these from the network
        values = np.zeros(len(self.buffer))
        
        if compute_gae:
            returns, advantages = self.compute_returns_and_advantages(values, rewards, dones)
        else:
            returns = rewards
            advantages = np.zeros_like(rewards)
        
        # Convert to tensors
        batch = PPOBatch(
            states=torch.FloatTensor(states).to(device),
            actions=torch.FloatTensor(actions).to(device),
            rewards=torch.FloatTensor(rewards).to(device),
            next_states=torch.FloatTensor(next_states).to(device),
            dones=torch.FloatTensor(dones).to(device),
            old_log_probs=torch.FloatTensor(old_log_probs).to(device),
            returns=torch.FloatTensor(returns).to(device),
            advantages=torch.FloatTensor(advantages).to(device)
        )
        
        logger.debug(f"Created batch: {len(self.buffer)} samples, "
                    f"advantages mean={advantages.mean():.4f}, std={advantages.std():.4f}")
        
        return batch

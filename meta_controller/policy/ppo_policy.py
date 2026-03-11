"""
PPO Policy for DRL Meta-Controller.

Wraps the Actor-Critic network and provides a high-level interface
for action sampling and policy evaluation.
"""

import logging
from typing import Tuple, Optional, Dict, Any
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .actor_critic import ActorCriticNetwork, PPOConfig

logger = logging.getLogger(__name__)


class PPOPolicy:
    """
    PPO 策略 (PPO Policy)
    
    High-level interface for the PPO policy, wrapping ActorCriticNetwork.
    
    Responsibilities:
    - Action sampling with proper device handling
    - Log probability computation for PPO updates
    - Policy parameter access for optimization
    """
    
    def __init__(self, 
                 config: PPOConfig,
                 device: str = 'auto'):
        """
        Initialize PPO policy.
        
        :param config: PPO configuration
        :param device: Device to run on ('cpu', 'cuda', or 'auto')
        """
        self.config = config
        
        # Determine device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        logger.info(f"Initializing PPOPolicy on device: {self.device}")
        
        # Create network
        self.network = ActorCriticNetwork(
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            hidden_dims=config.hidden_dims,
            activation=config.activation,
            use_layer_norm=config.use_layer_norm
        ).to(self.device)
        
        # Create optimizer
        self.optimizer = optim.Adam(
            self.network.parameters(), 
            lr=config.learning_rate
        )
        
        # Training statistics
        self.total_updates = 0
        self.total_samples = 0
    
    def select_action(self, 
                     state: np.ndarray,
                     deterministic: bool = False) -> Tuple[np.ndarray, float]:
        """
        Select an action for a given state.
        
        :param state: State vector of shape (state_dim,)
        :param deterministic: Whether to use deterministic action
        :return: Tuple of (action, log_prob)
            - action: numpy array of shape (action_dim,)
            - log_prob: log probability of the action
        """
        # Convert to tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Sample action
        with torch.no_grad():
            action, log_prob = self.network.sample_action(
                state_tensor, 
                deterministic=deterministic
            )
        
        # Convert to numpy
        action_np = action.squeeze(0).cpu().numpy()
        log_prob_np = log_prob.squeeze(0).cpu().item()
        
        return action_np, log_prob_np
    
    def select_action_batch(self, 
                           states: np.ndarray,
                           deterministic: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Select actions for a batch of states.
        
        :param states: State matrix of shape (batch_size, state_dim)
        :param deterministic: Whether to use deterministic actions
        :return: Tuple of (actions, log_probs)
            - actions: numpy array of shape (batch_size, action_dim)
            - log_probs: numpy array of shape (batch_size,)
        """
        # Convert to tensor
        states_tensor = torch.FloatTensor(states).to(self.device)
        
        # Sample actions
        with torch.no_grad():
            actions, log_probs = self.network.sample_action(
                states_tensor,
                deterministic=deterministic
            )
        
        # Convert to numpy
        actions_np = actions.cpu().numpy()
        log_probs_np = log_probs.squeeze(-1).cpu().numpy()
        
        return actions_np, log_probs_np
    
    def evaluate_actions(self, 
                        states: torch.Tensor,
                        actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate log probabilities and values for state-action pairs.
        
        Used during PPO updates.
        
        :param states: State tensor of shape (batch_size, state_dim)
        :param actions: Action tensor of shape (batch_size, action_dim)
        :return: Tuple of (log_probs, values, entropy)
        """
        log_probs, values, entropy = self.network.evaluate_actions(states, actions)
        return log_probs, values, entropy
    
    def get_value(self, state: np.ndarray) -> float:
        """
        Get value estimate for a state.
        
        :param state: State vector
        :return: Value estimate
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            value = self.network.get_value(state_tensor).squeeze().cpu().item()
        
        return value
    
    def get_value_batch(self, states: np.ndarray) -> np.ndarray:
        """
        Get value estimates for a batch of states.
        
        :param states: State matrix
        :return: Value estimates
        """
        states_tensor = torch.FloatTensor(states).to(self.device)
        
        with torch.no_grad():
            values = self.network.get_value(states_tensor).squeeze(-1).cpu().numpy()
        
        return values
    
    def update(self, 
              states: torch.Tensor,
              actions: torch.Tensor,
              old_log_probs: torch.Tensor,
              advantages: torch.Tensor,
              returns: torch.Tensor) -> Dict[str, float]:
        """
        Perform one PPO update step.
        
        :param states: State tensor
        :param actions: Action tensor
        :param old_log_probs: Log probabilities under old policy
        :param advantages: Advantage estimates
        :param returns: Return estimates
        :return: Dictionary of loss components
        """
        # Evaluate current policy
        log_probs, values, entropy = self.evaluate_actions(states, actions)
        
        # Compute importance sampling ratio
        ratio = torch.exp(log_probs - old_log_probs)
        
        # Compute clipped surrogate loss
        surr1 = ratio * advantages.detach()
        surr2 = torch.clamp(ratio, 1 - self.config.clip_epsilon, 
                          1 + self.config.clip_epsilon) * advantages.detach()
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Compute value function loss
        value_loss = F.mse_loss(values.squeeze(), returns.detach())
        
        # Compute entropy bonus
        entropy_bonus = entropy.mean()
        
        # Total loss
        loss = (policy_loss 
                + self.config.value_coef * value_loss 
                - self.config.entropy_coef * entropy_bonus)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        nn.utils.clip_grad_norm_(self.network.parameters(), self.config.max_grad_norm)
        
        self.optimizer.step()
        
        # Update statistics
        self.total_updates += 1
        self.total_samples += len(states)
        
        # Log metrics
        metrics = {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy_bonus.item(),
            'ratio_mean': ratio.mean().item(),
            'ratio_std': ratio.std().item(),
            'approx_kl': (old_log_probs - log_probs).mean().item(),
            'clip_fraction': ((ratio - 1.0).abs() > self.config.clip_epsilon).float().mean().item()
        }
        
        return metrics
    
    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoint."""
        checkpoint = {
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config.to_dict(),
            'total_updates': self.total_updates,
            'total_samples': self.total_samples
        }
        torch.save(checkpoint, path)
        logger.info(f"Policy checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.total_updates = checkpoint['total_updates']
        self.total_samples = checkpoint['total_samples']
        logger.info(f"Policy checkpoint loaded from {path}")
    
    def get_parameters(self) -> list:
        """Get all policy parameters for optimization."""
        return list(self.network.parameters())
    
    def train(self) -> None:
        """Set network to training mode."""
        self.network.train()
    
    def eval(self) -> None:
        """Set network to evaluation mode."""
        self.network.eval()

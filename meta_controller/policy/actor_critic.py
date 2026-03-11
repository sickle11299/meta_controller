"""
Actor-Critic Network for DRL Meta-Controller.

Implements a shared encoder architecture with separate actor and critic heads,
optimized for continuous action spaces.
"""

import logging
from typing import List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class ActorCriticNetwork(nn.Module):
    """
    Actor-Critic 网络 (Actor-Critic Network)
    
    Architecture:
    - Shared feature extractor layers
    - Actor head: outputs action distribution parameters (μ, σ)
    - Critic head: outputs state value V(s)
    
    This design shares computation between policy and value functions,
    improving sample efficiency and training stability.
    """
    
    def __init__(self, 
                 state_dim: int = 11,
                 action_dim: int = 5,
                 hidden_dims: List[int] = [256, 128],
                 activation: str = 'tanh',
                 use_layer_norm: bool = True,
                 init_method: str = 'xavier'):
        """
        Initialize Actor-Critic network.
        
        :param state_dim: Input state dimension
        :param action_dim: Output action dimension
        :param hidden_dims: List of hidden layer dimensions
        :param activation: Activation function name
        :param use_layer_norm: Whether to use LayerNorm
        :param init_method: Weight initialization method
        """
        super(ActorCriticNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims
        
        # Activation function
        if activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Build shared layers
        self.shared_layers = self._build_shared_layers(use_layer_norm)
        
        # Actor head: outputs μ (mean) and log_std (log standard deviation)
        self.actor_mu = nn.Linear(hidden_dims[-1], action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        
        # Critic head: outputs scalar value
        self.critic = nn.Linear(hidden_dims[-1], 1)
        
        # Initialize weights
        self._init_weights(init_method)
        
        logger.info(f"ActorCriticNetwork created: {state_dim} -> {hidden_dims} -> "
                   f"actor({action_dim}) + critic(1)")
    
    def _build_shared_layers(self, use_layer_norm: bool) -> nn.Sequential:
        """Build shared feature extraction layers."""
        layers = []
        
        prev_dim = self.state_dim
        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            
            layers.append(self.activation)
            prev_dim = hidden_dim
        
        return nn.Sequential(*layers)
    
    def _init_weights(self, method: str) -> None:
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if method == 'xavier':
                    nn.init.xavier_uniform_(module.weight)
                elif method == 'orthogonal':
                    nn.init.orthogonal_(module.weight)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(module.weight)
                else:
                    raise ValueError(f"Unknown init method: {method}")
                
                nn.init.constant_(module.bias, 0.0)
        
        logger.debug(f"Weights initialized with {method} method")
    
    def forward(self, 
                state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.
        
        :param state: State tensor of shape (batch_size, state_dim)
        :return: Tuple of (mu, std, value)
            - mu: Action mean of shape (batch_size, action_dim)
            - std: Action std of shape (batch_size, action_dim)
            - value: State value of shape (batch_size, 1)
        """
        # Shared feature extraction
        features = self.shared_layers(state)
        
        # Actor output: action distribution parameters
        mu = self.actor_mu(features)
        std = torch.exp(self.actor_log_std).expand_as(mu)
        
        # Critic output: state value
        value = self.critic(features)
        
        return mu, std, value
    
    def get_distribution(self, 
                        state: torch.Tensor) -> torch.distributions.Normal:
        """
        Get action distribution for a state.
        
        :param state: State tensor
        :return: Normal distribution object
        """
        mu, std, _ = self.forward(state)
        return torch.distributions.Normal(mu, std)
    
    def get_value(self, state: torch.Tensor) -> torch.Tensor:
        """
        Get state value estimate.
        
        :param state: State tensor
        :return: Value tensor of shape (batch_size, 1)
        """
        _, _, value = self.forward(state)
        return value
    
    def sample_action(self, 
                     state: torch.Tensor,
                     deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample an action from the policy.
        
        :param state: State tensor
        :param deterministic: Whether to use deterministic action (mean)
        :return: Tuple of (action, log_prob)
        """
        dist = self.get_distribution(state)
        
        if deterministic:
            action = dist.mean
        else:
            action = dist.sample()
        
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        
        return action, log_prob
    
    def evaluate_actions(self, 
                        states: torch.Tensor,
                        actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate log probabilities and values for given state-action pairs.
        
        Used during PPO updates to compute importance sampling ratios.
        
        :param states: State tensor of shape (batch_size, state_dim)
        :param actions: Action tensor of shape (batch_size, action_dim)
        :return: Tuple of (log_probs, values, entropy)
        """
        dist = self.get_distribution(states)
        values = self.get_value(states).squeeze(-1)
        
        log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().mean()
        
        return log_probs, values, entropy


class PPOConfig:
    """Configuration container for PPO training."""
    
    def __init__(self,
                 state_dim: int = 11,
                 action_dim: int = 5,
                 learning_rate: float = 3e-4,
                 gamma: float = 0.99,
                 lam: float = 0.95,
                 clip_epsilon: float = 0.2,
                 epochs: int = 10,
                 batch_size: int = 64,
                 max_grad_norm: float = 0.5,
                 entropy_coef: float = 0.01,
                 value_coef: float = 0.5,
                 hidden_dims: List[int] = [256, 128],
                 activation: str = 'tanh',
                 use_layer_norm: bool = True):
        """Initialize PPO configuration."""
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.lam = lam
        self.clip_epsilon = clip_epsilon
        self.epochs = epochs
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.hidden_dims = hidden_dims
        self.activation = activation
        self.use_layer_norm = use_layer_norm
    
    @classmethod
    def from_dict(cls, config: dict) -> 'PPOConfig':
        """Create PPOConfig from dictionary."""
        return cls(**{k: v for k, v in config.items() if hasattr(cls(k), k)})
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'learning_rate': self.learning_rate,
            'gamma': self.gamma,
            'lam': self.lam,
            'clip_epsilon': self.clip_epsilon,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'max_grad_norm': self.max_grad_norm,
            'entropy_coef': self.entropy_coef,
            'value_coef': self.value_coef,
        }

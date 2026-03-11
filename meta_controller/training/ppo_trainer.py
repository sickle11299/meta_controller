"""
PPO Trainer for DRL Meta-Controller.

Implements the PPO-Clip algorithm with GAE advantage estimation.
Manages the training loop and optimization of the policy network.
"""

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ..policy.ppo_policy import PPOPolicy
from ..buffer.ppo_buffer import PPOBuffer, PPOTTransition
from ..policy.actor_critic import PPOConfig

logger = logging.getLogger(__name__)


class PPOTrainer:
    """
    PPO 训练器 (PPO Trainer)
    
    Implements Proximal Policy Optimization (PPO) with:
    - Clipped surrogate objective
    - Generalized Advantage Estimation (GAE)
    - Multiple epochs of updates on same data
    - Adaptive KL penalty monitoring
    
    Corresponds to Algorithm 1, Line 11: "update policy/value networks via PPO"
    """
    
    def __init__(self, 
                 policy: PPOPolicy,
                 config: PPOConfig):
        """
        Initialize PPO trainer.
        
        :param policy: PPO policy to train
        :param config: PPO configuration
        """
        self.policy = policy
        self.config = config
        
        logger.info(f"PPOTrainer initialized with config: "
                   f"lr={config.learning_rate}, gamma={config.gamma}, "
                   f"lam={config.lam}, clip_epsilon={config.clip_epsilon}")
    
    def update(self, buffer: PPOBuffer) -> Dict[str, float]:
        """
        Perform PPO update using collected transitions.
        
        This is the main training method that implements the PPO algorithm.
        
        :param buffer: Replay buffer with collected transitions
        :return: Dictionary of training metrics
        """
        if len(buffer) < self.config.batch_size:
            logger.warning(f"Insufficient samples in buffer: {len(buffer)} < {self.config.batch_size}")
            return {'error': 'insufficient_samples'}
        
        # Set policy to training mode
        self.policy.train()
        
        # Get batch data
        batch = buffer.to_batch(device=self.policy.device, compute_gae=True)
        
        # Update values using current critic
        with torch.no_grad():
            values = self.policy.network.get_value(batch.states).squeeze().cpu().numpy()
            next_values = self.policy.network.get_value(batch.next_states).squeeze().cpu().numpy()
        
        # Recompute returns and advantages with actual values
        rewards = batch.rewards.cpu().numpy()
        dones = batch.dones.cpu().numpy()
        
        # Compute TD errors
        n_steps = len(rewards)
        deltas = np.zeros(n_steps)
        for t in range(n_steps):
            deltas[t] = rewards[t] + self.config.gamma * next_values[t] * (1 - dones[t]) - values[t]
        
        # Compute GAE advantages
        advantages = np.zeros(n_steps)
        gae = 0.0
        for t in reversed(range(n_steps)):
            gae = deltas[t] + self.config.gamma * self.config.lam * (1 - dones[t]) * gae
            advantages[t] = gae
        
        # Normalize advantages
        if advantages.std() > 1e-6:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Convert to tensors
        advantages_tensor = torch.FloatTensor(advantages).to(self.policy.device)
        returns_tensor = advantages_tensor + torch.FloatTensor(values).to(self.policy.device)
        
        # Create dataset
        dataset = TensorDataset(
            batch.states,
            batch.actions,
            batch.old_log_probs,
            advantages_tensor,
            returns_tensor
        )
        
        # Create data loader
        dataloader = DataLoader(
            dataset,
            batch_size=min(self.config.batch_size, len(dataset)),
            shuffle=True
        )
        
        # Training metrics
        all_metrics = []
        
        # Multiple epochs of updates
        for epoch in range(self.config.epochs):
            epoch_losses = []
            
            for batch_data in dataloader:
                states, actions, old_log_probs, advantages, returns = batch_data
                
                # Perform one update step
                metrics = self.policy.update(
                    states=states,
                    actions=actions,
                    old_log_probs=old_log_probs,
                    advantages=advantages,
                    returns=returns
                )
                
                epoch_losses.append(metrics)
            
            # Average metrics for this epoch
            epoch_metrics = self._average_metrics(epoch_losses)
            all_metrics.append(epoch_metrics)
            
            logger.debug(f"PPO epoch {epoch+1}/{self.config.epochs}: "
                        f"policy_loss={epoch_metrics['policy_loss']:.4f}, "
                        f"value_loss={epoch_metrics['value_loss']:.4f}")
        
        # Clear buffer after update
        buffer.clear()
        
        # Aggregate metrics across all epochs
        final_metrics = self._average_metrics(all_metrics)
        final_metrics['n_samples'] = len(buffer.buffer)
        final_metrics['n_updates'] = self.policy.total_updates
        
        logger.info(f"PPO update completed: "
                   f"policy_loss={final_metrics['policy_loss']:.4f}, "
                   f"value_loss={final_metrics['value_loss']:.4f}, "
                   f"entropy={final_metrics['entropy']:.4f}")
        
        return final_metrics
    
    def _average_metrics(self, metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
        """Average a list of metric dictionaries."""
        if len(metrics_list) == 0:
            return {}
        
        avg_metrics = {}
        for key in metrics_list[0].keys():
            values = [m[key] for m in metrics_list if key in m]
            if len(values) > 0:
                avg_metrics[key] = np.mean(values)
        
        return avg_metrics
    
    def update_from_transitions(self, 
                               transitions: List[PPOTTransition]) -> Dict[str, float]:
        """
        Update policy from a list of transitions (convenience method).
        
        :param transitions: List of transitions
        :return: Training metrics
        """
        # Create temporary buffer
        temp_buffer = PPOBuffer(
            capacity=len(transitions) + 1,
            gamma=self.config.gamma,
            lam=self.config.lam
        )
        
        # Store transitions
        for t in transitions:
            temp_buffer.store(t)
        
        # Update
        return self.update(temp_buffer)
    
    def compute_gae(self, 
                   rewards: np.ndarray,
                   values: np.ndarray,
                   next_values: np.ndarray,
                   dones: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Generalized Advantage Estimates.
        
        Utility method for computing GAE outside of training loop.
        
        :param rewards: Rewards at each timestep
        :param values: Value estimates V(s_t)
        :param next_values: Value estimates V(s_{t+1})
        :param dones: Done flags
        :return: Tuple of (advantages, returns)
        """
        n_steps = len(rewards)
        
        # Compute TD errors
        deltas = rewards + self.config.gamma * next_values * (1 - dones) - values
        
        # Compute advantages
        advantages = np.zeros(n_steps)
        gae = 0.0
        for t in reversed(range(n_steps)):
            gae = deltas[t] + self.config.gamma * self.config.lam * (1 - dones[t]) * gae
            advantages[t] = gae
        
        # Normalize
        if advantages.std() > 1e-6:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Returns
        returns = advantages + values
        
        return advantages, returns
    
    def get_training_stats(self) -> Dict[str, int]:
        """Get training statistics."""
        return {
            'total_updates': self.policy.total_updates,
            'total_samples': self.policy.total_samples
        }

"""
Control Loop Runner for DRL Meta-Controller.

Implements the main execution loop of Algorithm 1, coordinating all components
of the meta-controller system.
"""

import logging
import asyncio
from typing import Optional, Dict, Any, List
import numpy as np

from meta_controller.metrics.types import SystemMetrics, GradientFeatures, MetaState
from meta_controller.buffer.ppo_buffer import PPOBuffer, PPOTTransition
from meta_controller.metrics.collector import MetricsCollector
from meta_controller.features.builder import FeatureBuilder
from meta_controller.features.state_encoder import MetaStateEncoder
from meta_controller.policy.ppo_policy import PPOPolicy
from meta_controller.action.mapper import ActionMapper, MetaAction, SchedulerParams
from meta_controller.reward.hazard import HazardRateCalculator
from meta_controller.reward.calculator import RewardCalculator, RewardComponents
from meta_controller.training.ppo_trainer import PPOTrainer
from meta_controller.policy.actor_critic import PPOConfig
from meta_controller.interface.utaa_client import UTAAInterface

logger = logging.getLogger(__name__)


class ControlLoopRunner:
    """
    控制循环执行器 (Control Loop Runner)
    
    Implements the complete DRL Meta-Controller adaptation loop from Algorithm 1.
    
    Execution flow per iteration:
    1. Collect metric history H over observation window W (Line 1)
    2. Compute gradients ∇T, ∇R (Line 2)
    3. Construct state st (Line 3)
    4. Sample action at ~ πθ(.|st) (Line 4)
    5. Map action to scheduler parameters (Line 5)
    6. Broadcast parameters to UTAA (Line 6)
    7. Wait for duration ΔT (Line 7)
    8. Compute hazard integral H = ∫λ(t)dt (Line 8)
    9. Compute reward rt (Line 9)
    10. Store transition (st, at, rt, st+1) (Line 10)
    11. Update policy via PPO when enough samples (Line 11)
    """
    
    def __init__(self, 
                 config: Dict[str, Any],
                 utaa_interface: UTAAInterface):
        """
        Initialize control loop runner.
        
        :param config: Configuration dictionary (from YAML)
        :param utaa_interface: Interface to UTAA scheduler
        """
        self.config = config
        self.utaa_interface = utaa_interface
        
        # Extract configuration
        control_config = config.get('control', {})
        state_config = config.get('state', {})
        action_config = config.get('action', {})
        ppo_config = config.get('ppo', {})
        reward_config = config.get('reward', {})
        hazard_config = config.get('hazard', {})
        
        # Initialize components
        self._init_components(
            control_config, state_config, action_config,
            ppo_config, reward_config, hazard_config
        )
        
        # State tracking
        self.iteration = 0
        self.previous_action: Optional[MetaAction] = None
        self.last_reward: Optional[float] = None
        self.episode_rewards: List[float] = []
        
        logger.info("ControlLoopRunner initialized")
    
    def _init_components(self,
                        control_config: dict,
                        state_config: dict,
                        action_config: dict,
                        ppo_config: dict,
                        reward_config: dict,
                        hazard_config: dict) -> None:
        """Initialize all sub-components."""
        
        # Metrics collector
        self.metrics_collector = MetricsCollector(
            utaa_interface=self.utaa_interface,
            window_size=control_config.get('history_window', 10)
        )
        
        # Feature builder
        self.feature_builder = FeatureBuilder(
            delta_t=1.0,
            order=1,
            use_center_diff=True
        )
        
        # State encoder
        self.state_encoder = MetaStateEncoder(
            normalize=state_config.get('normalize', True),
            method='zscore',
            clip_range=tuple(state_config.get('clip_range', [-5.0, 5.0]))
        )
        
        # Action mapper
        self.action_mapper = ActionMapper(
            epsilon_base=action_config.get('epsilon_base', 0.1),
            beta=action_config.get('beta', 1.0)
        )
        
        # PPO policy
        ppo_cfg = PPOConfig(
            state_dim=state_config.get('dim', 11),
            action_dim=action_config.get('dim', 5),
            learning_rate=ppo_config.get('learning_rate', 3e-4),
            gamma=ppo_config.get('gamma', 0.99),
            lam=ppo_config.get('lam', 0.95),
            clip_epsilon=ppo_config.get('clip_epsilon', 0.2),
            epochs=ppo_config.get('epochs', 10),
            batch_size=ppo_config.get('batch_size', 64),
            max_grad_norm=ppo_config.get('max_grad_norm', 0.5),
            entropy_coef=ppo_config.get('entropy_coef', 0.01),
            value_coef=ppo_config.get('value_coef', 0.5)
        )
        self.policy = PPOPolicy(ppo_cfg)
        
        # Replay buffer
        self.buffer = PPOBuffer(
            capacity=ppo_config.get('batch_size', 64) * 2,
            gamma=ppo_cfg.gamma,
            lam=ppo_cfg.lam
        )
        
        # PPO trainer
        self.trainer = PPOTrainer(self.policy, ppo_cfg)
        
        # Hazard calculator
        self.hazard_calculator = HazardRateCalculator(
            baseline_hazard=hazard_config.get('baseline', 0.1),
            coefficients=np.array(hazard_config.get('coefficients', [0.5, 0.3, 0.2, 0.1])),
            dt=hazard_config.get('dt', 1.0),
            model=hazard_config.get('model', 'cox')
        )
        
        # Reward calculator
        self.reward_calculator = RewardCalculator(
            psi=reward_config.get('psi', 1.0),
            xi=reward_config.get('xi', 0.5),
            phi=reward_config.get('phi', 0.1)
        )
        
        # Control parameters
        self.control_period = control_config.get('period_seconds', 60)
        self.max_iterations = control_config.get('max_iterations', 1000)
        
        logger.info("All components initialized")
    
    async def run(self, num_iterations: Optional[int] = None) -> Dict[str, Any]:
        """
        Run the complete control loop.
        
        Implements Algorithm 1 end-to-end.
        
        :param num_iterations: Number of iterations to run (uses config if None)
        :return: Training statistics
        """
        n_iter = num_iterations or self.max_iterations
        
        logger.info(f"Starting control loop for {n_iter} iterations")
        
        for i in range(n_iter):
            self.iteration = i
            
            try:
                # Execute one iteration of Algorithm 1
                await self.run_single_iteration()
                
                # Log progress
                if i % 10 == 0:
                    logger.info(f"Iteration {i}/{n_iter}: "
                               f"reward={self.last_reward:.4f}")
            
            except Exception as e:
                logger.error(f"Iteration {i} failed: {e}", exc_info=True)
                # Continue to next iteration
        
        # Return final statistics
        return self._get_statistics()
    
    async def run_single_iteration(self) -> None:
        """
        Execute a single iteration of Algorithm 1.
        
        This is the core method that implements all 11 lines of Algorithm 1.
        """
        # === Phase 1: Observation ===
        
        # Line 1: Collect metric history H over observation window W
        history = await self.metrics_collector.collect_window()
        
        # Line 2: Compute gradients ∇T, ∇R
        gradients = self.feature_builder.compute_gradients(history)
        
        # Line 3: Construct state st
        current_metrics = self.metrics_collector.get_latest()
        state = MetaState(
            current_metrics=current_metrics,
            gradients=gradients,
            history_window=history
        )
        state_vector = self.state_encoder.encode(state)
        
        # === Phase 2: Action Inference ===
        
        # Line 4: Sample action at ~ πθ(.|st)
        action_raw, log_prob = self.policy.select_action(state_vector)
        action = MetaAction(raw_values=action_raw)
        
        # Line 5: Map action to scheduler parameters
        params = self.action_mapper.map_to_params(action)
        
        # === Phase 3: Broadcast to UTAA ===
        
        # Line 6: Send parameters to UTAA
        success = await self.utaa_interface.send_params(params)
        if not success:
            logger.warning("Failed to send parameters to UTAA")
        
        # Line 7: Wait for duration ΔT
        await asyncio.sleep(self.control_period)
        
        # === Phase 4: Reward Computation ===
        
        # Line 8: Compute hazard integral
        metrics_after = await self.utaa_interface.request_metrics()
        hazard_rates = self.hazard_calculator.compute_sequence(history)
        hazard_integral = self.hazard_calculator.compute_integral(hazard_rates)
        
        # Get task statistics
        n_success, n_total = self.utaa_interface.mock_env.get_task_stats()
        
        # Line 9: Compute reward rt
        reward = self.reward_calculator.calculate(
            n_success=n_success,
            n_total=n_total,
            hazard_integral=hazard_integral,
            current_action=action.raw_values,
            previous_action=self.previous_action.raw_values if self.previous_action else None
        )
        
        self.last_reward = reward.total_reward
        
        # Line 10: Store transition (st, at, rt, st+1)
        next_state = MetaState(
            current_metrics=metrics_after,
            gradients=self.feature_builder.compute_gradients(history[-5:]),  # Recent gradients
            history_window=history
        )
        
        transition = PPOTTransition(
            state=state,
            action=action,
            reward=reward.total_reward,
            next_state=next_state,
            done=False,
            log_prob=log_prob
        )
        
        self.buffer.store(transition)
        
        # Line 11: Update policy via PPO (when enough samples collected)
        ppo_config = self.config.get('ppo', {})
        batch_size = ppo_config.get('batch_size', 64)
        
        if len(self.buffer) >= batch_size:
            update_metrics = self.trainer.update(self.buffer)
            logger.debug(f"PPO update completed: {update_metrics}")
        
        # Track statistics
        self.episode_rewards.append(reward.total_reward)
        self.previous_action = action
    
    def _get_statistics(self) -> Dict[str, Any]:
        """Get training statistics."""
        stats = {
            'iterations': self.iteration + 1,
            'total_samples': self.policy.total_samples,
            'total_updates': self.policy.total_updates,
            'last_reward': self.last_reward,
            'avg_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0.0,
            'final_w4': self.utaa_interface.mock_env.w4,
            'final_epsilon_risk': self.utaa_interface.mock_env.epsilon_risk
        }
        return stats
    
    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoint."""
        self.policy.save_checkpoint(path)
        logger.info(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint."""
        self.policy.load_checkpoint(path)
        logger.info(f"Checkpoint loaded from {path}")

"""
DRL Meta-Controller Module for LCRSF Framework

This module implements the DRL Meta-Controller Adaptation Loop (Algorithm 1)
for periodic hyperparameter adaptation in the UTAA scheduler.
"""

__version__ = "0.1.0"
__author__ = "LCRSF Team"

# Import submodules first to avoid circular dependencies
from . import metrics
from . import features  
from . import policy
from . import action
from . import reward
from . import buffer
from . import training
from . import interface

# Then import specific classes
from .metrics.types import SystemMetrics, GradientFeatures, MetaState
from .features.builder import FeatureBuilder
from .features.state_encoder import MetaStateEncoder
from .policy.actor_critic import ActorCriticNetwork
from .policy.ppo_policy import PPOPolicy
from .action.mapper import ActionMapper, MetaAction, SchedulerParams
from .reward.hazard import HazardRateCalculator
from .reward.calculator import RewardCalculator, RewardComponents
from .buffer.ppo_buffer import PPOBuffer, PPOTTransition
from .training.ppo_trainer import PPOTrainer, PPOConfig
from .interface.utaa_client import UTAAInterface, MockUTAAEnvironment
from .control_loop import ControlLoopRunner

__all__ = [
    # Data types
    "SystemMetrics",
    "GradientFeatures", 
    "MetaState",
    "MetaAction",
    "SchedulerParams",
    "PPOTTransition",
    "RewardComponents",
    "PPOConfig",
    
    # Modules
    "FeatureBuilder",
    "MetaStateEncoder",
    "ActorCriticNetwork",
    "PPOPolicy",
    "ActionMapper",
    "HazardRateCalculator",
    "RewardCalculator",
    "PPOBuffer",
    "PPOTrainer",
    "UTAAInterface",
    "MockUTAAEnvironment",
    "ControlLoopRunner",
]

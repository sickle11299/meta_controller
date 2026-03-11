"""
Metrics Collector for DRL Meta-Controller.

Collects system metrics from UTAA scheduler and maintains a sliding window
of historical data for gradient computation.
"""

import logging
from typing import List, Optional, Deque
from collections import deque
import asyncio

from ..metrics.types import SystemMetrics
from ..interface.mock_env import MockUTAAEnvironment
from ..interface.utaa_client import UTAAInterface

logger = logging.getLogger(__name__)


class MetricsCollector:
    """
    指标采集器 (Metrics Collector)
    
    Responsible for:
    - Collecting metric snapshots from UTAA (real or mock)
    - Maintaining a sliding window of history H
    - Providing latest metrics and historical data
    
    Corresponds to Algorithm 1, Line 1: "collect metric history H over observation window W"
    """
    
    def __init__(self, 
                 utaa_interface: UTAAInterface,
                 window_size: int = 10):
        """
        Initialize metrics collector.
        
        :param utaa_interface: Interface to UTAA scheduler
        :param window_size: Size of history window W
        """
        self.utaa_interface = utaa_interface
        self.window_size = window_size
        self.history: Deque[SystemMetrics] = deque(maxlen=window_size)
        
        logger.info(f"MetricsCollector initialized with window_size={window_size}")
    
    async def collect_once(self) -> SystemMetrics:
        """
        Collect a single metric snapshot from UTAA.
        
        :return: SystemMetrics snapshot
        """
        try:
            metrics = await self.utaa_interface.request_metrics()
            
            # Add to history
            self.history.append(metrics)
            
            logger.debug(f"Collected metrics at t={metrics.timestamp:.2f}",
                        extra={'cpu_load': metrics.cpu_load,
                              'task_success_rate': metrics.task_success_rate})
            
            return metrics
        
        except Exception as e:
            logger.error(f"Failed to collect metrics: {e}")
            raise
    
    async def collect_window(self) -> List[SystemMetrics]:
        """
        Collect a full window of historical metrics.
        
        If history is incomplete, collects additional samples.
        
        :return: List of SystemMetrics with length <= window_size
        """
        # Check if we need to collect more samples
        samples_needed = self.window_size - len(self.history)
        
        if samples_needed > 0:
            logger.debug(f"Collecting {samples_needed} additional samples")
            
            for _ in range(samples_needed):
                try:
                    metrics = await self.utaa_interface.request_metrics()
                    self.history.append(metrics)
                    await asyncio.sleep(0.1)  # Small delay between samples
                except Exception as e:
                    logger.warning(f"Failed to collect sample: {e}")
                    break
        
        return list(self.history)
    
    def get_latest(self) -> Optional[SystemMetrics]:
        """
        Get the most recent metric snapshot.
        
        :return: Latest SystemMetrics or None if history is empty
        """
        if len(self.history) == 0:
            return None
        return self.history[-1]
    
    def get_history(self) -> List[SystemMetrics]:
        """
        Get the full history window.
        
        :return: List of SystemMetrics (oldest to newest)
        """
        return list(self.history)
    
    def get_history_array(self) -> Optional['np.ndarray']:
        """
        Get history as numpy array for batch processing.
        
        :return: numpy array of shape (window_size, 7) or None
        """
        import numpy as np
        
        if len(self.history) == 0:
            return None
        
        return np.array([m.to_vector() for m in self.history])
    
    def clear_history(self) -> None:
        """Clear all historical data."""
        self.history.clear()
        logger.debug("Metrics history cleared")
    
    @property
    def is_ready(self) -> bool:
        """Check if enough history has been collected."""
        return len(self.history) >= self.window_size
    
    def __len__(self) -> int:
        """Return number of samples in history."""
        return len(self.history)

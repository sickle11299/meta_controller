"""
UTAA Interface for DRL Meta-Controller.

Provides communication interface between the meta-controller and UTAA scheduler.
Supports both mock (for testing) and real (REST/gRPC) protocols.
"""

import logging
from typing import Optional, Callable, Dict, Any
import asyncio

from ..metrics.types import SystemMetrics
from ..action.mapper import SchedulerParams
from .mock_env import MockUTAAEnvironment

logger = logging.getLogger(__name__)


class UTAAInterface:
    """
    UTAA 通信接口 (UTAA Communication Interface)
    
    Abstracts communication with the UTAA scheduler.
    
    Responsibilities:
    - Send hyperparameters (w4, εrisk) to UTAA
    - Request current system metrics
    - Handle protocol-specific details (mock/REST/gRPC)
    
    Corresponds to Algorithm 1, Line 6: "broadcast (w, εrisk) to UTAA"
    """
    
    def __init__(self, 
                 mock_env: Optional[MockUTAAEnvironment] = None,
                 protocol: str = 'mock',
                 endpoint: Optional[str] = None,
                 timeout: float = 5.0):
        """
        Initialize UTAA interface.
        
        :param mock_env: Mock environment for testing (required if protocol='mock')
        :param protocol: Communication protocol: 'mock', 'rest', or 'grpc'
        :param endpoint: REST/gRPC endpoint URL (required if protocol!='mock')
        :param timeout: Request timeout in seconds
        """
        self.protocol = protocol
        self.timeout = timeout
        
        if protocol == 'mock':
            if mock_env is None:
                raise ValueError("mock_env required for mock protocol")
            self.mock_env = mock_env
            logger.info("UTAAInterface initialized in MOCK mode")
        
        elif protocol == 'rest':
            if endpoint is None:
                raise ValueError("endpoint required for REST protocol")
            self.endpoint = endpoint
            # Will be implemented in Phase F
            logger.info(f"UTAAInterface initialized in REST mode: {endpoint}")
        
        elif protocol == 'grpc':
            if endpoint is None:
                raise ValueError("endpoint required for gRPC protocol")
            self.endpoint = endpoint
            # Will be implemented in Phase F
            logger.info(f"UTAAInterface initialized in gRPC mode: {endpoint}")
        
        else:
            raise ValueError(f"Unknown protocol: {protocol}")
    
    async def send_params(self, params: SchedulerParams) -> bool:
        """
        Send hyperparameters to UTAA scheduler.
        
        Implements Algorithm 1, Line 6.
        
        :param params: Scheduler parameters to send
        :return: True if successful
        """
        try:
            if self.protocol == 'mock':
                return await self._send_params_mock(params)
            elif self.protocol == 'rest':
                return await self._send_params_rest(params)
            elif self.protocol == 'grpc':
                return await self._send_params_grpc(params)
            else:
                raise ValueError(f"Unknown protocol: {self.protocol}")
        
        except Exception as e:
            logger.error(f"Failed to send params: {e}")
            return False
    
    async def _send_params_mock(self, params: SchedulerParams) -> bool:
        """Send parameters to mock environment."""
        self.mock_env.update_params(params.w4, params.epsilon_risk)
        logger.debug(f"[MOCK] Sent params: w4={params.w4:.4f}, "
                    f"epsilon_risk={params.epsilon_risk:.4f}")
        return True
    
    async def _send_params_rest(self, params: SchedulerParams) -> bool:
        """Send parameters via REST API (placeholder for Phase F)."""
        # TODO: Implement REST API call
        logger.warning("REST protocol not yet implemented")
        return False
    
    async def _send_params_grpc(self, params: SchedulerParams) -> bool:
        """Send parameters via gRPC (placeholder for Phase F)."""
        # TODO: Implement gRPC call
        logger.warning("gRPC protocol not yet implemented")
        return False
    
    async def request_metrics(self) -> SystemMetrics:
        """
        Request current system metrics from UTAA.
        
        :return: SystemMetrics snapshot
        """
        try:
            if self.protocol == 'mock':
                return await self._request_metrics_mock()
            elif self.protocol == 'rest':
                return await self._request_metrics_rest()
            elif self.protocol == 'grpc':
                return await self._request_metrics_grpc()
            else:
                raise ValueError(f"Unknown protocol: {self.protocol}")
        
        except Exception as e:
            logger.error(f"Failed to request metrics: {e}")
            raise
    
    async def _request_metrics_mock(self) -> SystemMetrics:
        """Request metrics from mock environment."""
        metrics = self.mock_env.step(delta_t=1.0)
        logger.debug(f"[MOCK] Received metrics at t={metrics.timestamp:.2f}")
        return metrics
    
    async def _request_metrics_rest(self) -> SystemMetrics:
        """Request metrics via REST API (placeholder for Phase F)."""
        # TODO: Implement REST API call
        logger.warning("REST protocol not yet implemented")
        raise NotImplementedError("REST metrics request not implemented")
    
    async def _request_metrics_grpc(self) -> SystemMetrics:
        """Request metrics via gRPC (placeholder for Phase F)."""
        # TODO: Implement gRPC call
        logger.warning("gRPC protocol not yet implemented")
        raise NotImplementedError("gRPC metrics request not implemented")
    
    async def subscribe_metrics(self, callback: Callable[[SystemMetrics], None]) -> None:
        """
        Subscribe to metric updates (asynchronous push).
        
        :param callback: Function to call with new metrics
        """
        # TODO: Implement subscription mechanism
        logger.warning("Metric subscription not yet implemented")
    
    async def close(self) -> None:
        """Close connection and cleanup resources."""
        logger.info("UTAAInterface closed")

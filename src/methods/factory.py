"""Factory for creating pricing method instances."""

from typing import Dict, Any
from ..core.types import PricingMethod
from .base import BasePricingMethod
from .lp import LPMethod
from .minmax_costflow import MinMaxCostFlowMethod
from .maps import MAPSMethod
from .linucb import LinUCBMethod


class PricingMethodFactory:
    """Factory for creating pricing method instances."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize factory with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
    
    def create(self, method: PricingMethod) -> BasePricingMethod:
        """
        Create a pricing method instance.
        
        Args:
            method: Pricing method type
            
        Returns:
            Pricing method instance
            
        Raises:
            NotImplementedError: If method is not implemented
        """
        if method == PricingMethod.LP:
            return LPMethod(self.config)
        elif method == PricingMethod.MIN_MAX_COST_FLOW:
            return MinMaxCostFlowMethod(self.config)
        elif method == PricingMethod.MAPS:
            return MAPSMethod(self.config)
        elif method == PricingMethod.LIN_UCB:
            return LinUCBMethod(self.config)
        else:
            raise NotImplementedError(f"Method {method} not implemented") 
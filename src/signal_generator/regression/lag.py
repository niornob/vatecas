import numpy as np
import pandas as pd
from typing import Dict, Optional
from math import factorial

from signal_generator.regression.base.oracle import Oracle

class Lag(Oracle):
    """
    A simple Oracle subclass that predicts future values based on lagged historical values.
    
    This oracle implements a naive forecasting approach where the prediction for each asset
    is simply the value from 'lag_days' periods ago. For example, with lag_days=1, 
    it predicts tomorrow's price will be the same as today's price.
    """
    
    def __init__(
        self,
        lag_days: int = 1,
        name: str = "Lag Oracle",
        version: str = "1.0",
        params: Optional[Dict[str, object]] = None,
    ):
        """
        Initialize the Lag Oracle.
        
        Args:
            lag_days: Number of days to lag. Default is 1 (use previous day's value)
            name: Human-readable name for the oracle
            version: Version identifier for the oracle
            params: Dictionary of parameters specific to the oracle implementation
        """
        super().__init__(name=name, version=version, params=params)
        self.lag_days = lag_days
        
        if lag_days <= 0:
            raise ValueError("lag_days must be a positive integer")
    
    def _predict_mean(self, data: Dict[str, pd.Series]) -> np.ndarray:
        """
        Predict mean values using lagged historical data.
        
        This method returns the values from 'lag_days' periods ago for each ticker.
        If there isn't enough historical data, it returns the earliest available value.
        
        Args:
            data: Dictionary mapping ticker symbols to historical price series
            
        Returns:
            Array of mean predictions for each ticker in the same order as data.keys()
            
        Raises:
            ValueError: If data is empty or series have insufficient length
        """
        if not data:
            raise ValueError("Input data cannot be empty")
        
        predictions = [s.iloc[-1] for s in data.values()]
        
        return np.array(predictions)
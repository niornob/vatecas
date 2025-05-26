from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple
import pandas as pd
import numpy as np
from collections import deque


class SignalModule(ABC):
    """
    Abstract base class for all signal modules.
    Subclasses must implement the `generate_signals` method.

    Attributes
    ----------
    name : str
        A human-readable identifier for the signal module.
    version : str
        Version string to track changes/hyperparameters.
    params : dict
        Dictionary of parameters or hyperparameters used by the module.
    """

    def __init__(
        self, name: str = "", version: str = "0.0", params: dict[str, object] = {}
    ):
        self.name = name
        self.version = version
        self.params = params

    @abstractmethod
    def generate_signals(
        self, 
        data: dict[str, pd.DataFrame],
        prediction_history: deque = deque([])
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Generate signals for each ticker in the input data.

        Parameters
        ----------
        data : Mapping[str, pd.DataFrame]
            Dictionary mapping each ticker symbol to a pandas DataFrame.
            Each DataFrame must be time-indexed and contain the columns:
            ['adjClose', 'adjOpen', 'high', 'low', 'volume'].
        
        prediction_history : deque
            A list of some fixed the most recent predictions made.
            If empty, it will be computed first before generating new signal.

        Returns
        -------
        signals : Mapping[str, float]
            Dictionary mapping each ticker symbol to a float between 0
            and 1. 0 will indicate strong sell and 1 strong buy signal.

        predicted_prices : Mapping[str, float]
            Predicted prices for tomorrow, for all tickers in the data.
        """
        ...  # To be implemented in subclass


from abc import ABC, abstractmethod
from typing import Any
import pandas as pd


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
        self, data: dict[str, pd.DataFrame]
    ) -> dict[str, float]:
        """
        Generate signals for each ticker in the input data.

        Parameters
        ----------
        data : Mapping[str, pd.DataFrame]
            Dictionary mapping each ticker symbol to a pandas DataFrame.
            Each DataFrame must be time-indexed and contain the columns:
            ['adjClose', 'close', 'high', 'low', 'open', 'volume'].

        Returns
        -------
        signals : Mapping[str, float]
            Dictionary mapping each ticker symbol to a float between 0
            and 1. 0 will indicate strong sell and 1 strong buy signal.
        """
        ...  # To be implemented in subclass

    @abstractmethod
    def diagnostics(
        self,
        **kwargs
    ) -> Any:
        """
        do diagnostic tests on the signal generator.
        """
        ...

from abc import ABC, abstractmethod
from typing import Mapping
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

    def __init__(self, name: str, version: str = '0.0', params: dict[str, object] = {}):
        self.name = name
        self.version = version
        self.params = params

    @abstractmethod
    def generate_signals(
        self,
        data: Mapping[str, pd.DataFrame],
        diagnostics: bool = False
    ) -> Mapping[str, pd.Series]:
        """
        Generate time-series signals for each ticker in the input data.

        Parameters
        ----------
        data : Mapping[str, pd.DataFrame]
            Dictionary mapping each ticker symbol to a pandas DataFrame.
            Each DataFrame must be time-indexed and contain the columns:
            ['Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume'].

        Returns
        -------
        signals : Mapping[str, pd.DataFrame]
            Dictionary mapping each ticker symbol to a pandas DataFrame
            of signals. Each returned DataFrame must:

            - Have the same index as `data[ticker]`.
            - Contain one or more float columns in [0.0, 1.0], e.g.
              'buy_confidence'.

            Example
            -------
            >>> signals['AAPL'].head()
                         buy_confidence
            2025-01-02          0.132432
            2025-01-02          0.673241
            ...
        """
        ...  # To be implemented in subclass

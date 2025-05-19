import pandas as pd
import numpy as np
from abc import ABC, abstractmethod



# ========================
# TRADE FREQUENCY CONTROLLERS
# ========================


class TradeFrequencyController(ABC):
    """
    Abstract base class for controlling trading frequency.
    Subclasses define how to determine if a trade should be executed based on timing.
    """

    @abstractmethod
    def should_trade(
        self,
        ticker: str,
        signal: float,
        current_time: pd.Timestamp,
        last_trade_time: pd.Timestamp,
    ) -> bool:
        """
        Determine if a trade should be executed based on timing and signal strength.

        Args:
            ticker: Asset ticker symbol.
            signal: Trading signal value between -1 and 1.
            current_time: Current timestamp.
            last_trade_time: Timestamp of the last trade for this asset.

        Returns:
            Boolean indicating whether to execute the trade.
        """
        pass


class ThresholdDecayController(TradeFrequencyController):
    """
    Controls trade frequency using a decaying threshold mechanism.
    The threshold for trading decreases over time since the last trade.
    """

    def __init__(
        self,
        trades_per_period: int = 1,
        period_length_days: int = 5,
        tau_max: float = 1.0,
        tau_min: float = 0.1,
        decay: str = "linear",
    ):
        """
        Initialize the threshold decay controller.

        Args:
            trades_per_period: Desired trades per period.
            period_length_days: Period length in trading days.
            tau_max: Maximum threshold immediately after a trade.
            tau_min: Minimum threshold after full decay.
            decay: Type of decay, either 'linear' or 'exponential'.
        """
        self.trades_per_period = trades_per_period
        self.period_length_days = period_length_days
        self.tau_max = tau_max
        self.tau_min = tau_min
        self.decay = decay

        if self.decay == "exponential" and self.tau_min <= 0:
            raise ValueError("tau_min must be positive for exponential decay.")

        # Compute average inter-trade interval
        self.dt_avg = self.period_length_days / max(self.trades_per_period, 1)

        # Compute decay constant for exponential decay
        if self.decay == "exponential" and self.tau_min > 0:
            self.lambda_ = -np.log(self.tau_min / self.tau_max) / self.dt_avg
        else:
            self.lambda_ = -float("inf")

    def _calculate_threshold(
        self, last_trade_time: pd.Timestamp, current_time: pd.Timestamp
    ) -> float:
        """
        Calculate the current threshold based on time since last trade.

        Args:
            last_trade_time: Time of the last trade.
            current_time: Current time.

        Returns:
            Current threshold value.
        """
        # Compute days since last trade
        delta = (current_time - last_trade_time) / pd.Timedelta(days=1)

        if self.decay == "linear":
            frac = min(delta / self.dt_avg, 1.0)
            return self.tau_max - (self.tau_max - self.tau_min) * frac
        else:  # exponential
            return self.tau_min + (self.tau_max - self.tau_min) * np.exp(
                -self.lambda_ * delta
            )

    def should_trade(
        self,
        ticker: str,
        signal: float,
        current_time: pd.Timestamp,
        last_trade_time: pd.Timestamp,
    ) -> bool:
        """
        Determine if a trade should be executed based on the threshold mechanism.

        Args:
            ticker: Asset ticker symbol.
            signal: Trading signal value between -1 and 1.
            current_time: Current timestamp.
            last_trade_time: Timestamp of the last trade for this asset.

        Returns:
            Boolean indicating whether to execute the trade.
        """
        threshold = self._calculate_threshold(last_trade_time, current_time)
        return abs(signal) >= threshold

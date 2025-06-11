import pandas as pd
from dataclasses import dataclass

# ========================
# DOMAIN MODELS
# ========================


@dataclass
class Position:
    """Represents a position in a specific asset."""

    ticker: str
    size: float = 0.0
    last_trade_time: pd.Timestamp = pd.Timestamp("2000-01-01T12:00:00+00:00")

    def value(self, price: float) -> float:
        """Calculate the current value of the position."""
        return self.size * price

    def update(self, size_delta: float, timestamp: pd.Timestamp) -> None:
        """Update position size and last trade time."""
        self.size += size_delta
        self.last_trade_time = timestamp


@dataclass
class Order:
    """Represents a trade order."""

    ticker: str
    size: float
    price: float
    timestamp: pd.Timestamp
    slippage: float = 0.0
    commission: float = 0.0

    @property
    def notional(self) -> float:
        """Calculate the notional value of the order."""
        return self.price * self.size

    @property
    def total_cost(self) -> float:
        """Calculate the total cost including slippage and commission."""
        return self.notional + abs(self.notional) * self.slippage + self.commission

    @property
    def is_buy(self) -> bool:
        """Check if the order is a buy order."""
        return self.size > 0

    @property
    def is_sell(self) -> bool:
        """Check if the order is a sell order."""
        return self.size < 0


@dataclass
class Signal:
    """Represents a trading signal for an asset."""

    ticker: str
    value: float  # Between -1 and 1
    price: float
    timestamp: pd.Timestamp
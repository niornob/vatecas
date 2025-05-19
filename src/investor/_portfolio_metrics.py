import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from math import floor
from typing import (
    Dict,
    List,
    Mapping,
    Protocol,
)

from ._portfolio import Portfolio

# ========================
# PERFORMANCE TRACKING
# ========================


@dataclass
class PerformanceSnapshot:
    """Represents a snapshot of portfolio performance metrics."""

    timestamp: pd.Timestamp
    cash: float
    equity: float
    positions: Dict[str, float]  # Ticker to position size


class PerformanceTracker:
    """
    Tracks and analyzes portfolio performance over time.
    """

    def __init__(self):
        """Initialize the performance tracker."""
        self.snapshots: List[PerformanceSnapshot] = []

    def take_snapshot(
        self, portfolio: Portfolio, prices: Dict[str, float], timestamp: pd.Timestamp
    ) -> None:
        """
        Record the current state of the portfolio.

        Args:
            portfolio: Portfolio object.
            prices: Dictionary of current prices.
            timestamp: Current timestamp.
        """
        # Calculate position values
        positions = {
            ticker: position.size
            for ticker, position in portfolio.state.positions.items()
        }

        # Calculate total equity
        equity = portfolio.state.cash + sum(
            positions.get(ticker, 0) * prices.get(ticker, 0)
            for ticker in set(positions) | set(prices)
        )

        # Create and store snapshot
        snapshot = PerformanceSnapshot(
            timestamp=timestamp,
            cash=portfolio.state.cash,
            equity=equity,
            positions=positions.copy(),
        )
        self.snapshots.append(snapshot)

    def get_results(self) -> Dict:
        """
        Calculate performance metrics from recorded snapshots.

        Returns:
            Dictionary of performance metrics and time series.
        """
        # Handle empty portfolio (no recorded states)
        if not self.snapshots:
            empty_series = pd.Series(dtype=float)
            return {
                "equity_curve": empty_series,
                "cash_curve": empty_series,
                "positions_value_curve": empty_series,
                "returns": empty_series,
                "cumulative_return": 0.0,
                "sharpe_ratio": float("nan"),
                "max_drawdown": 0.0,
                "holdings": {},
            }

        # Extract timestamp index
        timestamps = pd.Index([s.timestamp for s in self.snapshots])

        # Build equity and cash series
        equity = pd.Series([s.equity for s in self.snapshots], index=timestamps)
        cash = pd.Series([s.cash for s in self.snapshots], index=timestamps)
        pos_val = equity - cash

        # Calculate returns
        returns = equity.pct_change().fillna(0)

        # Build holdings dataframe
        tickers = set()
        for snapshot in self.snapshots:
            tickers.update(snapshot.positions.keys())

        holdings = {}
        for ticker in tickers:
            values = [s.positions.get(ticker, 0.0) for s in self.snapshots]
            holdings[ticker] = pd.Series(values, index=timestamps)

        # Compute metrics with safety checks
        cumulative = (
            (equity.iloc[-1] / equity.iloc[0] - 1)
            if equity.iloc[0] != 0
            else float("nan")
        )

        sharpe = (
            (returns.mean() * (252**0.5) / returns.std())
            if returns.std()
            else float("nan")
        )

        max_dd = (
            ((equity - equity.cummax()) / equity.cummax()).min()
            if not equity.empty
            else 0.0
        )

        return {
            "equity_curve": equity,
            "cash_curve": cash,
            "positions_value_curve": pos_val,
            "returns": returns,
            "cumulative_return": cumulative,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd,
            "holdings": holdings,
        }
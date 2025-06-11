import pandas as pd
from typing import (
    Dict,
    Mapping,
)

from .atomic_types import Position, Order
from ._sizing import SizingModel, FixedFractionalSizing
from ._frequency_control import ThresholdDecayController
from ._portfolio import Portfolio, OrderGenerator
from ._portfolio_metrics import PerformanceTracker

# ========================
# PORTFOLIO MANAGER (FACADE)
# ========================
class PortfolioManager:
    """
    Facade class that coordinates portfolio, order generation, and performance tracking.
    Provides a simplified interface for the client code.
    """

    def __init__(
        self,
        initial_capital: float = 0.0,
        initial_positions: Dict[str, Position] = {},
        initial_time: pd.Timestamp = pd.Timestamp("2000-01-01T12:00:00+00:00"),
        max_position_size: float = float("inf"),
        commission: float = 0.0,
        slippage: float = 0.0,
        sizing_model: SizingModel = FixedFractionalSizing(fraction=0.9),
        trades_per_period: int = 1,
        period_length_days: int = 5,
        tau_max: float = 1.0,
        tau_min: float = 0.1,
    ):
        """
        Initialize the portfolio manager.

        Args:
            initial_capital: Initial cash amount.
            max_position_size: Maximum allowed position size in dollars.
            commission: Fixed commission per trade.
            slippage: Slippage as a fraction of trade value.
            sizing_fraction: Fraction of capital to use in position sizing.
            trades_per_period: Desired trades per period.
            period_length_days: Period length in trading days.
            tau_max: Maximum threshold immediately after a trade.
            tau_min: Minimum threshold after full decay.
            decay: Type of decay, either 'linear' or 'exponential'.
        """
        # Create components
        self.frequency_controller = ThresholdDecayController(
            trades_per_period=trades_per_period,
            period_length_days=period_length_days,
            tau_max=tau_max,
            tau_min=tau_min,
        )

        order_generator = OrderGenerator(
            sizing_model=sizing_model,
            frequency_controller=self.frequency_controller,
            commission=commission,
            slippage=slippage,
            max_position_size=max_position_size,
        )

        # Create core components
        self.portfolio = Portfolio(
            initial_capital=initial_capital,
            initial_positions=initial_positions,
            initial_time=initial_time,
            order_generator=order_generator,
        )

        self.performance_tracker = PerformanceTracker()

    def process_signals(
        self,
        signals: Dict[str, float],
        prices: Dict[str, float],
        timestamp: pd.Timestamp,
    ) -> Dict[str, Order]:
        """
        Process trading signals to generate and execute orders.
        Also records the portfolio state after execution.

        Args:
            signals: Dictionary of ticker to signal value (-1 to 1).
            prices: Dictionary of ticker to current price.
            timestamp: Current timestamp.

        Returns:
            Dictionary of executed orders.
        """
        # Process signals and execute orders
        orders = self.portfolio.process_signals(signals, prices, timestamp)

        # Record the portfolio state
        self.performance_tracker.take_snapshot(self.portfolio, prices, timestamp)

        return orders

    def get_results(self) -> Mapping:
        """
        Get portfolio performance results.

        Returns:
            Dictionary of performance metrics and time series.
        """
        results = self.performance_tracker.get_results()

        # Add additional data
        results["trades"] = [
            {
                "ticker": order.ticker,
                "timestamp": order.timestamp.isoformat(),
                "size": order.size,
                "price": order.price,
                "commission": order.commission,
            }
            for order in self.portfolio.order_history
        ]

        results["final_cash"] = self.portfolio.state.cash
        results["final_positions"] = {
            ticker: position.size
            for ticker, position in self.portfolio.state.positions.items()
        }

        return results

    @property
    def cash(self) -> float:
        """Get current cash amount."""
        return self.portfolio.state.cash

    @property
    def positions(self) -> Dict[str, float]:
        """Get current positions as dictionary of ticker to size."""
        return {
            ticker: position.size
            for ticker, position in self.portfolio.state.positions.items()
        }



# ========================
# USAGE EXAMPLE
# ========================


def create_portfolio(
    initial_capital: float = 10000.0,
    max_position_size: float = float("inf"),
    commission: float = 0.0,
    slippage: float = 0.0,
    sizing_model: SizingModel = FixedFractionalSizing(fraction=0.9),
    trades_per_period: int = 1,
    period_length_days: int = 5,
    decay: str = "linear",
    tau_max: float = 1.0,
    tau_min: float = 0.1,
) -> PortfolioManager:
    """
    Factory function to create a portfolio manager with the specified parameters.

    Args:
        initial_capital: Initial cash amount.
        max_position_size: Maximum allowed position size in dollars.
        commission: Fixed commission per trade.
        slippage: Slippage as a fraction of trade value.
        sizing_fraction: Fraction of capital to use in position sizing.
        trades_per_period: Desired trades per period.
        period_length_days: Period length in trading days.
        decay: Type of decay, either 'linear' or 'exponential'.
        tau_max: Maximum threshold immediately after a trade.
        tau_min: Minimum threshold after full decay.

    Returns:
        Configured PortfolioManager instance.
    """
    return PortfolioManager(
        initial_capital=initial_capital,
        max_position_size=max_position_size,
        commission=commission,
        slippage=slippage,
        sizing_model=sizing_model,
        trades_per_period=trades_per_period,
        period_length_days=period_length_days,
        tau_max=tau_max,
        tau_min=tau_min,
    )


# Example usage:
if __name__ == "__main__":
    # Create a portfolio
    portfolio = create_portfolio(
        initial_capital=100000,
        max_position_size=10000,
        commission=5.0,
        slippage=0.001,
        sizing_model=FixedFractionalSizing(fraction=0.9),
        trades_per_period=2,
        period_length_days=5,
        decay="linear",
    )

    manager = PortfolioManager()

    # Sample data
    timestamp = pd.Timestamp("2023-01-01", tz="UTC")
    signals = {"AAPL": 0.8, "MSFT": -0.5, "GOOGL": 0.2}
    prices = {"AAPL": 150.0, "MSFT": 250.0, "GOOGL": 2000.0}

    # Process signals
    orders = portfolio.process_signals(signals, prices, timestamp)

    # Advance time and process more signals
    timestamp = pd.Timestamp("2023-01-02", tz="UTC")
    signals = {"AAPL": 0.3, "MSFT": 0.1, "GOOGL": -0.7}
    prices = {"AAPL": 155.0, "MSFT": 245.0, "GOOGL": 1950.0}

    orders = portfolio.process_signals(signals, prices, timestamp)

    # Get results
    results = portfolio.get_results()
    print(f"Final cash: {results['final_cash']}")
    print(f"Cumulative return: {results['cumulative_return']:.2%}")
    print(f"Sharpe ratio: {results['sharpe_ratio']:.2f}")
    print(f"Max drawdown: {results['max_drawdown']:.2%}")

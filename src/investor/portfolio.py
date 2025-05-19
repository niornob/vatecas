"""
this file is under construction. do not use it.
it will be broken into several pieces.
"""

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


# ========================
# SIZING MODELS
# ========================


class SizingModel(Protocol):
    """Protocol defining position sizing logic interface."""

    def size_position(
        self,
        signal: float,
        price: float,
        available_cash: float,
        position_value: float,
        max_position_size: float,
    ) -> float:
        """
        Calculate position size based on signal strength and available resources.

        Args:
            signal: Value between -1 and 1. <0 is a sell, >0 is a buy.
            price: Current price of the asset.
            available_cash: Available cash for buying.
            position_value: Current position value for selling.
            max_position_size: Maximum allowed position size.

        Returns:
            Number of shares/contracts to trade.
        """
        ...


class FixedFractionalSizing:
    """
    Allocates a fixed fraction of capital scaled by signal strength,
    and converts the resulting dollar amount into number of shares.
    """

    def __init__(self, fraction: float):
        if not 0 <= fraction <= 1:
            raise ValueError("Fraction must be between 0 and 1")
        self.fraction = fraction

    def size_position(
        self,
        signal: float,
        price: float,
        available_cash: float,
        position_value: float,
        max_position_size: float,
    ) -> float:
        """
        Calculate position size based on signal strength and fixed fraction.

        Args:
            signal: Value between -1 and 1. <0 is a sell, >0 is a buy.
            price: Current price of the asset.
            available_cash: Available cash for buying.
            position_value: Current position value for selling.
            max_position_size: Maximum allowed position size.

        Returns:
            Number of shares/contracts to trade.
        """
        if abs(signal) > 1:
            raise ValueError("Signal must be between -1 and 1.")

        if price <= 0:
            return 0.0

        # Use current cash for buys, current position value for sells
        base = available_cash if signal > 0 else position_value
        dollar_amount = self.fraction * base * signal

        # Apply position size limit
        dollar_amount = min(abs(dollar_amount), max_position_size) * (
            1 if dollar_amount > 0 else -1
        )

        return round(dollar_amount / price)  # Assuming fractional shares are allowed


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


# ========================
# ORDER GENERATION
# ========================


class OrderGenerator:
    """
    Responsible for generating orders based on trading signals and portfolio constraints.
    """

    def __init__(
        self,
        sizing_model: SizingModel,
        frequency_controller: TradeFrequencyController,
        commission: float = 0.0,
        slippage: float = 0.0,
        max_position_size: float = float("inf"),
        min_trade_size: float = 1e-3,
    ):
        """
        Initialize the order generator.

        Args:
            sizing_model: Model used for position sizing.
            frequency_controller: Controller for trade frequency.
            commission: Fixed commission per trade.
            slippage: Slippage as a fraction of trade value.
            max_position_size: Maximum allowed position size in dollars.
            min_trade_size: Minimum trade size in number of shares.
        """
        self.sizing_model = sizing_model
        self.frequency_controller = frequency_controller
        self.commission = commission
        self.slippage = slippage
        self.max_position_size = max_position_size
        self.min_trade_size = min_trade_size

    def generate_orders(
        self,
        signals: Dict[str, Signal],
        positions: Dict[str, Position],
        cash: float,
        current_prices: Dict[str, float],
    ) -> Dict[str, Order]:
        """
        Generate orders based on signals and current portfolio state.

        Args:
            signals: Dictionary of ticker to Signal objects.
            positions: Dictionary of ticker to Position objects.
            cash: Available cash.
            current_prices: Dictionary of current prices for each ticker.

        Returns:
            Dictionary of ticker to Order objects.
        """
        available_cash = cash
        orders = {}

        # Pre-split into buys/sells for O(n) efficiency
        sells = [(tk, s) for tk, s in signals.items() if s.value < 0]
        buys = [(tk, s) for tk, s in signals.items() if s.value >= 0]

        # Sort sells (strongest first) and buys (strongest first)
        sorted_sells = sorted(sells, key=lambda x: x[1].value)
        sorted_buys = sorted(buys, key=lambda x: -x[1].value)
        sorted_signals = sorted_sells + sorted_buys

        for ticker, signal in sorted_signals:
            price = current_prices.get(ticker)
            if price is None or price <= 0:
                continue

            position = positions.get(ticker, Position(ticker=ticker))

            # Check if we should trade based on the frequency controller
            if not self.frequency_controller.should_trade(
                ticker, signal.value, signal.timestamp, position.last_trade_time
            ):
                continue

            # Calculate position value for this ticker
            position_value = position.size * price

            # Determine size using the sizing model
            size = self.sizing_model.size_position(
                signal=signal.value,
                price=price,
                available_cash=available_cash,
                position_value=position_value,
                max_position_size=self.max_position_size,
            )

            # Skip if size is below minimum
            if abs(size) < self.min_trade_size:
                continue

            # For sells, ensure we don't sell more than we have
            if size < 0:
                size = max(-position.size, size)
                if abs(size) < self.min_trade_size:
                    continue

            # Create a new order
            order = Order(
                ticker=ticker,
                size=size,
                price=price,
                timestamp=signal.timestamp,
                slippage=self.slippage,
                commission=self.commission,
            )

            # For buys, ensure we have enough cash
            if size > 0 and order.total_cost > available_cash:
                # Recalculate size to match available cash
                adjusted_size = max(
                    floor((available_cash - self.commission)
                    / (price * (1 + self.slippage) + 1e-5)),
                    0,
                )

                # Skip if the adjusted size is too small
                if adjusted_size < self.min_trade_size:
                    continue

                # Update the order with the new size
                order = Order(
                    ticker=ticker,
                    size=adjusted_size,
                    price=price,
                    timestamp=signal.timestamp,
                    slippage=self.slippage,
                    commission=self.commission,
                )

            # Update available cash and add order to result
            available_cash -= order.total_cost
            orders[ticker] = order

        return orders


# ========================
# PORTFOLIO CORE
# ========================


@dataclass
class PortfolioState:
    """Represents the current state of a portfolio."""

    cash: float
    positions: Dict[str, Position] = field(default_factory=dict)
    initial_cash: float = field(init=False)

    def __post_init__(self):
        self.initial_cash = self.cash

    def position_value(self, price_map: Dict[str, float]) -> float:
        """Calculate the total value of all positions."""
        return sum(
            pos.size * price_map.get(pos.ticker, 0.0) for pos in self.positions.values()
        )
    def equity(self, price_map: Dict[str, float]) -> float:
        """Calculate the total equity value (cash + positions)."""
        return self.cash + self.position_value(price_map)


class Portfolio:
    """
    Core portfolio class focused on state management.
    Handles positions, cash, and order execution.
    """

    def __init__(
        self,
        initial_capital: float,
        order_generator: OrderGenerator,
        initial_positions: Dict[str, Position],
        initial_time: pd.Timestamp = pd.Timestamp("2000-01-01T12:00:00+00:00")
    ):
        """
        Initialize the portfolio.

        Args:
            initial_capital: Initial cash amount.
            order_generator: Order generator component.
        """
        self.state = PortfolioState(cash=initial_capital, positions=initial_positions)
        self.order_generator = order_generator
        self.order_history: List[Order] = []
        self.initial_time = initial_time

    @property
    def cash(self) -> float:
        return self.state.cash
    
    @property
    def positions(self) -> Dict[str, Position]:
        return self.state.positions

    def process_signals(
        self,
        signals_dict: Dict[str, float],
        prices_dict: Dict[str, float],
        timestamp: pd.Timestamp,
    ) -> Dict[str, Order]:
        """
        Process trading signals to generate and execute orders.

        Args:
            signals_dict: Dictionary of ticker to signal value (-1 to 1).
            prices_dict: Dictionary of ticker to current price.
            timestamp: Current timestamp.

        Returns:
            Dictionary of executed orders.
        """
        # Convert raw signals to Signal objects
        signals = {
            ticker: Signal(
                ticker=ticker,
                value=value,
                price=prices_dict[ticker],
                timestamp=timestamp,
            )
            for ticker, value in signals_dict.items()
            if ticker in prices_dict and prices_dict[ticker] > 0
        }

        # Generate orders
        orders = self.order_generator.generate_orders(
            signals=signals,
            positions=self.state.positions,
            cash=self.state.cash,
            current_prices=prices_dict,
        )

        # Execute orders
        self.execute_orders(orders)

        return orders

    def execute_orders(self, orders: Dict[str, Order]) -> None:
        """
        Execute orders and update portfolio state.

        Args:
            orders: Dictionary of ticker to Order objects.
        """
        for ticker, order in orders.items():
            # Get or create position
            if ticker not in self.state.positions:
                self.state.positions[ticker] = Position(ticker=ticker)

            # Update position
            self.state.positions[ticker].update(order.size, order.timestamp)

            # Update cash
            self.state.cash -= order.total_cost

            # Record the order
            self.order_history.append(order)

            # Remove positions with zero size
            if abs(self.state.positions[ticker].size) < 1e-6:
                del self.state.positions[ticker]


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
        decay: str = "linear",
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
        frequency_controller = ThresholdDecayController(
            trades_per_period=trades_per_period,
            period_length_days=period_length_days,
            tau_max=tau_max,
            tau_min=tau_min,
            decay=decay,
        )

        order_generator = OrderGenerator(
            sizing_model=sizing_model,
            frequency_controller=frequency_controller,
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
        decay=decay,
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

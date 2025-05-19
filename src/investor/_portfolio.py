import pandas as pd
from dataclasses import dataclass, field
from math import floor
from typing import (
    Dict,
    List,
)

from .atomic_types import Position, Signal, Order
from ._sizing import SizingModel
from ._frequency_control import TradeFrequencyController

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
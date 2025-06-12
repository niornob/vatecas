from collections import OrderedDict
from dataclasses import dataclass, field
from typing import (
    Dict,
    List,
)

import pandas as pd

from portfolio_management._frequency_control import TradeFrequencyController
from portfolio_management._sizing import SizingModel
from portfolio_management.atomic_types import Order, Position, Signal

# ========================
# ORDER GENERATION
# ========================


class OrderGenerator:
    def __init__(
        self,
        sizing_model: SizingModel,
        frequency_controller: TradeFrequencyController,
        commission: float = 0.0,
        slippage: float = 0.0,
        max_position_fraction: float = 1.0,
    ):
        self.sizing_model = sizing_model
        self.commission = commission  # fraction of trade notional
        self.slippage = slippage  # fraction of price
        self.max_position_fraction = max_position_fraction
        self.frequency_controller = frequency_controller

    def generate_orders(
        self,
        signals: Dict[str, "Signal"],
        positions: Dict[str, "Position"],
        cash: float,
        current_prices: Dict[str, float],
    ) -> Dict[str, "Order"]:
        # 0) filter out signals that are below threshold
        signals_passed = {
            tk: sig
            for tk, sig in signals.items()
            if self.frequency_controller.should_trade(
                signal=sig.value,
                current_time=sig.timestamp,
                last_trade_time=(
                    positions[tk].last_trade_time
                    if tk in positions
                    else pd.Timestamp("2000-01-01T12:00:00+00:00")
                ),
            )
        }

        # 1) Determine raw order sizes (shares) from sizing model
        raw_orders = self.sizing_model.size_position(
            signals={t: sig.value for t, sig in signals_passed.items()},
            positions={t: pos.size for t, pos in positions.items()},
            prices=current_prices,
            cash=cash,
            max_position_fraction=self.max_position_fraction,
        )

        # 2) Separate buy and sell orders
        buy_orders = {t: qty for t, qty in raw_orders.items() if qty > 0}
        sell_orders = {t: qty for t, qty in raw_orders.items() if qty < 0}

        # 3) Create sell Order objects (we receive proceeds immediately)
        orders: Dict[str, Order] = {}
        available_cash = cash
        for ticker, qty in sell_orders.items():
            price = current_prices[ticker]
            slp_price = price * (1 - self.slippage)
            notional = slp_price * abs(qty)
            commission_amt = abs(notional) * self.commission
            order = Order(
                ticker=ticker,
                size=qty,
                price=slp_price,
                timestamp=signals[ticker].timestamp,
                slippage=self.slippage,
                commission=commission_amt,
            )
            orders[ticker] = order
            available_cash += notional - commission_amt

        # 4) Allocate buy orders in descending signal strength order
        #    to respect cash constraint and favor stronger signals
        # Compute candidate buy costs
        buy_list = []  # List of tuples (signal, ticker, target_qty)
        for ticker, qty in buy_orders.items():
            buy_list.append((signals[ticker].value, ticker, qty))
        buy_list.sort(reverse=True, key=lambda x: x[0])

        for _, ticker, target_qty in buy_list:
            price = current_prices[ticker]
            slp_price = price * (1 + self.slippage)
            # max shares we can afford at this price + commission
            # approximate per-share commission cost
            per_share_comm = self.commission * slp_price
            max_affordable_shares = max(
                available_cash / (slp_price + per_share_comm), 0
            )
            executed_qty = min(target_qty, max_affordable_shares)
            if executed_qty <= 0:
                continue
            notional = slp_price * executed_qty
            commission_amt = abs(notional) * self.commission
            order = Order(
                ticker=ticker,
                size=executed_qty,
                price=slp_price,
                timestamp=signals[ticker].timestamp,
                slippage=self.slippage,
                commission=commission_amt,
            )
            orders[ticker] = order
            # deduct cost
            available_cash -= notional + commission_amt

        return orders


# ========================
# PORTFOLIO CORE
# ========================


@dataclass
class PortfolioState:
    """Represents the current state of a portfolio."""

    timestamp: pd.Timestamp
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
        initial_time: pd.Timestamp = pd.Timestamp("1970-01-01T12:00:00+00:00"),
    ):
        """
        Initialize the portfolio.

        Args:
            initial_capital: Initial cash amount.
            order_generator: Order generator component.
        """
        self.state = PortfolioState(
            cash=initial_capital, positions=initial_positions, timestamp=initial_time
        )
        self.order_generator = order_generator
        self.order_history: List[Order] = []
        self.initial_time = initial_time

    @property
    def cash(self) -> float:
        return self.state.cash

    @property
    def positions(self) -> Dict[str, Position]:
        return self.state.positions

    @property
    def timestamp(self) -> pd.Timestamp:
        return self.state.timestamp

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
        timeordered_orders = OrderedDict(
            sorted(orders.items(), key=lambda item: item[1].timestamp)
        )
        for ticker, order in timeordered_orders.items():
            # if order is in the past (relative to state timestamp), then do nothing
            if order.timestamp < self.timestamp:
                continue

            # Get or create position
            if ticker not in self.state.positions:
                self.state.positions[ticker] = Position(ticker=ticker)

            # Update position
            self.state.positions[ticker].update(order.size, order.timestamp)

            # Update cash
            self.state.cash -= order.total_cost

            # Record the order
            self.order_history.append(order)

            # Update state timestamp
            self.state.timestamp = max(order.timestamp, self.state.timestamp)

            # Remove positions with zero size
            if abs(self.state.positions[ticker].size) < 1e-6:
                del self.state.positions[ticker]

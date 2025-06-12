from abc import ABC, abstractmethod
from typing import Dict, Optional


class SizingModel(ABC):
    def __init__(
        self,
        max_position_fraction: float = 1.0,
    ):
        self.max_position_fraction = max_position_fraction

    @abstractmethod
    def size_position(
        self,
        signals: Dict[str, float],
        positions: Dict[str, float],
        prices: Dict[str, float],
        cash: float,
        max_position_fraction: float,
    ) -> Dict[str, float]:
        """
        Compute order sizes for each asset, given signals, current positions,
        prices, available cash, and a constraint on maximum position size.

        :param signals: mapping from ticker to signal in [-1, 1]
        :param positions: mapping from ticker to current number of shares held
        :param prices: mapping from ticker to current price per share
        :param cash: available cash in the portfolio
        :param max_position_fraction: maximum fraction of total equity per position
        :return: mapping from ticker to signed number of shares to buy (>0) or sell (<0)
        """
        pass


class FractionalSizing(SizingModel):
    def size_position(
        self,
        signals: Dict[str, float],
        positions: Dict[str, float],
        prices: Dict[str, float],
        cash: float,
        max_position_fraction: Optional[float] = None,
    ) -> Dict[str, float]:
        max_position_fraction = max_position_fraction or self.max_position_fraction

        # Compute total equity: sum of position values + cash
        equity = cash + sum(
            positions.get(ticker, 0) * prices[ticker] for ticker in prices
        )

        # Filter long-only signals: only positive signals
        positive_signals = {t: s for t, s in signals.items() if s > 0}
        total_signal = sum(positive_signals.values())

        # Determine target dollar allocation per ticker
        target_dollars: Dict[str, float] = {}
        if total_signal > 0:
            for ticker, sig in positive_signals.items():
                # fractional weight = (signal / total_signal) * max_position_fraction
                weight = (sig / total_signal) * max_position_fraction
                target_dollars[ticker] = weight * equity
        # else: no long positions if no positive signals

        # Compute target shares and orders
        orders: Dict[str, float] = {}

        # Sell positions for assets not in target or with zero target
        for ticker, current_shares in positions.items():
            if target_dollars.get(ticker, 0) == 0 and current_shares > 0:
                # sell entire position
                orders[ticker] = -current_shares

        # Buy or adjust existing positions
        for ticker, dollar_amt in target_dollars.items():
            price = prices[ticker]
            target_shares = dollar_amt / price
            current_shares = positions.get(ticker, 0)
            orders[ticker] = target_shares - current_shares

        return orders

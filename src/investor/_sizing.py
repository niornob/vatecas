from typing import Protocol

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

        return round(dollar_amount / price)  # Assuming fractional shares are not allowed

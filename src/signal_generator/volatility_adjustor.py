import numpy as np
from typing import Tuple

class VolatilityAdjuster:
    """
    Handles volatility‐based signal adjustments.

    This class implements risk‐aware position sizing by scaling signals based on
    predicted volatility. High volatility assets get smaller position sizes to
    maintain consistent risk exposure across the portfolio.
    """

    def __init__(self):
        """
        (No arguments needed in this simple version, as the scaling logic is fixed.)
        """
        pass

    def adjust_for_volatility(
        self, signals: np.ndarray, asset_covariance: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Adjusts signals based on predicted volatility structure.

        Args:
            signals: Raw trading signal vector (array of shape [n_assets]).
            asset_covariance: Predicted covariance matrix from Oracle (n_assets × n_assets).

        Returns:
            Tuple of (adjusted_signals, vol_scaling_factors).
        """
        # 1. Extract individual volatilities
        asset_vols = np.sqrt(np.diag(asset_covariance))
        asset_vols = np.maximum(asset_vols, 1e-2)  # avoid division by zero

        # 2. Compute raw scaling = 1 / sqrt(volatility)
        vol_scaling = 1.0 / np.sqrt(asset_vols)

        # 3. Cap the scaling factors (so they don’t explode when vol is extremely low)
        vol_scaling = np.clip(vol_scaling, 0.1, 5.0)

        # 4. Apply scaling to raw signals
        adjusted_signals = signals * vol_scaling

        return adjusted_signals, vol_scaling

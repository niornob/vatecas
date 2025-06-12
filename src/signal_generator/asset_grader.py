"""
Defines the AssetGrader class. It grades each asset based on
predicted future price, past prices, and volatility.
Grades are between -1 and 1.
"""

from abc import abstractmethod
from collections import deque
from typing import Optional, Tuple

import numpy as np


class AssetGrader:
    """
    Handles volatility‐based signal adjustments.

    This class implements risk‐aware position sizing by scaling signals based on
    predicted volatility. High volatility assets get smaller position sizes to
    maintain consistent risk exposure across the portfolio.
    """

    def __init__(self, name: str = "UnnamedAssetGrade"):
        self.name = name

    @abstractmethod
    def grade_asset(
        self,
        prediction: np.ndarray,
        recent_prices: deque[np.ndarray],
        recent_predictions: deque[np.ndarray],
        asset_covariance: np.ndarray,
        target_volatility: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Grades assets based on prediction, historical price, and volatility.

        Args:
            prediction: array of predicted future prices.
            reference: array of past prices against which future prices will be compared.
            asset_covariance: Predicted covariance matrix from Oracle (n_assets × n_assets).
            target_volatility: A float, as a reference standard deviation for an asset price.

        Returns:
            Tuple of (adjusted_signals, vol_scaling_factors).
        """


class GraderPctReturnVolAdj(AssetGrader):
    def grade_asset(
        self,
        prediction: np.ndarray,
        recent_prices: deque[np.ndarray],
        recent_predictions: deque[np.ndarray],
        asset_covariance: np.ndarray,
        target_volatility: float = 2.0,
    ) -> Tuple[np.ndarray, np.ndarray]:

        # 0. Extract the reference, the oldest prediction in buffer.
        reference = recent_predictions[0]

        # 1. Compute percentage return
        signals = (prediction - reference) / reference

        # 2. Extract individual volatilities
        asset_vols = np.sqrt(np.diag(asset_covariance))
        asset_vols = np.maximum(asset_vols, 1e-2)  # avoid division by zero

        # 3. Compute raw scaling = 1 / sqrt(volatility)
        vol_scaling = target_volatility / asset_vols

        # 5. Apply scaling to raw signals
        adjusted_signals = np.clip(signals * vol_scaling, -1, 1)

        return adjusted_signals, vol_scaling


class GraderPctReturn(AssetGrader):
    def grade_asset(
        self,
        prediction: np.ndarray,
        recent_prices: deque[np.ndarray],
        recent_predictions: deque[np.ndarray],
        asset_covariance: Optional[np.ndarray] = None,
        target_volatility: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:

        # 0. Extract the reference, the oldest prediction in buffer.
        reference = recent_predictions[0]

        # 1. Compute percentage return
        signals = np.clip((prediction - reference) / reference, -1, 1)
        vol_scaling = np.ones_like(signals)

        return signals, vol_scaling

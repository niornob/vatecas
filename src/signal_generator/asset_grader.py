import numpy as np
from typing import Tuple
from abc import ABC, abstractmethod

class AssetGrader:
    """
    Handles volatility‐based signal adjustments.

    This class implements risk‐aware position sizing by scaling signals based on
    predicted volatility. High volatility assets get smaller position sizes to
    maintain consistent risk exposure across the portfolio.
    """

    def __init__(self):
        pass

    @abstractmethod
    def grade_asset(
        self,
        prediction: np.ndarray,
        reference: np.ndarray,
        asset_covariance: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Adjusts signals based on predicted volatility structure.

        Args:
            prediction: array of predicted future prices.
            reference: array of past prices against which future prices will be compared.
            asset_covariance: Predicted covariance matrix from Oracle (n_assets × n_assets).

        Returns:
            Tuple of (adjusted_signals, vol_scaling_factors).
        """
        pass
    

class PctReturnVolAdjusted(AssetGrader):
    def __init__(self, target_volatility: float = 2.0):
        """
        Args:
            target_volatility: reference volatility. more volatility will scale down signals and less will scale up.
        """
        self.target_vol = target_volatility
        pass

    def grade_asset(
        self, 
        prediction: np.ndarray,
        reference: np.ndarray,
        asset_covariance: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        # 1. Compute percentage return
        signals = (prediction - reference) / reference

        # 2. Extract individual volatilities
        asset_vols = np.sqrt(np.diag(asset_covariance))
        asset_vols = np.maximum(asset_vols, 1e-2)  # avoid division by zero

        # 3. Compute raw scaling = 1 / sqrt(volatility)
        vol_scaling = self.target_vol / asset_vols

        # 4. Cap the scaling factors (so they don’t explode when vol is extremely low)
        vol_scaling = np.clip(vol_scaling, 0.1, 5.0)

        # 5. Apply scaling to raw signals
        adjusted_signals = signals * vol_scaling

        return adjusted_signals, vol_scaling

class PctReturn(AssetGrader):
    def __init__(self, target_volatility: float = 2.0):
        """
        Args:
            target_volatility: reference volatility. more volatility will scale down signals and less will scale up.
        """
        self.target_vol = target_volatility
        pass

    def grade_asset(
        self, 
        prediction: np.ndarray,
        reference: np.ndarray,
        asset_covariance: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        # 1. Compute percentage return
        signals = (prediction - reference) / reference

        # 2. Extract individual volatilities
        asset_vols = np.sqrt(np.diag(asset_covariance))
        asset_vols = np.maximum(asset_vols, 1e-2)  # avoid division by zero

        # 3. Compute raw scaling = 1 / sqrt(volatility)
        vol_scaling = self.target_vol / asset_vols

        # 4. Cap the scaling factors (so they don’t explode when vol is extremely low)
        vol_scaling = np.clip(vol_scaling, 0.1, 5.0)

        return signals, vol_scaling
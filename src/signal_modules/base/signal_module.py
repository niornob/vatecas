"""
Signal generation module for converting Oracle predictions into actionable trading signals.

This module transforms raw predictions from Oracle instances into normalized trading signals
in the range [-1, +1], incorporating volatility-based position sizing and smoothing mechanisms.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
from abc import ABC, abstractmethod
from scipy.stats import rankdata
from scipy.special import expit

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from regression.base.oracle import Oracle
from regression.base.prediction_result import PredictionResult


@dataclass
class SignalResult:
    """
    Container for signal generation results.

    This class encapsulates all outputs from the signal generation process,
    providing both the final signals and intermediate calculations for analysis.
    """

    signals: np.ndarray  # Final trading signals [-1, +1] for each ticker
    raw_predictions: np.ndarray  # Original predictions from Oracle
    smoothed_predictions: np.ndarray  # Predictions after smoothing
    volatility_adjustments: np.ndarray  # Volatility-based scaling factors
    confidence_scores: np.ndarray  # Signal confidence based on various factors
    tickers: List[str]  # Ticker symbols in order


class SmoothingStrategy(ABC):
    """
    Abstract base class for prediction smoothing strategies.

    This allows for different smoothing approaches while maintaining a consistent interface.
    Moving averages are just one option - you could implement exponential smoothing,
    Kalman filtering, or other sophisticated approaches.
    """

    @abstractmethod
    def smooth(self, predictions: np.ndarray, window: int) -> np.ndarray:
        """Apply smoothing to prediction series."""
        pass


class MovingAverageSmoothing(SmoothingStrategy):
    """
    Simple moving average smoothing strategy.

    This implements the 2-day moving average approach you discovered works well
    with your Oracle's predictions. The strategy maintains a buffer of past
    predictions to compute rolling averages.
    """

    def __init__(self):
        pass

    def smooth(self, predictions: np.ndarray, window: int) -> np.ndarray:
        """
        Apply moving average smoothing to predictions.

        Args:
            predictions: Current predictions for all tickers.
            window: Number of periods for moving average

        Returns:
            Smoothed predictions array
        """

        assert (
            len(predictions) >= window
        ), f"not enough predictions ({len(predictions)}) to apply {window}-days averaging."

        truncated_predictions = pd.DataFrame(predictions).iloc[-window:]

        return np.array(list(truncated_predictions.mean().values))


class VolatilityAdjuster:
    """
    Handles volatility-based signal adjustments.

    This class implements risk-aware position sizing by scaling signals based on
    predicted volatility. High volatility assets get smaller position sizes to
    maintain consistent risk exposure across the portfolio.
    """

    def __init__(self, vol_target: float = 0.15, min_vol_threshold: float = 0.01):
        """
        Initialize volatility adjuster.

        Args:
            vol_target: Target volatility for position sizing
            min_vol_threshold: Minimum volatility to prevent division by zero
        """
        self.vol_target = vol_target
        self.min_vol_threshold = min_vol_threshold

    def adjust_for_volatility(
        self, signals: np.ndarray, asset_covariance: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Adjust signals based on predicted volatility structure.

        This method implements inverse volatility scaling - assets with higher
        predicted volatility receive proportionally smaller position sizes.
        This helps maintain consistent risk contribution across assets.

        Args:
            signals: Raw trading signals
            asset_covariance: Predicted covariance matrix from Oracle

        Returns:
            Tuple of (adjusted_signals, volatility_scaling_factors)
        """
        # Extract individual asset volatilities from covariance diagonal
        asset_volatilities = np.sqrt(np.diag(asset_covariance))

        # Prevent division by very small numbers
        asset_volatilities = np.maximum(asset_volatilities, self.min_vol_threshold)

        # Calculate volatility-based scaling factors
        # Higher volatility -> smaller scaling factor -> smaller position
        vol_scaling = self.vol_target / asset_volatilities

        # Normalize scaling factors to prevent extreme position sizes
        vol_scaling = np.clip(vol_scaling, 0.1, 2.0)

        # Apply volatility adjustment to signals
        adjusted_signals = signals * vol_scaling

        return adjusted_signals, vol_scaling


class SignalModule:
    """
    Main signal generation module that orchestrates the conversion of Oracle predictions
    into actionable trading signals.

    This class brings together all the components: Oracle predictions, smoothing,
    volatility adjustment, and signal normalization to produce final trading signals.
    """

    def __init__(
        self,
        oracle: Oracle,
        smoothing_window: int = 2,
        smoothing_strategy: Optional[SmoothingStrategy] = None,
        vol_adjuster: Optional[VolatilityAdjuster] = None,
        signal_strength_threshold: float = 0.1,
    ):
        """
        Initialize the SignalModule.

        Args:
            oracle: The Oracle instance for generating predictions
            smoothing_window: Window length for prediction smoothing
            smoothing_strategy: Strategy for smoothing predictions
            vol_adjuster: Handler for volatility-based adjustments
            signal_strength_threshold: Minimum threshold for signal generation
        """
        self.oracle = oracle
        self.smoothing_window = smoothing_window
        self.smoothing_strategy = smoothing_strategy or MovingAverageSmoothing()
        self.vol_adjuster = vol_adjuster or VolatilityAdjuster()
        self.signal_strength_threshold = signal_strength_threshold

    def generate_signals(
        self, data: Dict[str, pd.Series], past_predictions: Dict[str, pd.Series]
    ) -> SignalResult:
        """
        Generate trading signals from market data.

        This is the main orchestration method that coordinates all signal generation steps:
        1. Get predictions from Oracle (mean + covariance)
        2. Apply smoothing to predictions
        3. Convert to directional signals
        4. Apply volatility-based position sizing
        5. Normalize to [-1, +1] range

        Args:
            data: Dictionary mapping ticker symbols to price series

        Returns:
            SignalResult containing final signals and intermediate calculations
        """
        # Get predictions from Oracle
        prediction_result = self.oracle.predict(data)
        raw_predictions = prediction_result.predictions
        asset_covariance = prediction_result.asset_covariance
        market_volatility = prediction_result.pc1_variance

        # Apply smoothing to predictions
        preds_to_smooth = pd.DataFrame(past_predictions).values
        preds_to_smooth = np.concat([preds_to_smooth, [raw_predictions]])
        smoothed_predictions = self.smoothing_strategy.smooth(
            raw_predictions, min(len(preds_to_smooth), self.smoothing_window)
        )

        # Apply volatility-based position sizing
        vol_adjusted_signals, vol_scaling = self.vol_adjuster.adjust_for_volatility(
            smoothed_predictions, asset_covariance
        )

        # Normalize signals to [-1, +1] range
        final_signals = self._normalize_signals(vol_adjusted_signals, market_volatility)

        # Calculate confidence scores
        confidence_scores = self._calculate_confidence(
            raw_predictions, smoothed_predictions, vol_scaling, asset_covariance
        )

        return SignalResult(
            signals=final_signals,
            raw_predictions=raw_predictions,
            smoothed_predictions=smoothed_predictions,
            volatility_adjustments=vol_scaling,
            confidence_scores=confidence_scores,
            tickers=list(data.keys()),
        )

    def _normalize_signals(self, signals: np.ndarray, market_volatility: float = 0) -> np.ndarray:
        """
        Normalize signals to [-1, +1] range using rank-based normalization.

        This approach ensures signals are distributed across the full range
        while preserving relative ordering. It's more robust than simple
        min-max scaling which can be sensitive to outliers.
        """
        assert len(signals) > 0, "no signal found."

        n = max(np.abs(signals))

        return 2 * expit(signals * 4 / n ) - 1


    def _calculate_confidence(
        self,
        raw_predictions: np.ndarray,
        smoothed_predictions: np.ndarray,
        vol_scaling: np.ndarray,
        covariance: np.ndarray,
    ) -> np.ndarray:
        """
        Calculate confidence scores for signals based on various factors.

        Higher confidence indicates more reliable signals. Factors considered:
        - Consistency between raw and smoothed predictions
        - Volatility levels (lower vol -> higher confidence)
        - Prediction magnitude (stronger predictions -> higher confidence)
        """
        # Consistency between raw and smoothed predictions
        consistency = 1.0 - np.abs(raw_predictions - smoothed_predictions) / (
            np.abs(raw_predictions) + 1e-6
        )
        consistency = np.clip(consistency, 0, 1)

        # Volatility-based confidence (lower vol -> higher confidence)
        vol_confidence = 1.0 / (1.0 + np.sqrt(np.diag(covariance)))

        # Prediction strength confidence
        strength_confidence = np.abs(smoothed_predictions) / (
            np.max(np.abs(smoothed_predictions)) + 1e-6
        )

        # Combined confidence score
        confidence = (consistency + vol_confidence + strength_confidence) / 3.0

        return np.clip(confidence, 0, 1)

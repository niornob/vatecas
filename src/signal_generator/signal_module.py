"""
Signal generation module for converting Oracle predictions into actionable trading signals.

This module transforms raw predictions from Oracle instances into normalized trading signals
in the range [-1, +1], incorporating volatility-based position sizing and smoothing mechanisms.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
from scipy.stats import rankdata
import warnings

warnings.filterwarnings("ignore")

from signal_generator.regression.base.oracle import Oracle
from signal_generator.asset_grader import AssetGrader, PctReturn


@dataclass
class SignalResult:
    """
    Container for signal generation results.

    This class encapsulates all outputs from the signal generation process,
    providing both the final signals and intermediate calculations for analysis.
    """

    signals: np.ndarray  # Final trading signals [-1, +1] for each ticker
    raw_predictions: np.ndarray  # Original predictions from Oracle
    volatility_adjustments: np.ndarray  # Volatility-based scaling factors
    tickers: List[str]  # Ticker symbols in order


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
        smoothing_window: int = 1,
        asset_grader: Optional[AssetGrader] = None,
        signal_strength_threshold: float = 0.01,
        market_vol_target: float = 2,
        name: str = "Unnamed SignalModule",
        version: str = "-.-",
    ):
        """
        Initialize the SignalModule.

        Args:
            oracle: The Oracle instance for generating predictions
            smoothing_window: Window length for prediction smoothing
            asset_grader: grades predicted future prices based on reference and volatility
            signal_strength_threshold: Minimum threshold for signal generation
            market_vol_target: reference standard deviation of percentage changes of asset prices.
            lower than the reference will leave the signals unaffected.
            higher than the reference will scale down the signals proportionately.
        """
        self.oracle = oracle
        assert (
            smoothing_window >= 1
        ), "smoothing window must be at least 1 (1 => no smoothing)."
        self.smoothing_window = smoothing_window
        self.asset_grader = asset_grader or PctReturn(target_volatility=market_vol_target)
        self.signal_strength_threshold = signal_strength_threshold
        self.market_vol_target = market_vol_target
        self.name = name
        self.version = version

    def generate_signals(
            self, 
            data: Dict[str, pd.Series],
            past_predictions: Optional[np.ndarray] = None,
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
            past_predictions: past prediction to compare future prediction agains

        Returns:
            SignalResult containing final signals and intermediate calculations
        """
        # Get predictions from Oracle
        prediction_result = self.oracle.predict(data)
        raw_predictions = prediction_result.predictions
        asset_covariance = prediction_result.asset_covariance
        market_volatility = np.sqrt(prediction_result.pc1_variance)

        #past_prices = np.array([s.iloc[-self.smoothing_window] for s in data.values()])
        if past_predictions is None:
            data_lagged = {
                tk: s.iloc[:-self.smoothing_window]
                for tk, s in data.items()
            }
            past_predictions = self.oracle.predict(data_lagged).predictions

        # Apply volatility-based position sizing
        raw_signals, vol_scaling = self.asset_grader.grade_asset(
            prediction=raw_predictions,
            reference=past_predictions, 
            asset_covariance=asset_covariance
        )

        # Normalize signals to [-1, +1] range
        final_signals = self._normalize_signals(raw_signals, market_volatility)

        return SignalResult(
            signals=final_signals,
            raw_predictions=raw_predictions,
            volatility_adjustments=vol_scaling,
            tickers=list(data.keys()),
        )

    def _normalize_signals(
        self, signals: np.ndarray, market_volatility: float = 0
    ) -> np.ndarray:
        """
        Normalize signals to [-1, +1] range using sigmoid.

        This approach ensures signals are distributed across the full range
        while preserving relative ordering.
        """
        assert len(signals) > 0, "no signal found."

        """

        n = max(np.abs(signals))

        assert n > 0, "all signals are zero. this should not happen."

        vol_scaling = np.clip(self.market_vol_target / market_volatility, 0, 1)
        # print(vol_scaling)
        raw_signals = (
            (1 - 1e-5) * (2 * expit(signals / n) - 1) / (expit(1.0) - expit(-1.0))
        )

        return vol_scaling * raw_signals\
        """

        rank = rankdata(signals)
        rank = rank / len(rank)
        rank = 2 * rank - 1

        return rank
    



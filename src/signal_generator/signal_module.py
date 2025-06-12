"""
Signal generation module for converting Oracle predictions into actionable trading signals.

This module transforms raw predictions from Oracle instances into normalized trading signals
in the range [-1, +1], incorporating volatility-based position sizing and smoothing mechanisms.
A signal module will consist of the following components:
1. Oracle: from input data predict price, and volatility (standard deviation as pct of price)
2. AssetGrader: assigns a value to each asset between -1 and 1 based on past and
    future price/prediction and volatility. This is the raw signal.
3. SignalNormalizer: Modify/Stretch raw signals over the range [-1,1],
    use overall market-volatility if needed.
"""

import warnings
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from signal_generator.asset_grader import AssetGrader, GraderPctReturn
from signal_generator.regression.base.oracle import Oracle
from signal_generator.signal_normalizer import NormalizerRank, SignalNormalizer

warnings.filterwarnings("ignore")


@dataclass
class SignalResult:
    """
    Container for signal generation results.

    This class encapsulates all outputs from the signal generation process,
    providing both the final signals and intermediate calculations for analysis.
    """

    signals: np.ndarray  # Trading signals [-1, +1] for each ticker
    raw_predictions: np.ndarray  # Original predictions from Oracle
    volatility_adjustments: np.ndarray  # Volatility-based scaling factors
    tickers: List[str]  # Ticker symbols in order


class SignalModule:
    def __init__(
        self,
        oracle: Oracle,
        smoothing_window: int = 1,
        asset_grader: Optional[AssetGrader] = None,
        normalizer: Optional["SignalNormalizer"] = None,
        market_vol_target: float = 2,
        name: str = "UnnamedSignalModule",
    ):
        """
        Initialize the SignalModule.

        Args:
            oracle: The Oracle instance for generating predictions
            smoothing_window: Window length for coputing return (%)
            asset_grader: grades assets based on returns and volatility
            market_vol_target: reference volatility, single value for the whole market.
            lower than the reference will leave the signals unaffected.
            higher than the reference will scale down the signals proportionately.
        """
        self.oracle = oracle
        assert (
            smoothing_window >= 1
        ), "smoothing window must be at least 1 (1 => no smoothing)."
        self.smoothing_window = smoothing_window
        self.asset_grader = asset_grader or GraderPctReturn()
        self.normalizer = normalizer or NormalizerRank(name="SignalRank")
        self.market_vol_target = market_vol_target
        self.name = name

    def generate_signals(
        self,
        data: Dict[str, pd.Series],
        recent_predictions: Optional[deque[np.ndarray]] = None,
    ) -> SignalResult:
        """
        Generate trading signals from market data.

        This is the main orchestration method that coordinates all signal generation steps:
        1. Get predictions from Oracle (mean + covariance)
        2. Convert to directional signals
        3. Adjust signals based on volatility if needed
        4. Normalize to [-1, +1] range

        Args:
            data: Dictionary mapping ticker symbols to price series
            recent_predictions: past prediction to compare future prediction agains

        Returns:
            SignalResult containing final signals and intermediate calculations
        """
        # Get predictions from Oracle
        prediction_result = self.oracle.predict(data)
        raw_predictions = prediction_result.predictions
        asset_covariance = prediction_result.asset_covariance
        market_volatility = np.sqrt(prediction_result.pc1_variance)

        # If past predictions are not provided, build historical
        # data upto the appropriate past date to compute past predictions from
        recent_predictions = recent_predictions or deque(
            np.array([]), maxlen=self.smoothing_window
        )
        if len(recent_predictions) < self.smoothing_window:
            for i in range(1, self.smoothing_window + 1):
                data_lagged = {tk: s.iloc[:-i] for tk, s in data.items()}
                recent_predictions.appendleft(
                    self.oracle.predict(data_lagged).predictions
                )

        recent_prices = deque(
            pd.DataFrame(data).iloc[-self.smoothing_window :].values,
            maxlen=self.smoothing_window,
        )

        # Grade assets based on predictions and volatility
        raw_signals, vol_scaling = self.asset_grader.grade_asset(
            prediction=raw_predictions,
            recent_prices=recent_prices,
            recent_predictions=recent_predictions,
            asset_covariance=asset_covariance,
            target_volatility=self.market_vol_target,
        )

        # Normalize signals to [-1, +1] range
        final_signals = self.normalizer.normalize_signals(
            raw_signals=raw_signals,
            market_volatility=market_volatility,
            target_volatility=self.market_vol_target,
        )

        return SignalResult(
            signals=final_signals,
            raw_predictions=raw_predictions,
            volatility_adjustments=vol_scaling,
            tickers=list(data.keys()),
        )

from typing import Optional, Dict, List, Tuple, cast, Literal
import pandas as pd
import numpy as np

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parents[2]))
from utils.denoise import wavelet_denoise
from regression.base.oracle import Oracle

class StackedOracle(Oracle):
    """
    Enhanced ensemble oracle with variance modeling capabilities.

    This sophisticated implementation stacks multiple oracles while maintaining
    coherent variance forecasts across the ensemble.
    """

    def __init__(self, oracles: List[Oracle], windows: List[int], weights: List[float]):
        """Initialize StackedOracle with variance modeling capabilities."""
        if len(set([len(oracles), len(windows), len(weights)])) != 1:
            raise ValueError(
                f"Number of oracles ({len(oracles)}) must equal number of windows ({len(windows)}) and weights ({len(weights)})"
            )

        if not oracles:
            raise ValueError("At least one oracle must be provided")

        if any(w < 1 for w in windows):
            raise ValueError("All window sizes must be positive")

        if any(w < 0 for w in weights):
            raise ValueError("Weights must be non-negative.")

        super().__init__(name=f"StackedOracle_{len(oracles)}", version="2.0")

        self.oracles = oracles
        self.windows = windows
        self.weights = weights

        # Enhanced state tracking for variance
        self.residuals: List[Dict[str, pd.Series]] = []
        self.approx_residuals: List[Dict[str, pd.Series]] = []
        self.variance_contributions: List[np.ndarray] = []

    def _predict_mean(self, data: Dict[str, pd.Series]) -> np.ndarray:
        """
        Stacked mean prediction using residual decomposition.

        This method implements the sequential residual prediction approach
        while maintaining compatibility with the enhanced variance framework.
        """
        if not data:
            raise ValueError("Input data cannot be empty")

        tickers = list(data.keys())
        series_lengths = [len(series) for series in data.values()]
        if len(set(series_lengths)) > 1:
            raise ValueError("All input series must have the same length")

        # Reset state for new prediction
        self.residuals = []
        self.approx_residuals = []
        self.variance_contributions = []
        predictions: List[np.ndarray] = []

        # Initialize with original data
        self.residuals.append({tk: s.reset_index(drop=True) for tk, s in data.items()})

        # Sequential oracle processing for mean predictions
        for oracle_idx, (oracle, window) in enumerate(zip(self.oracles, self.windows)):
            try:
                # Get mean predictions only from component oracle
                pred_dict, _, _ = oracle.regress(
                    self.residuals[-1], window=window, extrapolate=True
                )

                # Extract extrapolated values
                extrapolated_values = np.array(
                    [pred_dict[tk].iloc[-1] for tk in tickers]
                )
                predictions.append(extrapolated_values)

                # Remove extrapolated values for residual calculation
                pred_dict = {tk: pred_dict[tk].iloc[:-1] for tk in tickers}
                self.approx_residuals.append(pred_dict)

                # Calculate denoised residuals for next iteration
                new_residuals = {}
                for ticker in tickers:
                    raw_residual = self.residuals[-1][ticker] - pred_dict[ticker]
                    new_residuals[ticker] = wavelet_denoise(
                        raw_residual,
                        wavelet="db20",
                        threshold_method="soft",
                        level=None,
                    )

                self.residuals.append(new_residuals)

            except Exception as e:
                raise RuntimeError(f"Oracle {oracle_idx} ({oracle.name}) failed: {e}")

        # Combine weighted predictions
        try:
            net_prediction = np.sum(
                [w * pred for w, pred in zip(self.weights, predictions)], axis=0
            )
        except Exception as e:
            raise RuntimeError(f"Failed to combine predictions: {e}")

        return net_prediction

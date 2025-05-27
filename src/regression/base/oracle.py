from abc import ABC, abstractmethod
from typing import Optional, Dict, List, Tuple, cast, Literal
import pandas as pd
import numpy as np
from arch import arch_model
from arch.univariate.base import ARCHModelResult
from sklearn.decomposition import PCA
import warnings

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from regression.base.prediction_result import PredictionResult
from regression.volatility.GARCH import GARCH


class Oracle(ABC):
    """
    Abstract base class for time series prediction models (oracles) with variance modeling.

    Enhanced oracle framework that predicts both asset prices and their covariance structure.
    This dual approach allows for comprehensive risk assessment alongside return forecasting.
    """

    def __init__(
        self,
        name: str = "generic_oracle_name",
        version: str = "0.0",
        params: Optional[Dict[str, object]] = None,
    ):
        """
        Initialize an Oracle instance with GARCH capabilities.

        Args:
            name: Human-readable name for the oracle
            version: Version identifier for the oracle
            params: Dictionary of parameters specific to the oracle implementation
            garch_params: Parameters for GARCH modeling (p, q, distribution, etc.)
        """
        self.name = name or self.__class__.__name__
        self.version = version
        self.params = params or {}

    @abstractmethod
    def _predict_mean(self, data: Dict[str, pd.Series]) -> np.ndarray:
        """
        Core mean prediction method to be implemented by subclasses.

        Args:
            data: Dictionary mapping ticker symbols to historical price series

        Returns:
            Array of mean predictions for each ticker in the same order as data.keys()
        """
        pass

    def _predict_variance(
        self,
        data: Dict[str, pd.Series],
        volatility_params: Optional[Dict[str, object]] = None,
    ) -> Tuple[np.ndarray, float, np.ndarray]:
        """
        Predict variance structure using GARCH models and PCA.

        This method implements a sophisticated variance forecasting approach:
        1. Convert price series to returns
        2. Fit individual GARCH models for each asset
        3. Perform PCA to identify the dominant market factor
        4. Forecast next-period covariance matrix

        Args:
            data: Dictionary mapping ticker symbols to historical price series

        Returns:
            Tuple of (asset_covariance_matrix, pc1_variance, pc1_loadings)
        """
        return GARCH(data=data, params=volatility_params)

    def _predict(self, data: Dict[str, pd.Series]) -> PredictionResult:
        """
        Combined prediction method that forecasts both mean and variance.

        This is the core method that orchestrates both mean and variance predictions,
        returning a comprehensive PredictionResult object.
        """
        # Get mean predictions from subclass implementation
        mean_predictions = self._predict_mean(data)

        # Get variance predictions using GARCH and PCA
        garch_params = {
            "p": 1,
            "q": 1,
            "distribution": "normal",
            "rescale": True,
            "mean": "zero",
        }
        asset_covariance, pc1_variance, pc1_loadings = self._predict_variance(
            data=data, volatility_params=garch_params
        )

        return PredictionResult(
            predictions=mean_predictions,
            asset_covariance=asset_covariance,
            pc1_variance=pc1_variance,
            pc1_loadings=pc1_loadings,
        )

    def regress(
        self, data: dict[str, pd.Series], window: int, extrapolate: bool = False
    ) -> Tuple[dict[str, pd.Series], dict[str, pd.Series], pd.Series]:
        """
        Perform rolling window regression with variance modeling.

        Enhanced regression that provides both price predictions and confidence intervals
        based on forecasted volatility.

        Args:
            data: Dictionary mapping ticker symbols to historical price series
            window: Number of historical points to use for each prediction
            extrapolate: If True, generate one additional prediction beyond the data

        Returns:
            Tuple of (predictions_dict, volatility_bands_dict, market_factor_series)
            - predictions_dict: Mean price predictions for each asset
            - volatility_bands_dict: Standard deviation bands for each asset
            - market_factor_series: Time series of market factor (PC1) volatility
        """
        # Input validation
        if not data:
            raise ValueError("Input data cannot be empty")

        # Convert to DataFrame for easier manipulation
        try:
            data_df = pd.DataFrame(data)
        except ValueError as e:
            raise ValueError(f"Failed to convert input data to DataFrame: {e}")

        data_len = len(data_df.index)
        if data_len == 0:
            raise ValueError("Input data series are empty")

        tickers = list(data_df.columns)
        pred_len = data_len + (1 if extrapolate else 0)

        # Initialize prediction containers
        pred = pd.DataFrame(
            np.zeros((pred_len, len(tickers))), columns=tickers, index=range(pred_len)
        )
        vol_bands = pd.DataFrame(
            np.zeros((pred_len, len(tickers))), columns=tickers, index=range(pred_len)
        )
        market_factor = pd.Series(np.zeros(pred_len), index=range(pred_len))

        if not extrapolate:
            pred.index = data_df.index
            vol_bands.index = data_df.index
            market_factor.index = data_df.index

        # Set initial conditions
        pred.iloc[0] = data_df.iloc[0]
        vol_bands.iloc[0] = 0  # No uncertainty for initial observation
        market_factor.iloc[0] = 0

        # Rolling window prediction with variance
        for i in range(1, pred_len):
            # Extract historical window
            window_start = max(0, i - window)
            sliced_df = data_df.iloc[window_start:i]

            # Convert to dictionary format expected by _predict
            history = {ticker: sliced_df[ticker] for ticker in tickers}

            try:
                # Get comprehensive prediction result
                result = self._predict(data=history)

                if len(result.predictions) != len(tickers):
                    raise ValueError(
                        f"Prediction length {len(result.predictions)} doesn't match ticker count {len(tickers)}"
                    )

                # Store mean predictions
                pred.iloc[i] = result.predictions

                # Store volatility bands (standard deviations)
                vol_bands.iloc[i] = result.asset_volatilities

                # Store market factor volatility
                market_factor.iloc[i] = np.sqrt(result.pc1_variance)

            except Exception as e:
                raise RuntimeError(f"Prediction failed at step {i}: {e}")

        return (
            {ticker: pred[ticker] for ticker in tickers},
            {ticker: vol_bands[ticker] for ticker in tickers},
            market_factor,
        )

    def residuals(
        self, data: Dict[str, pd.Series], window: int
    ) -> Dict[str, pd.Series]:
        """Calculate residuals considering only mean predictions (variance residuals would be complex)."""
        predictions, _, _ = self.regress(data, window=window, extrapolate=False)
        return {ticker: data[ticker] - predictions[ticker] for ticker in data.keys()}

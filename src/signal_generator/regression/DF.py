from typing import Optional, Dict, cast
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.dynamic_factor import (
    DynamicFactor,
    DynamicFactorResultsWrapper,
)
import warnings

from signal_generator.regression.base.oracle import Oracle  # adjust this import path as needed


class DFOracle(Oracle):
    """
    Oracle implementation using a multivariate state‐space (DynamicFactor) model
    on asset returns to capture latent trends and common movements.

    - k_factors:    number of latent factors
    - factor_order: autoregressive order (p) of each latent factor
    """

    def __init__(
        self,
        k_factors: int,
        factor_order: int,
        name: str = "StateSpacePredictiveOracle",
        version: str = "1.0",
        params: Optional[Dict[str, object]] = None,
    ):
        """
        Args:
            k_factors:    How many latent factors to extract from the return series.
            factor_order: AR(p) order for each latent factor (must be ≥ 1).
            name:         Optional human‐readable name.
            version:      Optional version string.
            params:       Optional dict of extra parameters.
        """
        super().__init__(name=name, version=version, params=params)
        if not isinstance(k_factors, int) or k_factors < 1:
            raise ValueError("`k_factors` must be a positive integer.")
        if not isinstance(factor_order, int) or factor_order < 1:
            raise ValueError("`factor_order` must be a positive integer.")
        self.k_factors = k_factors
        self.factor_order = factor_order

    def _predict_mean(self, data: Dict[str, pd.Series]) -> np.ndarray:
        """
        Overrides Oracle._predict_mean by fitting a DynamicFactor state‐space model on returns.

        Steps:
          1. Build a DataFrame of prices (shape L×n) from `data`.
          2. Compute returns_df = price_df.pct_change().dropna() → shape (L−1)×n.
          3. Reset returns_df.index to a RangeIndex to avoid index‐warnings.
          4. Fit DynamicFactor(returns_df, k_factors, factor_order, enforce_stationarity=False),
             specifying a higher maxiter to improve convergence.
          5. Use get_forecast(steps=1) on the casted results to obtain a one‐step‐ahead return forecast.
          6. Convert that forecasted return into a price forecast:
               last_prices = price_df.iloc[-1].values
               predicted_prices = last_prices * (1 + predicted_returns)
          7. Return predicted_prices as a 1D np.ndarray (length n, same ticker order).
        """
        # 1. Convert `data` into a price DataFrame (L × n)
        try:
            price_df = pd.DataFrame(data)
        except Exception as e:
            raise ValueError(f"Could not convert input data to DataFrame: {e}")

        if price_df.isnull().values.any():
            raise ValueError("Input price series contains NaN values. Please clean your data first.")

        L, n = price_df.shape
        if L < 2:
            raise ValueError("Need at least two price observations per series to compute returns.")

        # 2. Compute returns: (L–1) × n
        returns_df = price_df.pct_change().dropna(axis=0, how="all")
        total_return_rows = returns_df.shape[0]
        if total_return_rows < max(self.k_factors, self.factor_order):
            raise ValueError(
                f"Not enough return observations ({total_return_rows}) to fit "
                f"k_factors={self.k_factors}, factor_order={self.factor_order}."
            )

        # 3. Reset index to a simple RangeIndex to suppress statsmodels warnings
        returns_df = returns_df.reset_index(drop=True)

        # 4. Fit the DynamicFactor model on returns, disabling stationarity enforcement
        #    and increasing maxiter to 1000 to reduce convergence warnings.
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                mod = DynamicFactor(
                    returns_df,
                    k_factors=self.k_factors,
                    factor_order=self.factor_order,
                    enforce_stationarity=False,
                )
                raw_res = mod.fit(disp=False, maxiter=1000)
            mod_fit: DynamicFactorResultsWrapper = cast(DynamicFactorResultsWrapper, raw_res)
        except Exception as e:
            raise RuntimeError(f"DynamicFactor model fitting failed: {e}")

        # 5. Obtain a one‐step‐ahead forecast for returns using get_forecast(steps=1)
        try:
            pred_res = mod_fit.get_forecast(steps=1)
            returns_forecast = pred_res.predicted_mean  # DataFrame of shape (1 × n)
        except Exception as e:
            raise RuntimeError(f"DynamicFactor get_forecast failed: {e}")

        if returns_forecast.shape != (1, n):
            raise RuntimeError(
                f"Expected a (1 × {n}) return‐forecast, but got {returns_forecast.shape}."
            )

        # 6. Convert forecasted returns to price forecasts
        last_prices = price_df.iloc[-1].values                # shape: (n,)
        predicted_returns = returns_forecast.values.flatten()  # shape: (n,)
        predicted_prices = last_prices * (1.0 + predicted_returns)

        return predicted_prices
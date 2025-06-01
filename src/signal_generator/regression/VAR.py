from abc import abstractmethod
from typing import Optional, Dict
import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR

from signal_generator.regression.base.oracle import Oracle  # adjust import path as needed


class VAROracle(Oracle):
    """
    Oracle implementation using a Vector Autoregression (VAR) on past returns.

    - __init__ requires:
        lag: int
            The VAR order (number of lags) to fit on the return series.
        name, version, params: passed through to the base Oracle initializer.

    - _predict_mean:
        1. Builds a DataFrame of prices (shape L×n).
        2. Computes columnwise pct_change → (L−1)×n return‐matrix.
        3. Fits VAR(self.lag) on those returns.
        4. Forecasts one‐step‐ahead return vector (shape (n,)).
        5. Converts the forecasted returns back to prices using the last observed prices.
    """

    def __init__(
        self,
        lag: int,
        name: str = "VARPredictiveOracle",
        version: str = "1.0",
        params: Optional[Dict[str, object]] = None,
    ):
        """
        Args:
            lag: The VAR order (number of lags) for model fitting. Must be ≥1.
            name: Optional human‐readable name for this oracle.
            version: Optional version string.
            params: Optional dictionary of additional parameters.
        """
        super().__init__(name=name, version=version, params=params)
        if not isinstance(lag, int) or lag < 1:
            raise ValueError("`lag` must be a positive integer.")
        self.lag = lag

    def _predict_mean(self, data: Dict[str, pd.Series]) -> np.ndarray:
        """
        Overrides Oracle._predict_mean using a VAR on historical returns.

        Steps:
          1. Convert `data` (dict of equal‐length pd.Series) into a DataFrame of prices.
          2. Compute returns_df = price_df.pct_change().dropna() → shape (L−1)×n.
          3. Ensure that returns_df has at least `self.lag + 1` rows.
          4. Fit VAR(self.lag) on the entire returns_df.
          5. Forecast 1 step ahead: returns_forecast = model.forecast(last_lagged_values, steps=1).
          6. Convert returns_forecast to next‐period price forecast:
               last_prices = price_df.iloc[-1].values,
               predicted_prices = last_prices * (1 + returns_forecast[0]).
          7. Return a 1D np.ndarray of length n in the same ticker order.
        """
        # 1. Build price DataFrame (L × n)
        try:
            price_df = pd.DataFrame(data)
        except Exception as e:
            raise ValueError(f"Unable to convert input data to DataFrame: {e}")

        if price_df.isnull().values.any():
            raise ValueError("Input price series contains NaN values. Clean the data first.")

        L, n = price_df.shape
        if L <= self.lag:
            raise ValueError(
                f"Need more than {self.lag} observations to fit VAR of order {self.lag}; got only {L}."
            )

        # 2. Compute returns: (L−1) × n
        returns_df = price_df.pct_change().dropna(axis=0, how="all")
        # Now returns_df.shape[0] == L − 1

        total_return_rows = returns_df.shape[0]
        if total_return_rows < self.lag:
            raise ValueError(
                f"Not enough return‐observations ({total_return_rows}) to fit VAR of lag {self.lag}."
            )

        # 3. Fit VAR(self.lag) on entire returns history
        try:
            var_model = VAR(returns_df.reset_index(drop=True))
            var_result = var_model.fit(self.lag)
        except Exception as e:
            raise RuntimeError(f"VAR model fitting failed: {e}")

        # 4. Prepare the last `self.lag` return‐vectors for forecasting
        #    shape required: (lag × n)
        last_obs = returns_df.values[-self.lag :]

        # 5. Forecast one step ahead (returns)
        try:
            forecasts = var_result.forecast(y=last_obs, steps=1)
            # forecasts has shape (1, n)
        except Exception as e:
            raise RuntimeError(f"VAR forecasting failed: {e}")

        predicted_returns = np.asarray(forecasts[0])  # shape (n,)

        # 6. Convert back to price forecasts
        last_prices = price_df.iloc[-1].values       # shape (n,)
        predicted_prices = np.array(list(last_prices)) * (1.0 + predicted_returns)

        return predicted_prices

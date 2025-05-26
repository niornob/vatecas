import numpy as np
import pandas as pd
from typing import Dict, Optional, Literal
from abc import ABC
from statsmodels.tsa.ar_model import AutoReg

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from signal_modules._regressors._oracle_base import Oracle

class AR1Oracle(Oracle):
    """
    AR(1) implementation of the Oracle interface using statsmodels AutoReg.
    """

    def __init__(
        self,
        name: str = "AR1Oracle",
        version: str = "2.0",
        params: Optional[Dict[str, object]] = None,
    ):
        """
        params can include:
          - 'trend': 'n' (no constant) or 'c' (constant term)
          - any other AutoReg kwargs you'd like to pass.
        """
        super().__init__(name=name, version=version, params=params or {})
        # default to no constant; users can override via params
        # Validate and set trend parameter
        raw_trend = self.params.get("trend", "n")
        assert raw_trend in ("n", "c", "t", "ct"), (
            f"Invalid trend specifier: {raw_trend!r}; "
            "must be one of 'n','c','t','ct'"
        )
        self.trend: Literal["n", "c", "t", "ct"] = raw_trend

    def _predict(self, data: Dict[str, pd.Series]) -> np.ndarray:
        """
        For each series in `data`, fit an AR(1) via AutoReg and forecast the next value.

        Returns:
            1-D numpy array of next-step forecasts in the same key-order as `data`.
        """
        data = {tk: s.reset_index(drop=True) for tk, s in data.items()}
        forecasts = []

        for key, series in data.items():
            # Fit AR(1)
            #   - lags=1 for pure AR(1)
            #   - trend=self.trend: 'n' for no constant, 'c' for constant
            try:
                model = AutoReg(series, lags=1, trend=self.trend, old_names=False)
                fit = model.fit()
            except:
                forecasts.append(0)
                continue

            # Forecast one step ahead
            # `predict` arguments are in-sample index positions
            #   start = len(series), end = len(series) gives y_{T+1}
            #print(fit.predict(start=series.shape[0], end=series.shape[0]).iloc[0])
            next_val = fit.predict(start=series.shape[0], end=series.shape[0]).iloc[0]
            forecasts.append(next_val)

        return np.array(forecasts)

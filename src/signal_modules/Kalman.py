from typing import Dict, Any, cast
import numpy as np
import pandas as pd
from typing import Mapping, Dict
import matplotlib.pyplot as plt
from tqdm import tqdm
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints
from scipy.special import expit, softmax
from scipy.stats import rankdata

from .base import SignalModule


class UKFSignalModule(SignalModule):
    """
    Signal generator using an Unscented Kalman Filter over the vector of asset prices.

    Treats the m assets' `adjClose` as components of an m-dimensional state vector
    and assumes a random-walk model for their evolution.

    Tuning Parameters
    -----------------
    - process_noise : float (default=1e-3)
        Variance for the assumed process noise Q.
    - alpha, beta, kappa : UKF sigma-point parameters.
    - proj_weight, sharpe_weight, rank_weight : floats (default=1.0)
        Weights for combining the three signal components.
    """
    def __init__(
        self,
        name: str = "UKFSignalModule",
        version: str = "0.2",
        params: Dict[str, Any] = {}
    ):
        super().__init__(name=name, version=version, params=params)
        # will be set after generate_signals
        self.last_pred_returns: pd.Series = pd.Series()

    def generate_signals(
        self,
        data: Dict[str, pd.DataFrame],
        diagnostics: bool = False,
    ) -> Dict[str, float]:
        # List of tickers and number of assets
        tickers = list(data.keys())
        m = len(tickers)

        # Align and stack adjusted close prices
        price_df = pd.concat(
            [data[t]['adjClose'].rename(t) for t in tickers],
            axis=1,
        ).dropna()
        values = price_df.values  # shape (n_steps, m)

        # State-transition and observation
        def fx(x: np.ndarray, dt: float) -> np.ndarray:
            return x
        def hx(x: np.ndarray) -> np.ndarray:
            return x

        # Retrieve tuning parameters with proper typing
        proc_noise = cast(float, self.params.get('process_noise', 1e-3))
        alpha = cast(float, self.params.get('alpha', 0.1))
        beta = cast(float, self.params.get('beta', 2.0))
        kappa = cast(float, self.params.get('kappa', 0.0))
        w_proj = cast(float, self.params.get('proj_weight', 1.0))
        w_sharpe = cast(float, self.params.get('sharpe_weight', 1.0))
        w_rank = cast(float, self.params.get('rank_weight', 5.0))
        w_gain = cast(float, self.params.get('gain_weight', 1.0))
        gain_scale = cast(float, self.params.get('gain_scale', 10.0))

        # Sigma points and filter init
        points = MerweScaledSigmaPoints(n=m, alpha=alpha, beta=beta, kappa=kappa)
        ukf = UnscentedKalmanFilter(
            dim_x=m,
            dim_z=m,
            dt=1.0,
            fx=fx,
            hx=hx,
            points=points
        )
        ukf.x = values[0].copy()
        ukf.P = np.eye(m) * np.var(values, axis=0).mean()
        ukf.R = np.eye(m) * np.var(values - values.mean(axis=0), axis=0).mean()
        ukf.Q = np.eye(m) * proc_noise

        # Apply filter
        for z in values:
            # Predict step with regularization retry
            try:
                ukf.predict()
            except np.linalg.LinAlgError:
                ukf.P += np.eye(m) * 1e-6
                ukf.predict()
            # Update step with regularization retry
            try:
                ukf.update(z)
            except np.linalg.LinAlgError:
                ukf.P += np.eye(m) * 1e-6
                ukf.update(z)
        # One final predict, handling possible PD issues
        try:
            ukf.predict()
        except np.linalg.LinAlgError:
            ukf.P += np.eye(m) * 1e-6
            ukf.predict()
        predicted = ukf.x
        last_obs = values[-1]

        # Compute predicted returns safely
        returns_pred = (predicted - last_obs) / np.where(last_obs != 0, last_obs, 1)
        self.last_pred_returns = pd.Series(predicted, index=tickers)

        # Statistics for normalization
        mean_ret = returns_pred.mean()
        std_ret = returns_pred.std(ddof=0) if returns_pred.size > 1 else 0.0
        eps = 1e-8
        std_ret = std_ret if std_ret > 0 else eps

        # Three signal components
        proj = expit((returns_pred - mean_ret) / std_ret)
        sharpe = expit(returns_pred / std_ret)
        ranks = rankdata(returns_pred, method='average')
        rank_sig = ranks / m
        gain_scores = softmax(gain_scale * returns_pred)

        # Weighted combination of all four
        total_w = w_proj + w_sharpe + w_rank + w_gain
        combined = (
            w_proj * proj +
            w_sharpe * sharpe +
            w_rank * rank_sig +
            w_gain * gain_scores
        ) / total_w

        # Return dict of signals
        return {tickers[i]: float(combined[i]) for i in range(m)}

    def plot_cumulative_returns(self, data: Mapping[str, pd.DataFrame]) -> None:
        """
        One-step-ahead forecast vs actual cumulative price plots for each ticker.
        """
        # Prepare actual price DataFrame
        actual = pd.DataFrame({t: data[t]['adjClose'] for t in data})
        dates  = actual.index
        tickers = actual.columns

        # Initialize DataFrame for forecasts
        preds = pd.DataFrame(index=dates, columns=tickers, dtype=float)
        preds.iloc[0] = actual.iloc[0]

        # Iteratively predict one-step ahead
        for i in tqdm(range(1, len(dates)), desc='UKF forecasting'):
            history = {t: df.iloc[max(0,i-50):i] for t, df in data.items()}
            _ = self.generate_signals(history)
            ret_series = pd.Series(self.last_pred_returns, index=tickers)
            preds.iloc[i] = ret_series

        # Plot for each ticker
        for ticker in tickers:
            plt.figure(figsize=(10, 4))
            plt.plot(dates, actual[ticker], label=f"Actual {ticker}")
            plt.plot(dates, preds[ticker],  label="1-Step Forecast", linestyle='-', linewidth=1)
            plt.title(f"One-Step-Ahead Forecast vs Actual — {ticker}")
            plt.xlabel("Date")
            plt.ylabel("Price")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()
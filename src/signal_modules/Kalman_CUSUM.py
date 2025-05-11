import numpy as np
import pandas as pd
from .base import SignalModule


class KallmanCUSUM(SignalModule):
    def __init__(
        self,
        process_var: float = 1e-5,
        meas_var: float = 1e-2,
        cusum_threshold: float = 0,
        eps: float = 1e-6
    ):
        super().__init__(name="KallmanCUSUM")
        self.process_var = process_var
        self.meas_var = meas_var
        self.cusum_threshold = cusum_threshold
        self.eps = eps

    def generate_signals(
        self,
        data: dict[str, pd.DataFrame],
        price: str = 'adjClose',
    ) -> dict[str, pd.Series]:
        """
        Generate a buy/sell signal in [0,1] from a price series using a Kalman filter
        to estimate the drift and a CUSUM algorithm to detect trends.

        Parameters
        ----------
        data : dict mapping a ticker (str) to historical data (pd.DataFrame containing at least one collumn price)
            Time-indexed.
        process_var : float
            Variance of the process noise (Q) in Kalman filter (drift evolution).
        meas_var : float
            Variance of the measurement noise (R) in Kalman filter.
        cusum_threshold : float
            Reference value k for CUSUM (drift offset), default zero.
        eps : float
            Small constant to avoid division by zero.

        Returns
        -------
        signals : dict mapping a ticker (str) to a time series of signals,
            valued in [0,1], where values near 1 indicate buy pressure
            and values near 0 indicate sell pressure.
        """
        signals: dict[str, pd.Series] = {}

        for ticker, df in data.items():
            price_series = df[price]

            # Compute price differences
            delta = price_series.diff().fillna(0).values
            n = len(delta)

            # Allocate arrays for drift estimate and error covariance
            drift_est = np.zeros(n)
            p = np.zeros(n)

            # Initial estimates
            drift_est[0] = 0.0
            p[0] = 1.0

            # Kalman filter recursion
            for t in range(1, n):
                # Prediction step
                drift_pred = drift_est[t-1]
                p_pred = p[t-1] + self.process_var

                # Update step
                k_gain = p_pred / (p_pred + self.meas_var)
                drift_est[t] = drift_pred + k_gain * (delta[t] - drift_pred)
                p[t] = (1 - k_gain) * p_pred

            # CUSUM for positive and negative drift detection
            pos_cusum = np.zeros(n)
            neg_cusum = np.zeros(n)
            for t in range(1, n):
                pos_cusum[t] = max(0, pos_cusum[t-1] + drift_est[t] - self.cusum_threshold)
                neg_cusum[t] = max(0, neg_cusum[t-1] - (drift_est[t] + self.cusum_threshold))

            # Combine into a normalized signal
            delta = 0.05 * np.mean(pos_cusum + neg_cusum)
            signal_values = (pos_cusum + delta) / (pos_cusum + neg_cusum + 2*delta)

            ticker_signal = pd.Series(signal_values, index=price_series.index)
            
            signals[ticker] = ticker_signal

        return signals

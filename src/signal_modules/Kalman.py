import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.special import expit
from typing import Mapping, Dict
from .base import SignalModule

class MultiKalmanWithBias(SignalModule):
    """
    Multivariate Kalman filter over m stocks with adaptive bias term.
    Uses k = number of tickers as latent dimension, plus bias.
    Implements buy signals as a weighted average of:
      1. Projected return sigmoid,
      2. Sharpe-like signal,
      3. Cross-sectional rank.
    """

    def __init__(
        self,
        process_window: int = 20,
        meas_window: int = 5,
        bias_noise: float = 1e-6,
        version: str = '0.3'
    ):
        super().__init__(name="MultiKalmanWithBias", version=version, params={
            'process_window': process_window,
            'meas_window': meas_window,
            'bias_noise': bias_noise
        })
        self.process_window = process_window
        self.meas_window = meas_window
        self.bias_noise = bias_noise
        # placeholders, will set k, n_state, F in generate_signals
        self.k = None
        self.n_state = None
        self.F = None
        self.H = None
        self.last_x_hat = None

    def generate_signals(
        self,
        data: Mapping[str, pd.DataFrame],
        diagnostics: bool = False
    ) -> Dict[str, pd.Series]:
        # Prepare returns matrix
        tickers = list(data.keys())
        price_df = pd.DataFrame({t: data[t]['adjClose'] for t in tickers})
        returns = price_df.diff().fillna(0)
        T, m = returns.shape

        # Set latent dimension k to number of tickers
        self.k = m
        # State dimension = factors + bias
        self.n_state = self.k + 1
        # State-transition identity
        self.F = np.eye(self.n_state)

        # PCA to estimate loadings H_factors (m x k)
        pca = PCA(n_components=self.k)
        pca.fit(returns.values[:self.process_window])
        H_factors = pca.components_.T  # m x k
        # augment with bias column
        H_bias = np.ones((m, 1))
        self.H = np.hstack([H_factors, H_bias])  # m x (k+1)

        # Initialize state estimate and covariance
        x_hat = np.zeros((T, self.n_state))
        P = np.eye(self.n_state)

        # Kalman recursion
        for t in range(T):
            y_t = returns.iloc[t].values.reshape(-1, 1)
            # process covariance Q_t
            window_p = returns.iloc[max(0, t-self.process_window):t+1].values
            raw_Q = np.cov(window_p, rowvar=False)
            proc_var = (np.trace(raw_Q) / m) if raw_Q.ndim >= 2 else float(raw_Q) / m
            Q_f = np.eye(self.k) * proc_var
            Q_b = np.array([[self.bias_noise]])
            Q_t = np.block([ [Q_f, np.zeros((self.k,1))], [np.zeros((1,self.k)), Q_b] ])
            # measurement covariance R_t
            window_r = returns.iloc[max(0, t-self.meas_window):t+1].values
            raw_R = np.cov(window_r, rowvar=False)
            R_t = raw_R if raw_R.ndim >= 2 else np.eye(m) * (float(raw_R) / m)
            # prediction
            prev_x = x_hat[t-1] if t > 0 else np.zeros(self.n_state)
            x_pred = self.F @ prev_x
            P_pred = self.F @ P @ self.F.T + Q_t
            # update
            S = self.H @ P_pred @ self.H.T + R_t
            K = P_pred @ self.H.T @ np.linalg.inv(S)
            innovation = y_t - (self.H @ x_pred.reshape(-1,1))
            x_hat[t] = (x_pred + (K @ innovation).ravel())
            P = (np.eye(self.n_state) - K @ self.H) @ P_pred

        self.last_x_hat = x_hat

        # Reconstruct predicted returns (T x m)
        pred_array = (self.H @ x_hat.T).T
        pred_df = pd.DataFrame(pred_array, index=returns.index, columns=tickers)

        # Signal construction
        eps = 1e-8
        # 1. Projected-return sigmoid (rolling z-score)
        roll_mean = pred_df.rolling(self.process_window, min_periods=1).mean()
        roll_std = pred_df.rolling(self.process_window, min_periods=1).std().fillna(eps)
        proj_z = (pred_df - roll_mean) / roll_std
        proj_sig = expit(proj_z)
        # 2. Sharpe-like sigmoid
        sharpe_sig = expit(pred_df / roll_std)
        # 3. Cross-sectional rank
        cs_rank = pred_df.rank(axis=1, pct=True)
        # Weighted average
        w1, w2, w3 = 1.0, 1.0, 1.0
        signal_df = (w1*proj_sig + w2*sharpe_sig + w3*cs_rank) / (w1 + w2 + w3)

        # Output as pd.Series per ticker
        signals: Dict[str, pd.Series] = {}
        for t in tickers:
            s = signal_df[t].copy()
            s.name = 'buy_confidence'
            signals[t] = s

        if diagnostics:
            self.plot_cumulative_returns(data)

        return signals

    def plot_cumulative_returns(
        self,
        data: Mapping[str, pd.DataFrame]
    ) -> None:
        """
        Plot cumulative observed returns vs model-predicted returns.
        """
        if self.last_x_hat is None:
            raise RuntimeError("Call generate_signals() before plotting.")

        tickers = list(data.keys())
        price_df = pd.DataFrame({t: data[t]['adjClose'] for t in tickers})
        returns = price_df.diff().fillna(0)
        cum_obs = returns.cumsum()

        # predicted returns = H @ x_hat'(t)
        x_df = pd.DataFrame(self.last_x_hat, index=returns.index)
        pred = (self.H @ x_df.T).T
        # pred is already a DataFrame; just set its columns
        pred_df = pred.copy()
        pred_df.columns = tickers
        cum_pred = pred_df.cumsum()

        for t in tickers:
            plt.figure(figsize=(10,6))
            plt.plot(cum_obs.index, cum_obs[t], label=f"Obs {t}")
            plt.plot(cum_pred.index, cum_pred[t], '--', label=f"Pred {t}")
            plt.legend()
            plt.title(f"({t}) Cumulative Observed vs Predicted Returns")
            plt.xlabel("Date")
            plt.ylabel("Cumulative Return")
            plt.show()

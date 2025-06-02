import numpy as np
import pandas as pd
from typing import Optional, Dict, cast
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints

from .base.oracle import Oracle


class UKFlog(Oracle):
    """
    Oracle implementation using a 2m‐dimensional UKF (log‐price + volatility per ticker).
    - _predict_mean expects `data: dict[str, pd.Series]` of *historical prices* (not log).
    - Internally, we convert prices → log(prices) and build a UKF state of size 2m:
        [ log-price_1, …, log-price_m, vol_1, …, vol_m ]^T
      where each volatility is treated as a stochastic state.
    - We return a vector of *predicted prices* (not log-prices).
    """

    def __init__(
        self,
        name: str = "UKF_log",
        version: str = "1.0",
        params: Optional[Dict[str, object]] = None,
        process_noise: float = 0.01,
        obs_noise_scale: float = 1.0,
        alpha: float = 0.1,
        beta: float = 2.0,
        kappa: float = 0.0,
        warmup_period: int = 0,
    ):
        super().__init__(name, version, params)

        # UKF hyperparameters
        self.process_noise = process_noise
        self.obs_noise_scale = obs_noise_scale
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self.warmup_period = warmup_period

    def _predict_mean(self, data: dict[str, pd.Series]) -> np.ndarray:
        """
        - `data` is a dict mapping ticker → pandas.Series of historical *prices*.
        - We:
          1. Validate and align series into a DataFrame of shape (T, m).
          2. Convert to log-prices.
          3. Call _create_and_apply_filter on the log-prices array.
          4. Exponentiate the returned log-price forecast to get actual price predictions.
        """
        if not data:
            raise ValueError("Input data dictionary cannot be empty")

        # --- Step 1: Align and validate lengths ---
        lengths = [len(s) for s in data.values()]
        if len(set(lengths)) > 1:
            raise ValueError("All pandas Series in data must have the same length")

        # Build a DataFrame so that columns align by index
        df_prices = pd.DataFrame(data)  # shape = (T, m), columns = tickers
        price_matrix = df_prices.to_numpy()  # shape (T, m)

        if price_matrix.size == 0:
            raise ValueError("No data points available for prediction")

        # --- Step 2: Convert to log-prices ---
        # (We assume prices > 0; if any price ≤ 0, log is undefined.)
        if np.any(price_matrix <= 0):
            raise ValueError("All price values must be positive to take log.")

        log_matrix = np.log(price_matrix)  # shape (T, m)

        # --- Step 3: UKF on log-prices (returns predicted log-prices) ---
        log_prediction = self._create_and_apply_filter(
            values=log_matrix,
            process_noise=self.process_noise,
            obs_noise_scale=self.obs_noise_scale,
            alpha=self.alpha,
            beta=self.beta,
            kappa=self.kappa,
            warmup_period=self.warmup_period,
        )  # returns a 1D array of length m (predicted log-prices)

        # --- Step 4: Exponentiate to get price predictions ---
        price_prediction = np.exp(log_prediction)  # shape (m,)
        return price_prediction

    def _create_and_apply_filter(
        self,
        values: np.ndarray,
        process_noise: float,
        obs_noise_scale: float = 1.0,
        alpha: float = 0.5,
        beta: float = 2.0,
        kappa: float = 0.0,
        warmup_period: int = 0,
    ) -> np.ndarray:
        """
        - `values` is an (T x m) array of historical log-prices.
        - We build a UKF with state dimension = 2m (log-prices + volatilities).
        - Returns a 1D array of length m: the predicted log-prices at time T+1.
        """
        T, m = values.shape
        state_dim = 2 * m  # [X_1,…,X_m, sigma_1,…,sigma_m]

        # --- 1) Define fx and hx for a 2m-dimensional state ---
        def fx(x: np.ndarray, dt: float = 1.0) -> np.ndarray:
            """
            x: 2m-vector = [ X_1,…,X_m, sigma_1,…,sigma_m ]
            We assume:
              X^{(i)}_{k+1} = X^{(i)}_k  - 0.5 * (sigma^{(i)}_k)^2   (deterministic part)
              sigma^{(i)}_{k+1} = sigma^{(i)}_k                     (volatility is “frozen”)
            Process noise is added via Q.
            """
            X = x[:m].copy()
            vol = x[m:].copy()

            # For each ticker i: next log-price = X_i - 0.5 * vol_i^2
            X_next = X - 0.5 * (vol**2)

            # Next volatility stays the same (random-walk component is in Q)
            vol_next = vol

            return np.concatenate([X_next, vol_next])

        def hx(x: np.ndarray) -> np.ndarray:
            """
            Measurement function: we only observe the log-prices (not volatilities).
            x is 2m-vector; return an m-vector of log-prices.
            """
            return x[:m]

        # --- 2) Initialize UKF with Merwe sigma points ---
        points = MerweScaledSigmaPoints(
            n=state_dim, alpha=alpha, beta=beta, kappa=kappa
        )
        ukf = UnscentedKalmanFilter(
            dim_x=state_dim, dim_z=m, dt=1.0, fx=fx, hx=hx, points=points
        )

        # --- 3) Initialize ukf.x and ukf.P ---
        # We need an initial guess for [log-prices, volatilities].

        # 3.a) Estimate each ticker’s “initial volatility” from historical log returns:
        if T > 1:
            log_returns = np.diff(values, axis=0)  # shape (T-1, m)
            # Use the last min(10, T-1) returns to form a volatility estimate:
            window = min(10, log_returns.shape[0])
            recent_returns = log_returns[-window:, :]
            # vol_est[i] = std of the last `window` returns of ticker i
            vol_est = np.std(recent_returns, axis=0, ddof=0)  # shape (m,)
        else:
            # If there's only one time point, we cannot compute returns. Initialize vol to a small positive number.
            vol_est = np.full((m,), fill_value=1e-3)

        # 3.b) Choose initial state (log-price, vol) depending on how many samples we have:
        if T > max(10, warmup_period):
            # If enough data, anchor log-prices at last observed log-price,
            # and volatilities at our estimated vol.
            last_log_prices = values[-1, :]  # shape (m,)
            initial_vols = vol_est  # shape (m,)
            ukf.x = np.concatenate([last_log_prices, initial_vols])
        else:
            # If too few points, initialize log-prices at the first value,
            # and volatilities at vol_est (or a small default).
            first_log_prices = values[0, :]
            ukf.x = np.concatenate([first_log_prices, vol_est])

        # 3.c) Build an initial covariance P of size 2m x 2m:
        base_var = np.var(values, axis=0).mean()  # scalar average of per-ticker var
        if T > 1:
            recent_changes = log_returns[-min(20, log_returns.shape[0]) :, :]
            if recent_changes.shape[0] > 1:
                change_var = np.var(recent_changes, axis=0).mean()
            else:
                change_var = base_var
        else:
            change_var = base_var

        # Price‐side covariance (diagonal)
        price_var_init = max(base_var, change_var) * 2.0  # scalar
        P_price = np.eye(m) * price_var_init

        # Volatility‐side covariance: choose a rough scale. We can base it on var of vol_est:
        vol_var = np.var(vol_est) if (vol_est.size > 1) else base_var
        vol_var_init = max(cast(float, vol_var), 1e-4) * 2.0
        P_vol = np.eye(m) * vol_var_init

        # Assemble P as block-diagonal [P_price    0
        #                               0       P_vol ]
        ukf.P = np.block(
            [[P_price, np.zeros((m, m))], [np.zeros((m, m)), P_vol]]
        )  # shape (2m, 2m)

        # --- 4) Build ukf.R (measurement noise) ---
        # We observe log-prices directly with some noise.  Use the same idea:
        base_obs_noise = np.var(values - values.mean(axis=0), axis=0).mean()
        if T > 10:
            recent_obs_noise = np.var(
                values[-10:, :] - np.mean(values[-10:, :], axis=0), axis=0
            ).mean()
            R_price = recent_obs_noise * obs_noise_scale
        else:
            R_price = base_obs_noise * obs_noise_scale

        ukf.R = np.eye(m) * R_price  # shape (m, m)

        # --- 5) Build ukf.Q (process noise) ---
        # We need a 2m x 2m matrix.  We split it into two diagonal blocks:
        #   - Q_price = process_noise * (recent_obs_noise)   (for the price‐part)
        #   - Q_vol   = process_noise * (recent_obs_noise)   (for the volatility‐part)
        # (One could choose a different scale for vol, e.g. smaller, but we keep same here.)
        price_noise = (recent_obs_noise if T > 10 else base_obs_noise) * process_noise
        vol_noise = price_noise  # you might choose a smaller vol_noise in practice

        Q_price = np.eye(m) * price_noise
        Q_vol = np.eye(m) * vol_noise

        ukf.Q = np.block(
            [[Q_price, np.zeros((m, m))], [np.zeros((m, m)), Q_vol]]
        )  # shape (2m, 2m)

        # --- 6) Warmup loop (optional) ---
        warmup_data = values[:warmup_period] if warmup_period > 0 else np.empty((0, m))
        for z in warmup_data:
            try:
                ukf.predict()
                ukf.update(z)  # z is an m-vector of log-prices at that time
            except np.linalg.LinAlgError:
                ukf.P += np.eye(state_dim) * 1e-6
                ukf.predict()
                ukf.update(z)

        # --- 7) Main filtering loop ---
        main_data = values[warmup_period:] if warmup_period > 0 else values
        for z in main_data:
            try:
                ukf.predict()
            except np.linalg.LinAlgError:
                ukf.P += np.eye(state_dim) * 1e-6
                ukf.predict()
            try:
                ukf.update(z)  # z is an m-vector of log-prices
            except np.linalg.LinAlgError:
                ukf.P += np.eye(state_dim) * 1e-6
                ukf.update(z)

        # --- 8) Final one-step‐ahead predict ---
        try:
            ukf.predict()
        except np.linalg.LinAlgError:
            ukf.P += np.eye(state_dim) * 1e-6
            ukf.predict()

        # The UKF’s state vector is now [pred_log_prices (m,), pred_vols (m,)].
        # We only return the *log-price* portion, i.e. ukf.x[:m].
        predicted_log_prices = ukf.x[:m].copy()  # shape (m,)
        return predicted_log_prices

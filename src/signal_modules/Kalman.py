from typing import Dict, Any, cast
import numpy as np
import pandas as pd
from typing import Mapping
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints
from scipy.stats import rankdata
import matplotlib.pyplot as plt
from tqdm import tqdm
import os



from .base import SignalModule


class UKFSignalModule(SignalModule):
    def __init__(
        self,
        name: str = "UKFSignalModule",
        version: str = "0.2",
        params: Dict[str, Any] = {},
    ):
        print("model loaded from: ", os.path.abspath(__file__))
        super().__init__(name=name, version=version, params=params)
        self.last_prediction: pd.Series = pd.Series()

    def _create_and_apply_filter(
        self,
        values: np.ndarray,
        process_noise: float,
        obs_noise_scale: float = 1.0,
        alpha: float = 0.1,
        beta: float = 2.0,
        kappa: float = 0.0,
        warmup_period: int = 0,
    ) -> np.ndarray:
        """Create and apply a UKF, returning last `history_length` predicted states."""
        m = values.shape[1]

        def fx(x: np.ndarray, dt: float) -> np.ndarray:
            return x

        def hx(x: np.ndarray) -> np.ndarray:
            return x

        points = MerweScaledSigmaPoints(n=m, alpha=alpha, beta=beta, kappa=kappa)
        ukf = UnscentedKalmanFilter(
            dim_x=m, dim_z=m, dt=1.0, fx=fx, hx=hx, points=points
        )

        # Initialization
        if len(values) > max(10, warmup_period):
            ukf.x = values[-1].copy()
            recent_changes = np.diff(values[-min(20, len(values)-1):], axis=0)
            base_var = np.var(values, axis=0).mean()
            change_var = (
                np.var(recent_changes, axis=0).mean()
                if len(recent_changes) > 1
                else base_var
            )
            ukf.P = np.eye(m) * max(base_var, change_var) * 2.0
        else:
            ukf.x = values[0].copy()
            ukf.P = np.eye(m) * np.var(values, axis=0).mean()

        base_obs_noise = np.var(values - values.mean(axis=0), axis=0).mean()
        if len(values) > 10:
            recent_obs_noise = np.var(
                values[-10:] - np.mean(values[-10:], axis=0), axis=0
            ).mean()
            ukf.R = np.eye(m) * recent_obs_noise * obs_noise_scale
        else:
            ukf.R = np.eye(m) * base_obs_noise * obs_noise_scale
        ukf.Q = np.eye(m) * process_noise

        # Warmup
        warmup_data = values[:warmup_period] if warmup_period > 0 else []
        for z in warmup_data:
            try:
                ukf.predict()
                ukf.update(z)
            except np.linalg.LinAlgError:
                ukf.P += np.eye(m) * 1e-6
                ukf.predict();
                ukf.update(z)

        main_data = values[warmup_period:] if warmup_period > 0 else values
        for z in main_data:
            try:
                ukf.predict()
            except np.linalg.LinAlgError:
                ukf.P += np.eye(m) * 1e-6
                ukf.predict()
            try:
                ukf.update(z)
            except np.linalg.LinAlgError:
                ukf.P += np.eye(m) * 1e-6
                ukf.update(z)

        try:
            ukf.predict()
        except np.linalg.LinAlgError:
            ukf.P += np.eye(m) * 1e-6
            ukf.predict()
        
        return ukf.x

    def generate_signals(
        self,
        data: Dict[str, pd.DataFrame],
    ) -> Dict[str, float]:
        tickers = list(data.keys())
        m = len(tickers)
        price_df = pd.concat([data[t]["adjClose"].rename(t) for t in tickers], axis=1).dropna()
        values = price_df.values

        # parameters
        proc_noise = cast(float, self.params.get("fast_process_noise", 1e-2))
        obs_scale = cast(float, self.params.get("fast_observation_noise_scale", 1e3))
        warmup = cast(int, self.params.get("warmup", min(30, max(0, len(values)//3))))
        alpha = cast(float, self.params.get("alpha", 0.1))
        beta = cast(float, self.params.get("beta", 2.0))
        kappa = cast(float, self.params.get("kappa", 0.0))

        predicted_price = self._create_and_apply_filter(
            values, proc_noise, obs_scale, alpha, beta, kappa, warmup
        )

        #print(predicted_price)

        self.last_prediction = pd.Series(predicted_price, index=tickers)
        last_obs = values[-1]
        returns_pred = (predicted_price - last_obs) / np.where(last_obs != 0, last_obs, 1)
        ranks = rankdata(returns_pred, method="average")
        signals = ranks / m

        return {tickers[i]: float(signals[i]) for i in range(m)}

    def diagnostics(self, data: Mapping[str, pd.DataFrame], window_size: int = 50) -> None:
        actual = pd.DataFrame({t: data[t]["adjClose"] for t in data})
        dates = actual.index
        tickers = actual.columns
        preds = pd.DataFrame(index=dates, columns=tickers, dtype=float)
        preds.iloc[0] = actual.iloc[0]
        window_size = min(window_size, len(actual))

        for i in tqdm(range(1, len(dates)), desc="UKF forecasting"):
            history = {t: df.iloc[max(0, i-window_size):i] for t, df in data.items()}
            _ = self.generate_signals(history)
            preds.iloc[i, :]    = self.last_prediction.to_numpy()

        for ticker in tickers:
            plt.figure(figsize=(10,4))
            plt.plot(dates, actual[ticker], label=f"Actual {ticker}")
            plt.plot(dates, preds[ticker], label="1-Step Combined Forecast", linestyle='-', linewidth=1)
            plt.title(f"One-Step-Ahead Forecast vs Actual — {ticker}")
            plt.xlabel("Date"); plt.ylabel("Price")
            plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

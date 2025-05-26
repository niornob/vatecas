import numpy as np
from scipy.signal import find_peaks
from typing import Optional, Tuple, Any, Dict
import pandas as pd
import matplotlib.pyplot as plt
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from signal_modules._regressors._oracle_base import Oracle


class KalmanFilter(Oracle):
    """
    Oracle implementation using Unscented Kalman Filter for time series prediction.

    This class wraps the UKF-based prediction functionality to conform to the oracle
    interface, handling the conversion between pandas Series dictionary format and
    the numpy array format expected by the underlying Kalman filter.
    """

    def __init__(
        self,
        name: str = "KalmanOracle",
        version: str = "1.0",
        params: Optional[Dict[str, object]] = None,
        process_noise: float = 0.01,
        obs_noise_scale: float = 1.0,
        alpha: float = 0.1,
        beta: float = 2.0,
        kappa: float = 0.0,
        warmup_period: int = 0
    ):
        """
        Initialize the Kalman Filter Oracle.

        Args:
            name: Name identifier for this oracle instance
            version: Version string for tracking
            params: Additional parameters dictionary (optional)
            process_noise: Process noise parameter for the UKF (Q matrix scaling)
            obs_noise_scale: Observation noise scaling factor (R matrix scaling)
            alpha: UKF sigma point spread parameter (typically 0.001 to 1)
            beta: UKF parameter for prior knowledge of distribution (2 is optimal for Gaussian)
            kappa: UKF secondary scaling parameter (usually 0 or 3-n where n is state dimension)
            warmup_period: Number of initial data points to use for filter warmup
        """
        super().__init__(name, version, params)

        # Store UKF parameters as instance variables for easy access
        self.process_noise = process_noise
        self.obs_noise_scale = obs_noise_scale
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self.warmup_period = warmup_period

    def _predict(self, data: dict[str, pd.Series]) -> np.ndarray:
        # Input validation
        if not data:
            raise ValueError("Input data dictionary cannot be empty")

        # Check that all Series have the same length
        series_lengths = [len(series) for series in data.values()]
        if len(set(series_lengths)) > 1:
            raise ValueError("All pandas Series in data must have the same length")

        # Convert dictionary of Series to numpy array
        # pd.DataFrame automatically aligns Series by index and creates columns
        df = pd.DataFrame(data)
        values = df.to_numpy()

        # Validate that we have enough data points
        if len(values) == 0:
            raise ValueError("No data points available for prediction")

        # Call the existing Kalman filter method with stored parameters
        prediction = self._create_and_apply_filter(
            values=values,
            process_noise=self.process_noise,
            obs_noise_scale=self.obs_noise_scale,
            alpha=self.alpha,
            beta=self.beta,
            kappa=self.kappa,
            warmup_period=self.warmup_period,
        )

        return prediction

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
            recent_changes = np.diff(values[-min(20, len(values) - 1) :], axis=0)
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
                ukf.predict()
                ukf.update(z)

        # Main filtering loop
        main_data = values[warmup_period:] if warmup_period > 0 else values
        for idx, z in enumerate(main_data, start=1):
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

        # Final prediction for next time step
        try:
            ukf.predict()
        except np.linalg.LinAlgError:
            ukf.P += np.eye(m) * 1e-6
            ukf.predict()

        # Return the predicted next state as a 1D numpy array
        return ukf.x.copy()

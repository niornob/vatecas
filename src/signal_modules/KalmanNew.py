from typing import Dict, Any, cast, Tuple
import numpy as np
import pandas as pd
from typing import Mapping
from scipy.stats import rankdata
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from scipy.special import expit
from collections import deque

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from signal_modules.base import SignalModule
from signal_modules._regressors.Kalman_filter import KalmanFilter


class UKFSignalModule(SignalModule):
    def __init__(
        self,
        name: str = "UKFSignalModule",
        version: str = "0.2",
        params: Dict[str, Any] = {},
    ):
        print(name, "model loaded from:", os.path.abspath(__file__))
        super().__init__(name=name, version=version, params=params)
        self.window = params.get("window", 40)
        self.last_signals: deque[np.ndarray] = deque([], maxlen=10)

    def generate_signals(
        self, data: Dict[str, pd.DataFrame], prediction_history: deque = deque([])
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        tickers = list(data.keys())
        m = len(tickers)

        # parameters
        proc_noise = cast(float, self.params.get("process_noise", 1e-3))
        obs_scale = cast(float, self.params.get("observation_noise_scale", 1e-2))
        warmup = cast(int, self.params.get("warmup", 0))
        alpha = cast(float, self.params.get("alpha", 0.1))
        beta = cast(float, self.params.get("beta", 2.0))
        kappa = cast(float, self.params.get("kappa", 0.0))

        regressor = KalmanFilter(
            process_noise=proc_noise,
            obs_noise_scale=obs_scale,
            alpha=alpha,
            beta=beta,
            kappa=kappa,
            warmup_period=warmup,
        )

        price = {tk: df["adjClose"] for tk, df in data.items()}
        predicted_price = regressor._predict(data=price)
        tomorrow = max(list(price.values())[0].index) + pd.Timedelta(days=1)
        augmented_price_df = pd.DataFrame(price)
        price_today = augmented_price_df.values[-1]
        #print(price_today)
        augmented_price_df.loc[tomorrow] = predicted_price
        predicted_price_2 = regressor._predict(
            data={tk: augmented_price_df[tk] for tk in augmented_price_df.columns}
        )

        if len(prediction_history) < 3:
            assert(len(list(price.values())[0]) > 3)
            for i in range(3):
                pred = regressor._predict(
                    data={tk: s.iloc[: i - 3] for tk, s in price.items()}
                )
                prediction_history.append(pred)

        momentum = prediction_history[-1] - prediction_history[-2]
        two_step_momentum = prediction_history[-1] - prediction_history[-3]
        one_step_return = predicted_price - prediction_history[-1]
        one_step_returnX = predicted_price - price_today
        two_step_return = predicted_price_2 - prediction_history[-1]
        two_step_return_lagged = predicted_price - prediction_history[-2]
        acceleration = one_step_return - momentum

        returns_pred = (two_step_return_lagged+acceleration) / prediction_history[-1]
        self.last_signals.append(expit(returns_pred) - 0.5)
        highest_signal = max(x for sig in self.last_signals for x in sig)

        signals = assign_ranks(returns_pred, highest_signal)

        return {tickers[i]: float(signals[i]) for i in range(m)}, {
            tk: price for tk, price in zip(tickers, predicted_price)
        }
    


import numpy as np


def assign_ranks(arr: np.ndarray, high: float) -> np.ndarray:
    rank = np.array([expit(x) for x in arr]) - 0.5
    rank_max = max(abs(rank))

    return rank / (2 * max(rank_max, high)) + 0.5

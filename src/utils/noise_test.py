# time_series_diagnostics.py
"""
A module providing a suite of statistical tests and metrics to diagnose whether a time series
is white noise or contains meaningful signal (autocorrelation, trend, periodicity, etc.).
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf, pacf, adfuller, kpss
from statsmodels.sandbox.stats.runs import runstest_1samp
from scipy.signal import periodogram


def hurst_exponent(ts: np.ndarray) -> float:
    """
    Estimate the Hurst exponent of a time series.
    H ≈ 0.5: uncorrelated (white noise),
    H > 0.5: persistent/correlated,
    H < 0.5: mean-reverting.
    """
    N = len(ts)
    Y = np.cumsum(ts - np.mean(ts))
    R = np.max(Y) - np.min(Y)
    S = np.std(ts)
    if S == 0 or N <= 1:
        return np.nan
    return np.log(R / S) / np.log(N)


def diagnose_series(X: pd.Series) -> None:
    """
    Run a battery of tests on the input series and print results.

    Parameters
    ----------
    X : pd.Series
        Input time series (e.g. returns).
    plot_psd : bool
        If True, show the power spectral density plot.
    """
    print("\nTime Series Diagnostics Report")
    print("-------------------------------")

        # 1. Autocorrelation & Ljung-Box
    lags = min(20, len(X) - 1)
    acf_res = acf(X.dropna(), nlags=lags, alpha=0.05)
    acf_vals = acf_res[0] if isinstance(acf_res, tuple) else acf_res
    acf_conf = acf_res[1] if isinstance(acf_res, tuple) and len(acf_res) > 1 else None
    # Ljung-Box: handle both tuple and DataFrame return types
    lb_res = acorr_ljungbox(X.dropna(), lags=[lags], return_df=False)
    if isinstance(lb_res, tuple):
        lb_stat_arr, lb_p_arr = lb_res
    else:
        # DataFrame with columns ['lb_stat','lb_pvalue']
        lb_stat_arr = lb_res['lb_stat'].to_numpy()
        lb_p_arr = lb_res['lb_pvalue'].to_numpy()
    print(f"Ljung-Box test (lag={lags}): stat={lb_stat_arr[0]:.3f}, p-value={lb_p_arr[0]:.3f}")
    print(f"Sample ACF (first 5 lags): {np.round(acf_vals[1:6], 3).tolist()}")


if __name__ == "__main__":
    # Example usage with random noise and sine wave + noise
    import sys
    if len(sys.argv) > 1:
        # load a CSV with a single column named 'value'
        df = pd.read_csv(sys.argv[1], parse_dates=True, index_col=0)
        series = df.iloc[:,0]
    else:
        # synthetic example
        t = np.linspace(0, 1, 500)
        series = pd.Series(np.sin(2 * np.pi * 5 * t) + 0.5 * np.random.randn(len(t)))
    diagnose_series(series)

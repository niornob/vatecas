import numpy as np
import pandas as pd
from arch import arch_model
from typing import Optional, Literal


def garch_pq(
    X: pd.Series,
    p: int = 1,
    q: int = 1,
    mean: Literal[
        "Constant", "Zero", "LS", "AR", "ARX", "HAR", "HARX", "constant", "zero"
    ] = "zero",
    vol: Literal["GARCH", "ARCH", "EGARCH", "FIGARCH", "APARCH", "HARCH"] = "GARCH",
    dist: Literal[
        "normal",
        "gaussian",
        "t",
        "studentst",
        "skewstudent",
        "skewt",
        "ged",
        "generalized error",
    ] = "normal",
) -> float:
    """
    Fit a GARCH(p,q) model on the entire input series and return the next one-step-ahead volatility forecast.

    This function takes a historical time series, fits a GARCH model to all available data,
    and produces a single volatility forecast for the next time period.

    Parameters
    ----------
    X : pd.Series
        A univariate time series (e.g. returns), indexed by time.
        This represents the complete historical data available for fitting.
    p : int, optional
        The GARCH lag order (default 1).
    q : int, optional
        The ARCH lag order (default 1).
    mean : Literal
        How the mean of X evolves over time.
    vol : Literal
        GARCH model variant to use.
    dist : Literal
        Distribution assumption for the innovations.

    Returns
    -------
    float
        The predicted volatility (conditional standard deviation) for the next time period
        after the last observation in X.
    """
    # Check if we have enough data to fit the model
    min_required = max(p, q) + 1
    if len(X) < min_required:
        raise ValueError(
            f"Need at least {min_required} observations to fit GARCH({p},{q}), got {len(X)}"
        )

    # Define and fit the GARCH model using all available data
    # This estimates parameters using the entire historical series
    am = arch_model(X, mean=mean, vol=vol, p=p, q=q, dist=dist)
    res = am.fit(disp="off")

    # Generate one-step-ahead forecast of variance
    # This uses the fitted model to predict volatility for the next period
    fc = res.forecast(horizon=1, reindex=False)
    var_forecast = fc.variance.values[-1, 0]

    # Return the standard deviation (volatility) rather than variance
    return np.sqrt(var_forecast)


def garch_series(
    X: pd.Series,
    p: int = 1,
    q: int = 1,
    mean: Literal[
        "Constant", "Zero", "LS", "AR", "ARX", "HAR", "HARX", "constant", "zero"
    ] = "zero",
    vol: Literal["GARCH", "ARCH", "EGARCH", "FIGARCH", "APARCH", "HARCH"] = "GARCH",
    dist: Literal[
        "normal",
        "gaussian",
        "t",
        "studentst",
        "skewstudent",
        "skewt",
        "ged",
        "generalized error",
    ] = "normal",
    min_obs: Optional[int] = None,
) -> pd.Series:
    """
    Compute recursive one-step-ahead GARCH volatility forecasts for an entire time series.

    This function implements a rolling window approach where at each time point i,
    it uses all data up to (but not including) time i to forecast volatility at time i.
    This simulates real-time forecasting where you only know the past when making predictions.

    Parameters
    ----------
    X : pd.Series
        A univariate time series (e.g. returns), indexed by time.
    p : int, optional
        The GARCH lag order (default 1).
    q : int, optional
        The ARCH lag order (default 1).
    mean : Literal
        How the mean of X evolves over time.
    vol : Literal
        GARCH model variant to use.
    dist : Literal
        Distribution assumption for the innovations.
    min_obs : int, optional
        Minimum number of observations before starting forecasts.
        If None, defaults to max(p, q) + 1.

    Returns
    -------
    Y : pd.Series
        Series with the same index as X. Y.loc[i] contains the GARCH volatility forecast
        for time i, computed using only data X.loc[:i-1] (i.e., all data before time i).
        The first `min_obs` entries are NaN since insufficient data is available.
    """
    # Determine minimum observations needed - we need enough data to estimate all parameters
    if min_obs is None:
        min_obs = max(p, q) + 1

    # Initialize output series with same index as input, filled with NaN
    Y = pd.Series(index=X.index, dtype=float)

    # Rolling window approach: for each time point, use only historical data to forecast
    for idx in range(min_obs, len(X)):
        # Create training slice: all data up to (but not including) current time point
        # This ensures we're only using information that would have been available
        # at the time of making the forecast
        train_slice = X.iloc[:idx]

        try:
            # Use our single-prediction function to get volatility forecast
            # This applies GARCH model to historical data and predicts next period
            volatility_forecast = garch_pq(
                train_slice, p=p, q=q, mean=mean, vol=vol, dist=dist
            )

            # Store the forecast at the current time index
            Y.iloc[idx] = volatility_forecast

        except (ValueError, Exception):
            # If model fitting fails (e.g., convergence issues), leave as NaN
            # This can happen with difficult-to-fit data or numerical instabilities
            Y.iloc[idx] = np.nan

    return Y

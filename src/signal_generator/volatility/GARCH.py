from typing import Dict, Literal, Optional, Tuple, cast

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from .GARCHpq import garch_pq


def GARCH(
    data: Dict[str, pd.Series], params: Optional[Dict[str, object]] = None
) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Predict variance structure using GARCH models and PCA.

    This method implements a sophisticated variance forecasting approach:
    1. Convert price series to returns
    2. Fit individual GARCH models for each asset using our modular garch_pq function
    3. Perform PCA to identify the dominant market factor
    4. Forecast next-period covariance matrix

    The key improvement here is that we leverage our specialized garch_pq function
    for all GARCH modeling, ensuring consistency and reducing code duplication.

    Args:
        data: Dictionary mapping ticker symbols to historical price series
        params: Optional dictionary of GARCH model parameters

    Returns:
        Tuple of (asset_covariance_matrix, pc1_variance, pc1_loadings)
    """
    # Set default parameters - these will be passed to our garch_pq function
    params = params or {
        "p": 1,
        "q": 1,
        "distribution": "normal",
        "rescale": True,
        "mean": "zero",
    }

    tickers = list(data.keys())
    n_assets = len(tickers)

    # Convert price series to returns (percentage changes)
    # This step remains the same as it's preparing data for our analysis
    returns_data = {}
    for ticker, prices in data.items():
        # Calculate percentage returns, handling potential division by zero
        pct_returns = prices.pct_change().dropna()
        if len(pct_returns) < 10:  # Need minimum data for GARCH
            # Fall back to simple rolling volatility if insufficient data
            pct_returns = prices.diff().dropna() / prices.shift(1).dropna()
        returns_data[ticker] = pct_returns * 100  # Scale to percentage points

    # Create returns matrix for PCA and correlation analysis
    returns_df = pd.DataFrame(returns_data).dropna()

    if len(returns_df) < 10:
        # Insufficient data - return simple variance estimates
        # This fallback mechanism protects against edge cases with very little data
        simple_vars = np.var(returns_df.values, axis=0, ddof=1)
        simple_cov = np.cov(returns_df.T)
        return (
            simple_cov,
            np.mean(simple_vars),
            np.ones(n_assets) / np.sqrt(n_assets),
        )

    # Fit GARCH models for each asset using our modular garch_pq function
    garch_forecasts = {}

    for ticker in tickers:
        try:
            returns_series = returns_data[ticker]

            # Extract parameters and convert them to the format expected by garch_pq
            # We need to handle the parameter mapping carefully since the original
            # function used slightly different parameter names
            garch_volatility = garch_pq(
                X=returns_series,
                p=cast(int, params.get("p", 1)),
                q=cast(int, params.get("q", 1)),
                mean=cast(
                    Literal[
                        "Constant",
                        "Zero",
                        "LS",
                        "AR",
                        "ARX",
                        "HAR",
                        "HARX",
                        "constant",
                        "zero",
                    ],
                    params.get("mean", "zero"),
                ),
                vol=cast(
                    Literal["GARCH", "ARCH", "EGARCH", "FIGARCH", "APARCH", "HARCH"],
                    params.get("vol", "GARCH"),
                ),
                dist=cast(
                    Literal[
                        "normal",
                        "gaussian",
                        "t",
                        "studentst",
                        "skewstudent",
                        "skewt",
                        "ged",
                        "generalized error",
                    ],
                    params.get("distribution", "normal"),
                ),
            )

            # Convert volatility to variance for consistency with the rest of the function
            garch_forecasts[ticker] = garch_volatility**2

        except Exception as e:
            # Robust fallback mechanism: if our GARCH modeling fails for any reason,
            # we fall back to a simple rolling volatility estimate
            # This ensures the function remains reliable even with problematic data
            print(
                f"GARCH modeling failed for {ticker}: {e}. Using rolling volatility fallback."
            )

            rolling_vol = (
                returns_data[ticker]
                .rolling(window=min(20, len(returns_data[ticker])))
                .std()
                .iloc[-1]
            )
            garch_forecasts[ticker] = rolling_vol**2

    # The PCA analysis and covariance matrix construction remain unchanged
    # This part of the function was already well-structured and doesn't need modification

    # Perform PCA on returns to identify principal components
    # This captures the common factors driving asset price movements
    pca = PCA(n_components=min(n_assets, len(returns_df)))
    pca_fit = pca.fit(returns_df)

    # Extract first principal component information
    # The first PC usually represents the broad market factor
    pc1_loadings = pca_fit.components_[0]  # Loadings on first PC
    pc1_variance = pca_fit.explained_variance_[0]  # Variance explained by first PC

    # Construct covariance matrix combining GARCH forecasts with correlation structure
    # This is a sophisticated approach that maintains the observed correlation patterns
    # while updating the volatility forecasts based on our GARCH models
    correlation_matrix = returns_df.corr().values

    # Create diagonal matrix of GARCH volatility forecasts
    # We extract the square root since garch_forecasts contains variances
    garch_vols = np.array([np.sqrt(garch_forecasts[ticker]) for ticker in tickers])
    vol_matrix = np.outer(garch_vols, garch_vols)

    # Combine correlation structure with updated volatilities
    # This gives us a forward-looking covariance matrix that respects both
    # the correlation structure and the GARCH volatility forecasts
    forecasted_covariance = correlation_matrix * vol_matrix

    # Ensure positive definiteness for numerical stability
    # This mathematical adjustment prevents numerical issues that can arise
    # when using the covariance matrix in optimization or simulation
    eigenvals, eigenvecs = np.linalg.eigh(forecasted_covariance)
    eigenvals = np.maximum(
        eigenvals, 1e-8
    )  # Floor eigenvalues to ensure positive definiteness
    forecasted_covariance = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T

    return forecasted_covariance, pc1_variance, pc1_loadings

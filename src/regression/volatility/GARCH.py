from typing import Optional, Dict, List, Tuple, cast, Literal
import pandas as pd
import numpy as np
from arch import arch_model
from sklearn.decomposition import PCA
import warnings



def GARCH(
        data: Dict[str, pd.Series],
        params: Optional[Dict[str, object]] = None
    ) -> Tuple[np.ndarray, float, np.ndarray]:
        """
        Predict variance structure using GARCH models and PCA.

        This method implements a sophisticated variance forecasting approach:
        1. Convert price series to returns
        2. Fit individual GARCH models for each asset
        3. Perform PCA to identify the dominant market factor
        4. Forecast next-period covariance matrix

        Args:
            data: Dictionary mapping ticker symbols to historical price series

        Returns:
            Tuple of (asset_covariance_matrix, pc1_variance, pc1_loadings)
        """
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
            simple_vars = np.var(returns_df.values, axis=0, ddof=1)
            simple_cov = np.cov(returns_df.T)
            return (
                simple_cov,
                np.mean(simple_vars),
                np.ones(n_assets) / np.sqrt(n_assets),
            )

        # Fit GARCH models for each asset to get volatility forecasts
        garch_forecasts = {}

        for ticker in tickers:
            try:
                returns_series = returns_data[ticker]

                # Check if we have a cached model for this ticker
                model_key = f"{ticker}_{len(returns_series)}"

                
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")

                    # GARCH(1,1) is most common and usually sufficient
                    garch_model = arch_model(
                        returns_series,
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
                        rescale=cast(bool, params.get("rescale", True)),
                    )

                    fitted_model = garch_model.fit(disp="off", show_warning=False)

                # Forecast next-period variance
                forecast = fitted_model.forecast(horizon=1, reindex=False)
                next_var = cast(float, forecast.variance.iloc[-1, 0])
                garch_forecasts[ticker] = next_var

            except Exception as e:
                # Fall back to rolling volatility if GARCH fails
                rolling_vol = (
                    returns_data[ticker]
                    .rolling(window=min(20, len(returns_data[ticker])))
                    .std()
                    .iloc[-1]
                )
                garch_forecasts[ticker] = rolling_vol**2

        # Perform PCA on returns to identify principal components
        pca = PCA(n_components=min(n_assets, len(returns_df)))
        pca_fit = pca.fit(returns_df)

        # Extract first principal component information
        pc1_loadings = pca_fit.components_[0]  # Loadings on first PC
        pc1_variance = pca_fit.explained_variance_[0]  # Variance explained by first PC

        # Construct covariance matrix combining GARCH forecasts with correlation structure
        # This maintains the correlation structure while updating volatility forecasts
        correlation_matrix = returns_df.corr().values

        # Create diagonal matrix of GARCH volatility forecasts
        garch_vols = np.array([np.sqrt(garch_forecasts[ticker]) for ticker in tickers])
        vol_matrix = np.outer(garch_vols, garch_vols)

        # Combine correlation structure with updated volatilities
        forecasted_covariance = correlation_matrix * vol_matrix

        # Ensure positive definiteness (numerical stability)
        eigenvals, eigenvecs = np.linalg.eigh(forecasted_covariance)
        eigenvals = np.maximum(
            eigenvals, 1e-8
        )  # Floor eigenvalues to ensure positive definiteness
        forecasted_covariance = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T

        return forecasted_covariance, pc1_variance, pc1_loadings
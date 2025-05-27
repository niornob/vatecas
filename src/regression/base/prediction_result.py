import numpy as np

class PredictionResult:
    """
    Container for prediction results that includes both mean forecasts and covariance structure.

    This class encapsulates the dual nature of our enhanced predictions: point estimates
    for asset prices and uncertainty quantification through covariance matrices.
    """

    def __init__(
        self,
        predictions: np.ndarray,
        asset_covariance: np.ndarray,
        pc1_variance: float,
        pc1_loadings: np.ndarray,
    ):
        """
        Initialize prediction result container.

        Args:
            predictions: Array of predicted values for each asset
            asset_covariance: Covariance matrix between assets (n_assets x n_assets)
            pc1_variance: Variance along first principal component (market factor)
            pc1_loadings: Loadings of each asset on first principal component
        """
        self.predictions = predictions
        self.asset_covariance = asset_covariance
        self.pc1_variance = pc1_variance
        self.pc1_loadings = pc1_loadings

    @property
    def asset_volatilities(self) -> np.ndarray:
        """Extract individual asset volatilities from covariance matrix."""
        return np.sqrt(np.diag(self.asset_covariance))

    @property
    def asset_correlations(self) -> np.ndarray:
        """Compute correlation matrix from covariance matrix."""
        vol_matrix = np.outer(self.asset_volatilities, self.asset_volatilities)
        return self.asset_covariance / vol_matrix
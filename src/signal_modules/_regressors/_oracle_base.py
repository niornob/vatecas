from abc import ABC, abstractmethod
from typing import Optional, Dict, List, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import warnings


class Oracle(ABC):
    """
    Abstract base class for time series prediction models (oracles).

    An oracle represents any predictive model that can generate forecasts
    based on historical time series data within a specified window.
    """

    def __init__(
        self,
        name: str = "",
        version: str = "0.0",
        params: Optional[Dict[str, object]] = None,
    ):
        """
        Initialize an Oracle instance.

        Args:
            name: Human-readable name for the oracle
            version: Version identifier for the oracle
            params: Dictionary of parameters specific to the oracle implementation
        """
        self.name = name or self.__class__.__name__
        self.version = version
        self.params = params or {}

    @abstractmethod
    def _predict(self, data: Dict[str, pd.Series]) -> np.ndarray:
        """
        Core prediction method to be implemented by subclasses.

        Args:
            data: Dictionary mapping ticker symbols to historical price series

        Returns:
            Array of predictions for each ticker in the same order as data.keys()
        """
        pass

    def regress(
        self,
        data: Dict[str, pd.Series],
        window: int,
        extrapolate: bool = False,
    ) -> Dict[str, pd.Series]:
        """
        Perform rolling window regression on time series data.

        This method steps through time, using only historical data within
        the specified window to predict each subsequent point.

        Args:
            data: Dictionary mapping ticker symbols to historical price series
            window: Number of historical points to use for each prediction
            extrapolate: If True, generate one additional prediction beyond the data

        Returns:
            Dictionary mapping ticker symbols to predicted price series

        Raises:
            ValueError: If data is empty, window is invalid, or data series have mismatched lengths
        """
        # Input validation
        if not data:
            raise ValueError("Input data cannot be empty")

        # Convert to DataFrame for easier manipulation
        try:
            data_df = pd.DataFrame(data)
        except ValueError as e:
            raise ValueError(f"Failed to convert input data to DataFrame: {e}")

        data_len = len(data_df.index)
        if data_len == 0:
            raise ValueError("Input data series are empty")

        if window < 1:
            raise ValueError("Window size must be at least 1")
        if window > data_len:
            raise ValueError(
                f"Window size ({window}) cannot exceed data length ({data_len})"
            )

        tickers = list(data_df.columns)
        pred_len = data_len + (1 if extrapolate else 0)

        # Initialize prediction DataFrame
        pred = pd.DataFrame(
            np.zeros((pred_len, len(tickers))), columns=tickers, index=range(pred_len)
        )

        # Set initial condition (first prediction equals first observation)
        pred.iloc[0] = data_df.iloc[0]

        # Rolling window prediction
        for i in range(1, pred_len):
            # Extract historical window, ensuring we don't go before the start
            window_start = max(0, i - window)
            sliced_df = data_df.iloc[window_start:i]

            # Convert to dictionary format expected by _predict
            history = {ticker: sliced_df[ticker] for ticker in tickers}

            try:
                prediction = self._predict(data=history)
                if len(prediction) != len(tickers):
                    raise ValueError(
                        f"Prediction length {len(prediction)} doesn't match ticker count {len(tickers)}"
                    )
                pred.iloc[i] = prediction
            except Exception as e:
                raise RuntimeError(f"Prediction failed at step {i}: {e}")

        return {ticker: pred[ticker] for ticker in tickers}


class StackedOracle(Oracle):
    """
    A sophisticated ensemble oracle that combines multiple oracles using residual-based stacking.

    This implementation uses a sequential approach where each oracle attempts to predict
    the residuals (unexplained patterns) left by previous oracles. The final prediction
    combines all individual oracle contributions.
    """

    def __init__(self, oracles: List[Oracle], windows: List[int], weights: List[float]):
        """
        Initialize a StackedOracle with multiple component oracles.

        Args:
            oracles: List of oracle instances to stack
            windows: List of window sizes, one for each oracle
            weights: List of weights to scale respective oracles

        Raises:
            ValueError: If oracles and windows lists have different lengths
        """
        if len(set([len(oracles), len(windows), len(weights)])) != 1:
            raise ValueError(
                f"Number of oracles ({len(oracles)}) must equal number of windows ({len(windows)}) and weights ({len(weights)})"
            )

        if not oracles:
            raise ValueError("At least one oracle must be provided")

        if any(w < 1 for w in windows):
            raise ValueError("All window sizes must be positive")

        if any(w < 0 for w in weights):
            raise ValueError("Weights must be positive.")

        super().__init__(name=f"StackedOracle_{len(oracles)}", version="1.0")

        self.oracles = oracles
        self.windows = windows
        self.weights = weights

        # Initialize state tracking attributes
        self.residuals: List[Dict[str, pd.Series]] = []
        self.approx_residuals: List[Dict[str, pd.Series]] = []
        self._last_regression_data = None

    def regress(
        self, data: Dict[str, pd.Series], extrapolate: bool = False
    ) -> Dict[str, pd.Series]:
        """
        Perform stacked regression using all component oracles.

        Args:
            data: Dictionary mapping ticker symbols to historical price series
            extrapolate: If True, generate predictions beyond the historical data

        Returns:
            Dictionary mapping ticker symbols to combined predicted series
        """
        # Input validation
        if not data:
            raise ValueError("Input data cannot be empty")

        tickers = list(data.keys())

        # Validate that all series have the same length
        series_lengths = [len(series) for series in data.values()]
        if len(set(series_lengths)) > 1:
            raise ValueError("All input series must have the same length")

        # Clear previous state and initialize tracking
        self.residuals = []
        self.approx_residuals = []
        predictions: List[np.ndarray] = []

        # Store reference to input data for diagnostics
        self._last_regression_data = data.copy()

        # Initialize with original data as first residual
        self.residuals.append({tk: s.reset_index(drop=True) for tk, s in data.items()})

        # Sequential oracle processing
        for oracle_idx, (oracle, window) in enumerate(zip(self.oracles, self.windows)):
            try:
                # Get predictions from current oracle on current residuals
                pred = oracle.regress(
                    self.residuals[-1], window=window, extrapolate=extrapolate
                )

                # Handle extrapolation predictions separately
                if extrapolate:
                    # Extract the extrapolated values for final combination
                    extrapolated_values = np.array(
                        [pred[tk].iloc[-1] for tk in tickers]
                    )
                    predictions.append(extrapolated_values)

                    # Remove extrapolated values for residual calculation
                    pred = {tk: pred[tk].iloc[:-1] for tk in tickers}

                # Store this oracle's predictions
                self.approx_residuals.append(pred)

                # Calculate new residuals for next oracle
                new_residuals = {}
                for ticker in tickers:
                    try:
                        new_residuals[ticker] = (
                            self.residuals[-1][ticker] - pred[ticker]
                        )
                    except Exception as e:
                        raise RuntimeError(
                            f"Failed to calculate residuals for {ticker}: {e}"
                        )

                self.residuals.append(new_residuals)

            except Exception as e:
                raise RuntimeError(f"Oracle {oracle_idx} ({oracle.name}) failed: {e}")

        # Combine all oracle predictions
        net_regression = {}
        for ticker in tickers:
            try:
                # Sum all predictions from individual oracles
                oracle_contributions = [w * pred[ticker] for w, pred in zip(self.weights, self.approx_residuals)]
                net_regression[ticker] = np.sum(oracle_contributions, axis=0)

                # Convert back to pandas Series with proper indexing
                net_regression[ticker] = pd.Series(
                    net_regression[ticker], index=range(len(net_regression[ticker]))
                )
            except Exception as e:
                raise RuntimeError(f"Failed to combine predictions for {ticker}: {e}")

        # Add extrapolated predictions if requested
        if extrapolate and predictions:
            try:
                net_prediction = np.sum([w * series for w, series in zip(self.weights, predictions)], axis=0)
                for ticker, next_val in zip(tickers, net_prediction):
                    # Extend the series with the extrapolated value
                    current_length = len(net_regression[ticker])
                    net_regression[ticker].loc[current_length] = next_val
            except Exception as e:
                raise RuntimeError(f"Failed to add extrapolated predictions: {e}")

        return net_regression

    def _predict(self, data: Dict[str, pd.Series]) -> np.ndarray:
        """
        Generate single-step predictions using the stacked oracle approach.

        Args:
            data: Historical data for prediction

        Returns:
            Array of predictions for each ticker
        """
        pred = self.regress(data=data, extrapolate=True)
        try:
            return np.array([series.iloc[-1] for series in pred.values()])
        except Exception as e:
            raise RuntimeError(f"Failed to extract predictions: {e}")

    def diagnostics(self, data: Dict[str, pd.Series]) -> None:
        """
        Generate comprehensive diagnostic plots and metrics for the stacked oracle.

        This method analyzes how well each oracle in the stack performs and whether
        the residuals approach white noise (indicating good model fit).
        
        Creates a separate 2x2 plot for each ticker showing:
        - Original vs Predicted values
        - Residual evolution through oracle stages  
        - Cumulative oracle contributions
        - Final residual white noise analysis

        Args:
            data: Historical data to analyze
        """
        # Perform regression to populate internal state
        predictions = self.regress(data, extrapolate=False)

        if not self.residuals or not self.approx_residuals:
            raise RuntimeError("No regression data available. Run regress() first.")

        tickers = list(data.keys())
        
        # Create separate plot for each ticker with 2x2 grid
        for ticker_idx, ticker in enumerate(tickers):
            # Create individual figure for this ticker with 2x2 subplot arrangement
            fig, axes = plt.subplots(2, 2, figsize=(12, 6))
            fig.suptitle(f'Diagnostic Analysis for {ticker}', fontsize=16, fontweight='bold')
            
            # Generate all four diagnostic plots for current ticker
            self._plot_ticker_diagnostics(ticker, axes, predictions)
            
            # Adjust spacing between subplots for better readability
            plt.tight_layout()
            plt.subplots_adjust(top=0.93)  # Make room for the main title
            plt.show()

        # Print performance metrics after all plots
        self._print_performance_metrics(data, predictions, tickers)

    def _plot_ticker_diagnostics(
        self,
        ticker: str,
        axes: np.ndarray,
        predictions: Dict[str, pd.Series],
    ) -> None:
        """
        Generate diagnostic plots for a specific ticker in a 2x2 grid.
        
        Args:
            ticker: The ticker symbol to analyze
            axes: 2x2 numpy array of matplotlib axes objects
            predictions: Dictionary mapping tickers to predicted series
        """
        if self._last_regression_data is not None:
            original_data = self._last_regression_data[ticker]
        else:
            raise RuntimeError("No regression data available. Run regress() first.")
        
        original_idx = original_data.index

        # Plot 1 (Top-left): Original vs Predicted
        axes[0, 0].plot(original_data, label="Original", alpha=0.8, linewidth=2)
        axes[0, 0].plot(
            original_idx, predictions[ticker], label="Predicted", alpha=0.8, linewidth=2
        )
        axes[0, 0].set_title("Original vs Predicted", fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_xlabel("Time Index")
        axes[0, 0].set_ylabel("Value")

        # Plot 2 (Top-right): Residual Evolution
        for i, residual_dict in enumerate(self.residuals[1:], 1):
            residual_series = residual_dict[ticker]
            axes[0, 1].plot(
                original_idx, residual_series, alpha=0.7, label=f"After Oracle {i}", linewidth=1.5
            )
        axes[0, 1].set_title("Residual Evolution", fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_xlabel("Time Index")
        axes[0, 1].set_ylabel("Residual Value")

        # Plot 3 (Bottom-left): Oracle Contributions
        cumulative_pred = pd.Series(
            np.zeros(len(original_data))
        )
        for i, pred_dict in enumerate(self.approx_residuals):
            oracle_contrib = self.weights[i] * pred_dict[ticker]  # Apply weights for accurate visualization
            cumulative_pred += oracle_contrib
            #print(oracle_contrib, "\n", cumulative_pred)
            axes[1, 0].plot(
                original_idx, cumulative_pred, alpha=0.8, label=f"Up to Oracle {i+1}", linewidth=1.5
            )
        axes[1, 0].set_title("Cumulative Oracle Contributions", fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_xlabel("Time Index")
        axes[1, 0].set_ylabel("Cumulative Prediction")

        # Plot 4 (Bottom-right): Final Residual Analysis (White Noise Test)
        final_residuals = self.residuals[-1][ticker]
        
        # Scatter plot of residuals
        axes[1, 1].scatter(
            range(len(final_residuals)), final_residuals, alpha=0.6, s=20
        )
        axes[1, 1].axhline(y=0, color="r", linestyle="--", alpha=0.7, linewidth=2)
        
        # Add horizontal lines at ±2 standard deviations for reference
        residual_std = np.std(final_residuals)
        axes[1, 1].axhline(y=2*residual_std, color="orange", linestyle=":", alpha=0.5, label="±2σ")
        axes[1, 1].axhline(y=-2*residual_std, color="orange", linestyle=":", alpha=0.5)
        
        axes[1, 1].set_title("Final Residuals (White Noise Check)", fontweight='bold')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_xlabel("Time Index")
        axes[1, 1].set_ylabel("Final Residual Value")

    def _print_performance_metrics(
        self,
        original_data: Dict[str, pd.Series],
        predictions: Dict[str, pd.Series],
        tickers: List[str],
    ) -> None:
        """Print comprehensive performance metrics."""
        print("=" * 80)
        print("STACKED ORACLE PERFORMANCE ANALYSIS")
        print("=" * 80)

        for ticker in tickers:
            orig = original_data[ticker].reset_index(drop=True)
            pred = predictions[ticker].reset_index(drop=True)

            # print(orig, "\n", pred)

            # Calculate metrics
            mse = np.mean((orig - pred) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(orig - pred))

            # R-squared
            ss_res = np.sum((orig - pred) ** 2)
            ss_tot = np.sum((orig.to_numpy() - np.mean(orig)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

            print(f"\n{ticker} Performance Metrics:")
            print(f"  RMSE: {rmse:.6f}")
            print(f"  MAE:  {mae:.6f}")
            print(f"  R²:   {r_squared:.6f}")

            # White noise tests for final residuals
            final_residuals = self.residuals[-1][ticker]
            self._analyze_residual_properties(ticker, final_residuals)

    def _analyze_residual_properties(self, ticker: str, residuals: pd.Series) -> None:
        """Analyze statistical properties of residuals to check for white noise."""
        # Ljung-Box test for autocorrelation (white noise should have no correlation)
        try:
            from statsmodels.stats.diagnostic import acorr_ljungbox

            lb = acorr_ljungbox(
                residuals, lags=min(40, len(residuals) // 4), return_df=False
            )
            lb_pvalue = lb["lb_pvalue"]
            print(
                "Ljung-Box Test (<0.05 => autocorr.):", [round(x,3) for x in lb_pvalue]
            )
        except ImportError:
            # Fallback to simple autocorrelation check
            autocorr = residuals.autocorr(lag=1)
            autocorr_result = "PASS" if abs(autocorr) < 0.1 else "FAIL"
            print(f"  Lag-1 Autocorrelation: {autocorr_result} ({autocorr:.3f})")

            # Normality test
        try:
            _, shapiro_p = stats.shapiro(residuals)
            normality_result = "PASS" if shapiro_p > 0.05 else "FAIL"
            print(f"  Shapiro-Wilk (Normality): {normality_result} (p-value: {shapiro_p:.3f})")
        except:
            print(f"  Shapiro-Wilk test failed (sample size issue)")

        # Mean and variance
        residual_mean = np.mean(residuals)
        residual_std = np.std(residuals)
        print(f"  Residual Mean: {residual_mean:.6f}")
        print(f"  Residual Std:  {residual_std:.6f}")

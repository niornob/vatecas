from abc import ABC, abstractmethod
from typing import Optional, Dict, List, Tuple, cast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import warnings
from tqdm import tqdm
from scipy.stats import pearsonr

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parents[2]))
from utils.denoise import wavelet_denoise


class Oracle(ABC):
    """
    Abstract base class for time series prediction models (oracles).

    An oracle represents any predictive model that can generate forecasts
    based on historical time series data within a specified window.
    """

    def __init__(
        self,
        name: str = "generic_oracle_name",
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

        tickers = list(data_df.columns)
        pred_len = data_len + (1 if extrapolate else 0)

        # Initialize prediction DataFrame
        pred = pd.DataFrame(
            np.zeros((pred_len, len(tickers))), columns=tickers, index=range(pred_len)
        )
        if not extrapolate:
            pred.index = data_df.index

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

        return {
            ticker: wavelet_denoise(
                pred[ticker], wavelet="db20", level=None, threshold_method="soft"
            )
            for ticker in tickers
        }

    def plot_vs_actual(self, data: dict[str, pd.Series], window: int) -> None:
        """Generate comparison plots for actual vs predicted values."""
        actual_df = pd.DataFrame(data)
        preds_df = pd.DataFrame(self.regress(data, window=window))
        dates = actual_df.index
        tickers = actual_df.columns
        
        for ticker in tickers:
            # Create a figure with subplots for comprehensive analysis
            # Layout: Two full-width plots on top, then two square plots side by side at bottom
            fig = plt.figure(figsize=(14, 14))
            gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1], hspace=0.3, wspace=0.3)
            
            # Top plot spans both columns - cumulative returns
            ax1 = fig.add_subplot(gs[0, :])
            # Middle plot spans both columns - percentage returns over time
            ax2 = fig.add_subplot(gs[1, :])
            # Bottom plots are square and side by side
            ax3 = fig.add_subplot(gs[2, 0])
            ax4 = fig.add_subplot(gs[2, 1])
            
            fig.suptitle(f'Kalman Filter Analysis: {ticker}', fontsize=16, fontweight='bold')
            
            # ===== PLOT 1: Original cumulative returns comparison =====
            # This plot spans the full width at the top, giving us space to see the time series clearly
            ax1.plot(
                dates,
                actual_df[ticker],
                label=f"Actual {ticker}",
                linewidth=2,
                alpha=0.8,
            )
            ax1.plot(
                dates,
                preds_df[ticker],
                label="1-Step Kalman Forecast",
                linestyle="--",
                linewidth=1.5,
                alpha=0.7,
            )
            ax1.set_title("One-Step-Ahead Forecast vs Actual (Cumulative Returns)", 
                        fontsize=12, fontweight="bold")
            ax1.set_xlabel("Time", fontsize=10)
            ax1.set_ylabel(f"{ticker} Cumulative Return", fontsize=10)
            ax1.legend(fontsize=10)
            ax1.grid(True, alpha=0.3)
            
            # ===== PLOT 2: Percentage returns over time =====
            # This plot shows period-to-period changes, making it easy to spot when predictions diverge
            
            # Calculate percentage changes from cumulative returns
            actual_pct_changes = actual_df[ticker].diff().dropna()
            pred_pct_changes = preds_df[ticker].diff().dropna()
            
            # Align the series and get common dates for time series plotting
            common_index = actual_pct_changes.index.intersection(list(pred_pct_changes.index))
            actual_aligned = actual_pct_changes.loc[common_index]
            pred_aligned = pred_pct_changes.loc[common_index]
            common_dates = common_index
            
            # Plot the percentage changes over time
            ax2.fill_between(
                common_dates,
                actual_aligned,
                label=f"Actual {ticker} % Change",
                alpha=0.5,
                color='blue',
                edgecolor='none'
            )
            ax2.fill_between(
                common_dates,
                pred_aligned,
                label="Predicted % Change",
                alpha=0.5,
                color='red',
                edgecolor='none'
            )

            # Add zero line for reference (helps identify positive vs negative periods)
            ax2.axhline(y=0, color='black', linestyle='-', alpha=1, linewidth=1.5)
            
            # Highlight periods where signs disagree by shading background
            sign_disagreement = np.sign(actual_aligned) != np.sign(pred_aligned)
            if sign_disagreement.any():
                # Create background shading for periods of directional disagreement
                for i, disagreement in enumerate(sign_disagreement):
                    if disagreement and i < len(common_dates):
                        # Shade the area around disagreement periods
                        ax2.axvspan(common_dates[i], common_dates[i], 
                                alpha=0.2, color='yellow', linewidth=0)
            
            ax2.set_title("Percentage Returns Over Time: Identifying Prediction Alignment", 
                        fontsize=12, fontweight="bold")
            ax2.set_xlabel("Time", fontsize=10)
            ax2.set_ylabel("Percentage Change", fontsize=10)
            ax2.legend(fontsize=10)
            ax2.grid(True, alpha=0.3)
            
            # ===== PLOT 3: Percentage changes scatter plot =====
            # This square plot on the bottom left focuses on magnitude accuracy
            
            # Create scatter plot
            ax3.scatter(actual_aligned, pred_aligned, alpha=0.6, s=30)
            m, b = np.polyfit(actual_aligned, pred_aligned, deg=1)
            ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
            ax3.plot(actual_aligned, m*actual_aligned + b, color='blue', label='OLS Fit')

            # Calculate and display correlation
            if len(actual_aligned) > 1:
                correlation, p_value = pearsonr(actual_aligned, pred_aligned)
                ax3.text(0.05, 0.95, f'Correlation: {correlation:.3f}\np-value: {p_value:.3f}', 
                        transform=ax3.transAxes, fontsize=11, 
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8),
                        verticalalignment='top')
            
            # Add diagonal reference line (perfect prediction line)
            min_val = min(cast(float, actual_aligned.min()), cast(float, pred_aligned.min()))
            max_val = max(cast(float, actual_aligned.max()), cast(float, pred_aligned.max()))
            ax3.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, linewidth=1)
            
            ax3.set_title("Percentage Changes: Actual vs Predicted", fontsize=12, fontweight="bold")
            ax3.set_xlabel(f"Actual {ticker} % Change", fontsize=10)
            ax3.set_ylabel("Predicted % Change", fontsize=10)
            ax3.grid(True, alpha=0.3)
            
            # ===== PLOT 4: Sign correlation analysis =====
            # This square plot on the bottom right shows directional accuracy through bubble sizes
            
            # Convert percentage changes to signs (-1, 0, 1)
            actual_signs = np.sign(actual_aligned)
            pred_signs = np.sign(pred_aligned)
            
            # Count occurrences at each of the four possible sign combinations
            sign_combinations = {}
            for actual_sign in [-1, 1]:
                for pred_sign in [-1, 1]:
                    count = np.sum((actual_signs == actual_sign) & (pred_signs == pred_sign))
                    sign_combinations[(actual_sign, pred_sign)] = count
            
            # Calculate total for percentage calculations
            total_points = len(actual_signs)
            
            # Create the bubble plot
            # We'll position the bubbles at the four corners of a unit square
            positions = {(-1, -1): (-0.8, -0.8), (-1, 1): (-0.8, 0.8), 
                        (1, -1): (0.8, -0.8), (1, 1): (0.8, 0.8)}
            
            # Calculate bubble sizes (scale them for visibility)
            max_count = max(sign_combinations.values()) if sign_combinations.values() else 1
            
            for (actual_sign, pred_sign), count in sign_combinations.items():
                x, y = positions[(actual_sign, pred_sign)]
                # Scale bubble size: minimum size of 100, maximum proportional to count
                bubble_size = 100 + (count / max_count) * 800 if max_count > 0 else 100
                percentage = (count / total_points) * 100 if total_points > 0 else 0
                
                # Choose color based on whether signs agree
                color = 'green' if actual_sign == pred_sign else 'red'
                
                ax4.scatter(x, y, s=bubble_size, alpha=0.6, c=color)
                
                # Add count and percentage labels
                ax4.annotate(f'{count}\n({percentage:.1f}%)', 
                            (x, y), ha='center', va='center', 
                            fontsize=10, fontweight='bold')
            
            # Calculate sign correlation (this is essentially measuring directional accuracy)
            sign_correlation, sign_p_value = pearsonr(actual_signs, pred_signs) if len(actual_signs) > 1 else (0, 1)
            
            # Display sign correlation
            ax4.text(0.3, 0.6, f'Sign Correlation: {sign_correlation:.3f}\np-value: {sign_p_value:.3f}', 
                    transform=ax4.transAxes, fontsize=11,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8),
                    verticalalignment='top')
            
            # Customize the sign plot
            ax4.set_xlim(-1.2, 1.2)
            ax4.set_ylim(-1.2, 1.2)
            ax4.set_xlabel("Actual Return Sign", fontsize=10)
            ax4.set_ylabel("Predicted Return Sign", fontsize=10)
            ax4.set_title("Directional Accuracy Analysis", fontsize=12, fontweight="bold")
            ax4.grid(True, alpha=0.3)
            
            # Add quadrant labels for clarity
            ax4.text(-0.8, .4, "Predicted ↑\nActual ↓", ha='center', fontsize=9, 
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="pink", alpha=0.7))
            ax4.text(0.8, .4, "Predicted ↑\nActual ↑", ha='center', fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="lightgreen", alpha=0.7))
            ax4.text(-0.8, -.6, "Predicted ↓\nActual ↓", ha='center', fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="lightgreen", alpha=0.7))
            ax4.text(0.8, -.6, "Predicted ↓\nActual ↑", ha='center', fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="pink", alpha=0.7))
            
            # Set tick labels to be more meaningful
            ax4.set_xticks([-1, 1])
            ax4.set_xticklabels(['Negative', 'Positive'])
            ax4.set_yticks([-1, 1]) 
            ax4.set_yticklabels(['Negative', 'Positive'])
            
            plt.show()


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
        #self._last_regression_data = None

    def _predict(
        self, data: Dict[str, pd.Series]
    ) -> np.ndarray:
        """
        Perform prediction by sequentially applying oracles to predict residuals.

        Args:
            data: Dictionary mapping ticker symbols to historical price series

        Returns:
            an array containing the predicted next prices for the tickers
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
        #self._last_regression_data = data.copy()

        # Initialize with original data as first residual
        self.residuals.append({tk: s.reset_index(drop=True) for tk, s in data.items()})

        # Sequential oracle processing
        for oracle_idx, (oracle, window) in enumerate(zip(self.oracles, self.windows)):
            try:
                # Get predictions from current oracle on current residuals
                pred = oracle.regress(
                    self.residuals[-1], window=window, extrapolate=True
                )

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
                oracle_contributions = [
                    w * pred[ticker]
                    for w, pred in zip(self.weights, self.approx_residuals)
                ]
                net_regression[ticker] = np.sum(oracle_contributions, axis=0)

                # Convert back to pandas Series with proper indexing
                net_regression[ticker] = pd.Series(
                    net_regression[ticker], index=range(len(net_regression[ticker]))
                )
            except Exception as e:
                raise RuntimeError(f"Failed to combine predictions for {ticker}: {e}")

        # Add extrapolated predictions
        try:
            net_prediction = np.sum(
                [w * series for w, series in zip(self.weights, predictions)], axis=0
            )
        except Exception as e:
            raise RuntimeError(f"Failed to add extrapolated predictions: {e}")

        return net_prediction

    
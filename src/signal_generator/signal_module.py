"""
Signal generation module for converting Oracle predictions into actionable trading signals.

This module transforms raw predictions from Oracle instances into normalized trading signals
in the range [-1, +1], incorporating volatility-based position sizing and smoothing mechanisms.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
from abc import ABC, abstractmethod
from scipy.special import expit
from tqdm import tqdm
from collections import deque
from scipy.stats import rankdata
import time

import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from signal_generator.regression.base.oracle import Oracle


@dataclass
class SignalResult:
    """
    Container for signal generation results.

    This class encapsulates all outputs from the signal generation process,
    providing both the final signals and intermediate calculations for analysis.
    """

    signals: np.ndarray  # Final trading signals [-1, +1] for each ticker
    raw_predictions: np.ndarray  # Original predictions from Oracle
    volatility_adjustments: np.ndarray  # Volatility-based scaling factors
    tickers: List[str]  # Ticker symbols in order


class VolatilityAdjuster:
    """
    Handles volatility-based signal adjustments.

    This class implements risk-aware position sizing by scaling signals based on
    predicted volatility. High volatility assets get smaller position sizes to
    maintain consistent risk exposure across the portfolio.
    """

    def __init__(self):
        """
        Initialize volatility adjuster.

        Args:
            vol_target: Target volatility for position sizing
            min_vol_threshold: Minimum volatility to prevent division by zero
        """

    def adjust_for_volatility(
        self, signals: np.ndarray, asset_covariance: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Adjust signals based on predicted volatility structure.

        This method implements inverse volatility scaling - assets with higher
        predicted volatility receive proportionally smaller position sizes.
        This helps maintain consistent risk contribution across assets.

        Args:
            signals: Raw trading signals
            asset_covariance: Predicted covariance matrix from Oracle

        Returns:
            Tuple of (adjusted_signals, volatility_scaling_factors)
        """
        # Extract individual asset volatilities from covariance diagonal
        asset_volatilities = np.sqrt(np.diag(asset_covariance))

        # Prevent division by very small numbers
        asset_volatilities = np.maximum(asset_volatilities, 1e-2)

        # Calculate volatility-based scaling factors
        # Higher volatility -> smaller scaling factor -> smaller position
        vol_scaling = 1 / np.sqrt(asset_volatilities)

        # Normalize scaling factors to prevent extreme position sizes
        vol_scaling = np.clip(vol_scaling, 0.1, 5.0)

        # Apply volatility adjustment to signals
        adjusted_signals = signals * vol_scaling

        return adjusted_signals, vol_scaling


class SignalModule:
    """
    Main signal generation module that orchestrates the conversion of Oracle predictions
    into actionable trading signals.

    This class brings together all the components: Oracle predictions, smoothing,
    volatility adjustment, and signal normalization to produce final trading signals.
    """

    def __init__(
        self,
        oracle: Oracle,
        smoothing_window: int = 1,
        vol_adjuster: Optional[VolatilityAdjuster] = None,
        signal_strength_threshold: float = 0.01,
        market_vol_target: float = 2,
        name: str = "Unnamed SignalModule",
        version: str = "-.-",
    ):
        """
        Initialize the SignalModule.

        Args:
            oracle: The Oracle instance for generating predictions
            smoothing_window: Window length for prediction smoothing
            vol_adjuster: Handler for volatility-based adjustments
            signal_strength_threshold: Minimum threshold for signal generation
            market_vol_target: reference standard deviation of percentage changes of asset prices.
            lower than the reference will leave the signals unaffected.
            higher than the reference will scale down the signals proportionately.
        """
        self.oracle = oracle
        assert (
            smoothing_window >= 1
        ), "smoothing window must be at least 1 (1 => no smoothing)."
        self.smoothing_window = smoothing_window
        self.vol_adjuster = vol_adjuster or VolatilityAdjuster()
        self.signal_strength_threshold = signal_strength_threshold
        self.market_vol_target = market_vol_target
        self.name = name
        self.version = version

    def generate_signals(self, data: Dict[str, pd.Series]) -> SignalResult:
        """
        Generate trading signals from market data.

        This is the main orchestration method that coordinates all signal generation steps:
        1. Get predictions from Oracle (mean + covariance)
        2. Apply smoothing to predictions
        3. Convert to directional signals
        4. Apply volatility-based position sizing
        5. Normalize to [-1, +1] range

        Args:
            data: Dictionary mapping ticker symbols to price series

        Returns:
            SignalResult containing final signals and intermediate calculations
        """
        # Get predictions from Oracle
        prediction_result = self.oracle.predict(data)
        raw_predictions = prediction_result.predictions
        asset_covariance = prediction_result.asset_covariance
        market_volatility = np.sqrt(prediction_result.pc1_variance)

        past_prices = np.array([s.iloc[-self.smoothing_window] for s in data.values()])

        pred_returns_frac = (raw_predictions - past_prices) / past_prices

        # Apply volatility-based position sizing
        vol_adjusted_signals, vol_scaling = self.vol_adjuster.adjust_for_volatility(
            signals=pred_returns_frac, asset_covariance=asset_covariance
        )

        # Normalize signals to [-1, +1] range
        final_signals = self._normalize_signals(vol_adjusted_signals, market_volatility)

        return SignalResult(
            signals=final_signals,
            raw_predictions=raw_predictions,
            volatility_adjustments=vol_scaling,
            tickers=list(data.keys()),
        )

    def _normalize_signals(
        self, signals: np.ndarray, market_volatility: float = 0
    ) -> np.ndarray:
        """
        Normalize signals to [-1, +1] range using sigmoid.

        This approach ensures signals are distributed across the full range
        while preserving relative ordering.
        """
        assert len(signals) > 0, "no signal found."

        """

        n = max(np.abs(signals))

        assert n > 0, "all signals are zero. this should not happen."

        vol_scaling = np.clip(self.market_vol_target / market_volatility, 0, 1)
        # print(vol_scaling)
        raw_signals = (
            (1 - 1e-5) * (2 * expit(signals / n) - 1) / (expit(1.0) - expit(-1.0))
        )

        return vol_scaling * raw_signals\
        """

        rank = rankdata(signals)
        rank = rank / len(rank)
        rank = 2 * rank - 1

        return rank
    







    # =================================================
    #                Graphing utility
    # =================================================










    def diagnostics(self, data: Dict[str, pd.Series], lookback: int = 40) -> None:
        """
        Generate diagnostic visualizations showing price movements with signal strength overlays.

        This method provides comprehensive analysis of how trading signals align with price movements
        over time. For each ticker, it creates a visualization where:
        - Price movements are shown as line graphs
        - Predicted prices are overlaid as a separate line
        - Signal strength is represented as background color intensity
        - Blue indicates buy signals (positive values approaching +1)
        - Red indicates sell signals (negative values approaching -1)

        The method carefully avoids lookahead bias by only using historical data available
        at each point in time to generate signals.

        Args:
            data: Dictionary mapping ticker symbols to price series with datetime indices
        """

        # Step 1: Find shared timeline across all tickers
        # We need overlapping dates where all tickers have data
        print("Analyzing shared timeline across tickers...")

        # Get all unique indices and find intersection
        all_indices = [set(series.index) for series in data.values()]
        shared_indices = set.intersection(*all_indices)

        if len(shared_indices) == 0:
            raise ValueError("No shared dates found across all tickers")

        # Convert to sorted list for chronological processing
        shared_timeline = sorted(list(shared_indices))

        print(f"Found {len(shared_timeline)} shared trading days")
        print(f"Date range: {shared_timeline[0]} to {shared_timeline[-1]}")

        # Step 2: Ensure we have enough data for smoothing window
        # We need at least smoothing_window + 1 days to generate meaningful signals
        min_required_days = max(self.smoothing_window + 1, lookback)

        if len(shared_timeline) < min_required_days:
            raise ValueError(
                f"Insufficient data: need at least {min_required_days} days, got {len(shared_timeline)}"
            )

        # Step 3: Generate signals for each time point
        print("Generating historical signals...")

        # Storage for results
        tickers = list(data.keys())
        signals_over_time = []  # Will store signals for each date
        predictions_over_time = []  # Will store predicted prices for each date
        dates_with_signals = []  # Corresponding dates

        # Start from day where we have enough history for smoothing
        for i in tqdm(
            range(min_required_days, len(shared_timeline)), desc="Processing timeline"
        ):
            # t0 = time.perf_counter()

            current_date = shared_timeline[i]
            starting_idx = max(i - lookback, 0)

            # t1 = time.perf_counter()

            # Create historical data up to (but not including) current date
            # This ensures we're only using past information to generate signals
            historical_data = {}
            for ticker in tickers:
                # Get data up to the day before current date
                historical_dates = shared_timeline[
                    starting_idx:i
                ]  # Excludes current date
                historical_series = data[ticker].loc[historical_dates]
                historical_data[ticker] = historical_series

            # t2 = time.perf_counter()

            try:
                # Generate signals using only historical data
                signal_result = self.generate_signals(historical_data)
                current_signals = signal_result.signals.copy()
                current_predictions = signal_result.raw_predictions.copy()

                # t3 = time.perf_counter()

                signals_over_time.append(current_signals)
                predictions_over_time.append(current_predictions)
                dates_with_signals.append(current_date)
            except Exception as e:
                print(f"Warning: Could not generate signals for {current_date}: {e}")
                continue

            # t4 = time.perf_counter()

            # print(f"date prep: {t1-t0} | slicing: {t2-t1} | signal gen: {t3-t2} | appending: {t4-t3}")

        if len(signals_over_time) == 0:
            raise ValueError("No signals could be generated for the given data")

        # Convert to numpy arrays for easier manipulation
        signals_array = np.array(signals_over_time)  # Shape: (n_dates, n_tickers)
        predictions_array = np.array(
            predictions_over_time
        )  # Shape: (n_dates, n_tickers)

        print(
            f"Successfully generated signals for {len(dates_with_signals)} time points"
        )

        # Step 4: Create visualizations for each ticker
        print("Creating diagnostic visualizations...")

        # Set up the plotting style
        plt.style.use("default")
        n_tickers = len(tickers)

        # Calculate subplot layout (prefer wider than tall)
        n_cols = min(2, n_tickers)  # Max 2 columns
        n_rows = (n_tickers + n_cols - 1) // n_cols  # Ceiling division

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(10 * n_cols, 4 * n_rows))
        fig.suptitle(
            "Price Movements with Signal Strength Overlays and Predictions",
            fontsize=16,
            fontweight="bold",
        )

        # Handle case where we have only one subplot
        if n_tickers == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes if n_cols > 1 else [axes]
        else:
            axes = axes.flatten()

        for ticker_idx, ticker in enumerate(tickers):
            ax = axes[ticker_idx]

            # Get price data for visualization dates
            price_dates = dates_with_signals
            actual_prices = [data[ticker].loc[date] for date in price_dates]
            predicted_prices = predictions_array[
                :, ticker_idx
            ]  # All predictions for this ticker

            # Get signals for this ticker
            ticker_signals = signals_array[:, ticker_idx]  # All signals for this ticker

            # Create the background heatmap
            # We'll use a color mesh to create smooth color transitions

            # Create a 2D grid for the heatmap
            # X-axis: dates, Y-axis: price levels (we'll use a range around actual prices)
            all_prices = list(actual_prices) + list(predicted_prices)
            price_min, price_max = min(all_prices), max(all_prices)
            price_range = price_max - price_min
            price_padding = price_range * 0.1  # 10% padding

            y_bottom = price_min - price_padding
            y_top = price_max + price_padding

            # Create coordinate arrays for the mesh
            x_coords = np.arange(len(price_dates))
            y_coords = np.linspace(
                y_bottom, y_top, 100
            )  # 100 price levels for smooth gradient

            X, Y = np.meshgrid(x_coords, y_coords)

            # Create signal intensity matrix
            # Each column represents the signal strength for that time point
            # The key insight: we need to fill each column (representing a time period)
            # with the same signal value across all price levels (rows)
            Z = np.zeros_like(X, dtype=float)
            for time_idx, signal in enumerate(ticker_signals):
                Z[:, time_idx] = signal

            # Create custom colormap: red for sell (-1), white for neutral (0), blue for buy (+1)
            # We import LinearSegmentedColormap from matplotlib.colors, not matplotlib.cm.colors
            from matplotlib.colors import LinearSegmentedColormap

            colors = [
                "darkred",
                "red",
                "lightcoral",
                "white",
                "lightgreen",
                "green",
                "darkgreen",
            ]
            n_bins = 100
            cmap = LinearSegmentedColormap.from_list(
                "signal_strength", colors, N=n_bins
            )

            # Plot the background heatmap with increased alpha for better visibility
            im = ax.imshow(
                Z,
                extent=[0, len(price_dates) - 1, y_bottom, y_top],
                aspect="auto",
                cmap=cmap,
                vmin=-1,
                vmax=1,
                alpha=0.8,
                origin="lower",
            )

            # Plot the actual price line on top
            ax.plot(
                range(len(price_dates)),
                actual_prices,
                "black",
                linewidth=1,
                alpha=0.9,
                label="Actual Price",
            )

            # Plot the predicted price line on top
            ax.plot(
                range(len(price_dates)),
                predicted_prices,
                "blue",
                linewidth=1,
                alpha=0.9,
                linestyle="-",
                label="Predicted Price",
            )

            # Customize the plot
            ax.set_title(f"{ticker}", fontweight="bold")
            ax.set_ylabel("Price")
            ax.set_ylim(y_bottom, y_top)

            # Format x-axis dates
            # Show every nth date to avoid crowding
            n_labels = min(10, len(price_dates))  # Max 10 date labels
            step = max(1, len(price_dates) // n_labels)
            tick_positions = range(0, len(price_dates), step)
            tick_labels = [price_dates[i].strftime("%Y-%m-%d") for i in tick_positions]

            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels, rotation=45, ha="right")
            ax.set_xlabel("Date")

            # Add grid for better readability
            ax.grid(True, alpha=0.3)
            ax.legend()

            # Add signal statistics and prediction accuracy as text
            prediction_error = np.mean(
                np.abs(np.array(actual_prices) - predicted_prices)
            )
            signal_stats = f"Signals: μ={np.mean(ticker_signals):.3f}, σ={np.std(ticker_signals):.3f}"
            prediction_stats = f"MAE: {prediction_error:.3f}"
            combined_stats = f"{signal_stats}\n{prediction_stats}"

            ax.text(
                0.02,
                0.98,
                combined_stats,
                transform=ax.transAxes,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                verticalalignment="top",
                fontsize=9,
            )

        # Hide unused subplots
        for idx in range(n_tickers, len(axes)):
            axes[idx].set_visible(False)

        # Add a colorbar to show signal strength scale
        # Place it on the right side of the entire figure
        cbar_ax = fig.add_axes((0.92, 0.15, 0.02, 0.7))  # (left, bottom, width, height)
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label("Signal Strength", rotation=270, labelpad=20)
        cbar.set_ticks([-1, -0.5, 0, 0.5, 1])
        cbar.set_ticklabels(
            [
                "Strong Sell (-1)",
                "Sell (-0.5)",
                "Neutral (0)",
                "Buy (0.5)",
                "Strong Buy (1)",
            ]
        )

        plt.tight_layout()
        plt.subplots_adjust(right=0.9)  # Make room for colorbar
        plt.show()

        # Step 5: Print summary statistics
        print("\n" + "=" * 60)
        print("DIAGNOSTIC SUMMARY")
        print("=" * 60)

        print(f"Analysis period: {dates_with_signals[0]} to {dates_with_signals[-1]}")
        print(f"Number of signal observations: {len(dates_with_signals)}")
        print(f"Smoothing window used: {self.smoothing_window} days")

        print("\nSignal Statistics by Ticker:")
        print("-" * 40)
        for i, ticker in enumerate(tickers):
            ticker_signals = signals_array[:, i]
            print(
                f"{ticker:>8}: μ={np.mean(ticker_signals):+.3f}, "
                f"σ={np.std(ticker_signals):.3f}, "
                f"range=[{np.min(ticker_signals):+.3f}, {np.max(ticker_signals):+.3f}]"
            )

        print("\nPrediction Accuracy by Ticker:")
        print("-" * 40)
        for i, ticker in enumerate(tickers):
            actual = [data[ticker].loc[date] for date in dates_with_signals]
            predicted = predictions_array[:, i]
            mae = np.mean(np.abs(np.array(actual) - predicted))
            mape = (
                np.mean(np.abs((np.array(actual) - predicted) / np.array(actual))) * 100
            )
            print(f"{ticker:>8}: MAE={mae:.3f}, " f"MAPE={mape:.2f}%")

        # Cross-ticker correlation analysis
        if len(tickers) > 1:
            print(f"\nSignal Correlation Matrix:")
            print("-" * 30)
            signal_corr = np.corrcoef(signals_array.T)

            # Create a simple text-based correlation matrix
            print(f"{'':>8}", end="")
            for ticker in tickers:
                print(f"{ticker:>8}", end="")
            print()

            for i, ticker_i in enumerate(tickers):
                print(f"{ticker_i:>8}", end="")
                for j, ticker_j in enumerate(tickers):
                    print(f"{signal_corr[i,j]:>8.3f}", end="")
                print()

        return None

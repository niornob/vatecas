from collections import deque
from typing import Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm

from signal_generator.signal_diagnostics_utils import (
    create_signal_colormap,
    plot_ticker_with_heatmap,
)
from signal_generator.signal_module import SignalModule, SignalResult


class SignalDiagnostics:
    """
    Encapsulates all logic for:
     1. Finding a shared timeline across tickers
     2. Building historical slices at each date (up to that date)
     3. Calling SignalModule.generate_signals(...) on each historical slice
     4. Collecting signals, predictions, and dates
     5. Rendering per‐ticker visualization panels
     6. Printing summary statistics and correlation matrices
    """

    def __init__(
        self,
        signal_module: SignalModule,
        lookback: int = 40,
    ):
        """
        Args:
            signal_module: an instance of SignalModule
            lookback: number of past days to include at each step
            smoothing_window: passed through to SignalModule (for consistency)
        """
        self.signal_module = signal_module
        self.lookback = lookback
        self.smoothing_window = signal_module.smoothing_window

    def run(self, data: Dict[str, pd.Series]) -> None:
        """
        Main entry point. Given a dict { ticker: pd.Series(prices, indexed by datetime) },
        this method will:
          a) Determine the shared timeline (intersection of all indices)
          b) Ensure at least (lookback + smoothing_window) days exist
          c) For each date in that timeline (starting at index=lookback):
               - Build a dict of NumPy arrays representing prices up to yesterday
               - Call self.signal_module.generate_signals(...)
               - Store signals and predictions
          d) After looping, convert results into arrays and call plotting & summary prints.
        """
        tickers = list(data.keys())

        # 1) Find shared timeline (sorted list of common datetimes)
        shared_timeline = self._find_shared_timeline(data)
        min_required = max(self.lookback + 1, self.smoothing_window + 1)
        if len(shared_timeline) < min_required:
            raise ValueError(
                f"Need ≥ {min_required} shared dates, got {len(shared_timeline)}."
            )

        # 2) Precompute a mapping of datetime → integer index for each ticker
        index_positions = {
            ticker: {dt: pos for pos, dt in enumerate(data[ticker].index)}
            for ticker in tickers
        }

        # 3) Loop over each date index (start = lookback)
        signals_over_time: List[np.ndarray] = []
        predictions_over_time: List[np.ndarray] = []
        dates_with_signals: List[pd.Timestamp] = []

        past_predictions: deque[np.ndarray] = deque([], maxlen=self.smoothing_window)

        for i in tqdm(
            range(self.lookback, len(shared_timeline)), desc="Processing timeline"
        ):
            current_date = shared_timeline[i]

            # 3a) Build historical NumPy arrays for each ticker
            hist_data_numpy = self._prepare_historical_data(
                data, tickers, index_positions, shared_timeline, i
            )

            # 3b) Generate signals
            try:
                if len(past_predictions) == self.smoothing_window:
                    # print(past_predictions)
                    sig_res: SignalResult = self.signal_module.generate_signals(
                        data=hist_data_numpy, past_predictions=past_predictions[0]
                    )
                    # print(sig_res.raw_predictions)
                else:
                    sig_res: SignalResult = self.signal_module.generate_signals(
                        data=hist_data_numpy
                    )
                past_predictions.append(sig_res.raw_predictions.copy())
                # print(past_predictions)
                signals_over_time.append(sig_res.signals.copy())
                predictions_over_time.append(sig_res.raw_predictions.copy())
                dates_with_signals.append(current_date)

            except Exception as e:
                print(f"Warning: Could not generate signals for {current_date}: {e}")
                continue

        if not signals_over_time:
            raise RuntimeError(
                "No signals could be generated for any date in the timeline."
            )

        # 4) Convert to arrays (n_dates × n_tickers)
        signals_matrix = np.vstack(signals_over_time)
        predictions_matrix = np.vstack(predictions_over_time)

        # 5) Plot each ticker’s diagnostics
        self._plot_all_tickers(
            data, tickers, dates_with_signals, signals_matrix, predictions_matrix
        )

        # 6) Print summary stats
        self._print_summary_statistics(
            data, tickers, dates_with_signals, signals_matrix, predictions_matrix
        )

    def _find_shared_timeline(self, data: Dict[str, pd.Series]) -> List[pd.Timestamp]:
        """
        Returns a sorted list of datetime indices that appear in every Series.
        """
        all_indices = [set(series.index) for series in data.values()]
        shared = set.intersection(*all_indices)
        return sorted(shared)

    def _prepare_historical_data(
        self,
        data: Dict[str, pd.Series],
        tickers: List[str],
        index_positions: Dict[str, Dict[pd.Timestamp, int]],
        shared_timeline: List[pd.Timestamp],
        i: int,
    ) -> Dict[str, pd.Series]:
        """
        For each ticker, produce a NumPy array of all prices up to
        (but not including) shared_timeline[i].
        Uses integer‐based .iloc slicing internally to avoid repeated
        .loc[list_of_timestamps] overhead.
        """
        hist_arrays: Dict[str, pd.Series] = {}
        start_idx = max(0, i - self.lookback)  # earliest position in shared_timeline
        end_idx = i - 1  # last date index to include

        # Loop over tickers
        for ticker in tickers:
            # 1) find the first and last POSITION in the original Series for that date range
            start_date = shared_timeline[start_idx]
            end_date = shared_timeline[end_idx]
            pos_map = index_positions[ticker]
            start_pos = pos_map[start_date]
            end_pos = pos_map[end_date]

            # 2) slice the underlying Pandas Series by iloc, then convert to NumPy array
            series = data[ticker]
            hist_slice = series.iloc[
                start_pos : end_pos + 1
            ]  # array of length ≤ lookback
            hist_arrays[ticker] = hist_slice

        return hist_arrays

    def _plot_all_tickers(
        self,
        data: Dict[str, pd.Series],
        tickers: List[str],
        dates_with_signals: List[pd.Timestamp],
        signals_matrix: np.ndarray,
        predictions_matrix: np.ndarray,
    ) -> None:
        """
        For each ticker, delegate to a helper that builds a single panel
        (heatmap + lines + stats box).
        """
        # Create or reuse a colormap
        cmap = create_signal_colormap()

        n_tickers = len(tickers)
        n_cols = min(2, n_tickers)
        n_rows = (n_tickers + n_cols - 1) // n_cols

        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(10 * n_cols, 4 * n_rows))
        fig.suptitle(
            "Price Movements with Signal‐Strength Overlays and Predictions",
            fontsize=16,
            fontweight="bold",
        )

        # Flatten axes array
        if n_tickers == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes if n_cols > 1 else [axes]
        else:
            axes = axes.flatten()

        # For each ticker, plot
        for idx, ticker in enumerate(tickers):
            ax = axes[idx]
            # 1) Grab actual prices at each date_with_signals
            actual_prices = [data[ticker].loc[dt] for dt in dates_with_signals]
            # 2) Grab predicted prices (vector) from predictions_matrix[:, idx]
            predicted_prices = predictions_matrix[:, idx]
            # 3) Grab signals (vector) from signals_matrix[:, idx]
            ticker_signals = signals_matrix[:, idx]
            # 4) Call a helper that draws the heatmap + lines + stats box
            plot_ticker_with_heatmap(
                ax,
                dates_with_signals,
                actual_prices,
                predicted_prices,
                ticker_signals,
                cmap,
            )
            ax.set_title(f"{ticker}", fontweight="bold")

        # Hide any unused subplot axes
        for idx in range(n_tickers, len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()
        plt.subplots_adjust(right=0.9)  # make room for colorbar if needed
        plt.show()

    def _print_summary_statistics(
        self,
        data: Dict[str, pd.Series],
        tickers: List[str],
        dates_with_signals: List[pd.Timestamp],
        signals_matrix: np.ndarray,
        predictions_matrix: np.ndarray,
    ) -> None:
        """
        Prints summary to stdout:
          • Date range, number of observations, smoothing window
          • Per‐ticker μ, σ, range of signals
          • Per‐ticker MAE, MAPE of predictions
          • (If >1 ticker) cross‐ticker correlation matrix of signals
        """
        print("\n" + "=" * 60)
        print("DIAGNOSTIC SUMMARY")
        print("=" * 60)
        print(f"Analysis period: {dates_with_signals[0]} to {dates_with_signals[-1]}")
        print(f"Number of signal observations: {len(dates_with_signals)}")
        print(f"Smoothing window used: {self.smoothing_window} days")

        print("\nSignal Statistics by Ticker:")
        print("-" * 40)
        for i, ticker in enumerate(tickers):
            sigs = signals_matrix[:, i]
            print(
                f"{ticker:>8}: μ={np.mean(sigs):+.3f}, σ={np.std(sigs):.3f}, range=[{np.min(sigs):+.3f}, {np.max(sigs):+.3f}]"
            )

        print("\nPrediction Accuracy by Ticker:")
        print("-" * 40)
        for i, ticker in enumerate(tickers):
            actual = np.array([data[ticker].loc[dt] for dt in dates_with_signals])
            pred = predictions_matrix[:, i]
            mae = np.mean(np.abs(actual - pred))
            mape = np.mean(np.abs((actual - pred) / actual)) * 100
            print(f"{ticker:>8}: MAE={mae:.3f}, MAPE={mape:.2f}%")

        # If multiple tickers, show a simple correlation table
        if len(tickers) > 1:
            print("\nSignal Correlation Matrix:")
            print("-" * 30)
            corr_mat = np.corrcoef(signals_matrix.T)
            # Print a text‐based matrix
            header = f"{'':>8}" + "".join(f"{t:>8}" for t in tickers)
            print(header)
            for i, ti in enumerate(tickers):
                row = f"{ti:>8}" + "".join(
                    f"{corr_mat[i, j]:>8.3f}" for j in range(len(tickers))
                )
                print(row)

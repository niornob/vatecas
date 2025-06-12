from typing import Mapping, Sequence, Tuple, Dict, List, Optional
import pandas as pd
from tqdm import tqdm
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings

from portfolio_management.portfolio_manager import PortfolioManager
from signal_generator.signal_module import SignalModule
from utils.visualizations import (
    _equity_vs_benchmark,
    _equity_vs_benchmark_marked,
    _holdings_over_time,
)


class BacktestEngine:
    """
    A robust backtesting engine that combines multiple signal modules to simulate trading strategies.

    The engine handles data alignment, signal generation, and trade execution while maintaining
    strict temporal constraints to avoid look-ahead bias.
    """

    def __init__(
        self,
        data: Mapping[str, pd.DataFrame],
        signal_module: SignalModule,
        manager: PortfolioManager,
        lookback_days: int = 40,
    ):
        """
        Initialize the backtesting engine.

        Args:
            data: Dictionary mapping ticker symbols to their price DataFrames
            modules_weights: List of (SignalModule, weight) tuples for signal combination
            manager: PortfolioManager instance to handle portfolio operations
            lookback_days: Number of historical days needed for signal generation
        """
        self.signal_module = signal_module

        # Validate inputs
        self._validate_inputs(data, manager, lookback_days)

        # Store original data and create deep copies to avoid mutation
        self.raw_data = data
        self.data = {tk: df.copy() for tk, df in data.items()}

        # Create unified timeline and align data
        self._create_unified_timeline()
        self._align_data_to_timeline()

        # Set up timeline constraints
        self.lookback_period = pd.Timedelta(days=lookback_days)
        self._setup_trading_timeline(manager)

        # Initialize portfolio manager and results storage
        self.manager = manager
        self.signals: Dict[str, pd.Series] = {
            tk: pd.Series(dtype=float) for tk in data.keys()
        }
        self.execution_log: List[Dict] = []  # Track execution details for debugging

    def _validate_inputs(self, data, manager, lookback_days):
        """Validate all inputs to catch errors early."""
        if not data:
            raise ValueError("Data dictionary cannot be empty")

        if lookback_days < 1:
            raise ValueError("Lookback days must be positive")

        # Check that all dataframes have required columns
        required_cols = ["adjClose", "adjOpen"]
        for ticker, df in data.items():
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(
                    f"Ticker {ticker} missing required columns: {missing_cols}"
                )

    def _create_unified_timeline(self):
        """Create a unified timeline from all ticker data."""
        all_dates = set()
        for df in self.data.values():
            if not df.index.empty:
                all_dates.update(df.index)

        if not all_dates:
            raise ValueError("No valid dates found in any ticker data")

        self.timeline = pd.DatetimeIndex(sorted(all_dates))

    def _align_data_to_timeline(self):
        """Align all ticker data to the unified timeline."""
        self.aligned = {}
        for ticker, df in self.data.items():
            # Reindex to unified timeline and forward-fill missing values
            aligned_df = df.reindex(self.timeline)

            # Forward fill first, then handle any remaining NaN values at the beginning
            aligned_df = aligned_df.ffill()

            # For any remaining NaN values (at the start), use backward fill or zero
            aligned_df = aligned_df.bfill().fillna(0)

            self.aligned[ticker] = aligned_df

    def _setup_trading_timeline(self, manager):
        """Set up the trading timeline based on portfolio initial time."""
        initial_time = manager.portfolio.initial_time

        # Filter timeline to start from initial portfolio time
        self.trading_timeline = self.timeline[self.timeline >= initial_time]

        if self.trading_timeline.empty:
            raise ValueError("No trading days available after portfolio initial time")

        # Extend aligned data to include lookback period before trading starts
        lookback_start = initial_time - self.lookback_period
        self.aligned = {
            ticker: df.loc[df.index >= lookback_start]
            for ticker, df in self.aligned.items()
        }

    def run(self) -> None:
        """
        Execute the backtest simulation.

        Returns:
            Dictionary containing backtest results and metadata
        """

        print(f"Starting backtest for {len(self.trading_timeline)} trading days...")

        assert self.lookback_period.days < len(
            self.trading_timeline
        ), f"not enough data ({len(self.trading_timeline)}) for lookback_period ({self.lookback_period.days} days)."

        past_predictions: deque[np.ndarray] = deque(
            [], maxlen=self.signal_module.smoothing_window
        )

        for today in tqdm(self.trading_timeline, desc="Backtesting"):
            try:
                yesterday = today - pd.Timedelta(days=1)
                historical_data, enough_data = self._prepare_historical_data(yesterday)

                if not enough_data:
                    # print(f"{today}: not enough historical data. Continuing...")
                    continue

                # Generate signals based on data up to yesterday
                if len(past_predictions) == self.signal_module.smoothing_window:
                    signal = self.signal_module.generate_signals(
                        data=historical_data, past_predictions=past_predictions[0]
                    )
                else:
                    signal = self.signal_module.generate_signals(data=historical_data)
                
                signals = {tk: sig for tk, sig in zip(signal.tickers, signal.signals)}
                past_predictions.append(signal.raw_predictions.copy())

                # Store signals for analysis
                self._record_signals(signals, today)
                
                # Execute trades if we have valid opening prices
                opening_prices = self._get_opening_prices(today)
                if opening_prices:
                    self._execute_trades(signals, opening_prices, today)

            except Exception as e:
                print(f"Error processing {today}: {str(e)}")
                self.execution_log.append(
                    {"date": today, "error": str(e), "type": "execution_error"}
                )
                continue

        return

    def _prepare_historical_data(self, cutoff_date):
        """
        Prepare historical data up to the cutoff date.
        Also returns a bool value indicating whether there is enough historical data
        for the lookback_period.
        """
        lookback_start = cutoff_date - self.lookback_period

        historical_data = {}
        for ticker, df in self.aligned.items():
            # Get data from lookback_start to cutoff_date (inclusive)
            mask = df.index <= cutoff_date
            if sum(mask) < self.lookback_period.days:
                return {}, False
            mask = mask & (df.index >= lookback_start)
            historical_data[ticker] = df.loc[mask, "adjClose"].copy()

        return historical_data, True

    def _record_signals(self, signals, date):
        """Record generated signals for later analysis."""
        for ticker, signal in signals.items():
            new_signal = pd.Series([signal], index=[date])
            self.signals[ticker] = pd.concat([self.signals[ticker], new_signal])

    def _get_opening_prices(self, date):
        """Get opening prices for all tickers on a given date."""
        opening_prices = {}
        for ticker in self.aligned.keys():
            try:
                price = self.aligned[ticker].loc[date, "adjOpen"]
                if pd.notna(price) and price > 0:  # Ensure valid price
                    opening_prices[ticker] = price
            except (KeyError, IndexError):
                # Skip if no data available for this date
                continue

        return opening_prices

    def _execute_trades(self, signals, opening_prices, date):
        """Execute trades based on signals and record execution details."""
        # Only process signals for tickers with valid opening prices
        executable_signals = {
            ticker: signal
            for ticker, signal in signals.items()
            if ticker in opening_prices
        }
        
        if executable_signals:
            # print(executable_signals, opening_prices, date)
            self.manager.process_signals(executable_signals, opening_prices, date)

            # Log execution details
            self.execution_log.append(
                {
                    "date": date,
                    "signals": executable_signals.copy(),
                    "prices": opening_prices.copy(),
                    "type": "trade_execution",
                }
            )

    def _get_empty_results(self):
        """Return empty results structure when no trading occurs."""
        return {
            "signals": self.signals,
            "execution_log": self.execution_log,
            "portfolio_results": {},
            "metadata": {
                "total_trading_days": 0,
                "successful_executions": 0,
                "errors": 0,
            },
        }

    def _compile_results(self):
        """Compile final backtest results."""
        portfolio_results = self.manager.get_results()

        successful_executions = len(
            [log for log in self.execution_log if log.get("type") == "trade_execution"]
        )
        errors = len(
            [log for log in self.execution_log if log.get("type") == "execution_error"]
        )

        return {
            "signals": self.signals,
            "execution_log": self.execution_log,
            "portfolio_results": portfolio_results,
            "metadata": {
                "total_trading_days": len(self.trading_timeline),
                "successful_executions": successful_executions,
                "errors": errors,
            },
        }

    # ====================================================================
    #                        Plotting methods
    # ====================================================================

    def plot_signals(
        self,
        tickers: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (15, 10),
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plot trading signals for specified tickers over time.

        Args:
            tickers: List of tickers to plot. If None, plots all tickers.
            figsize: Figure size as (width, height)
            save_path: Path to save the plot. If None, displays the plot.
        """
        if not self.signals or all(series.empty for series in self.signals.values()):
            print("No signals available to plot. Run backtest first.")
            return

        # Filter tickers to plot
        if tickers is None:
            tickers = [tk for tk, series in self.signals.items() if not series.empty]
        else:
            tickers = [
                tk
                for tk in tickers
                if tk in self.signals and not self.signals[tk].empty
            ]

        if not tickers:
            print("No valid tickers with signals found.")
            return

        # Set up the plot
        n_tickers = len(tickers)
        fig, axes = plt.subplots(n_tickers, 1, figsize=figsize, sharex=True)

        # Handle single ticker case
        if n_tickers == 1:
            axes = [axes]

        # Color mapping for signal strength
        colors = ["red", "orange", "gray", "lightgreen", "green"]

        for i, ticker in enumerate(tickers):
            signals = self.signals[ticker]

            if signals.empty:
                continue

            ax = axes[i]

            # Create color map based on signal strength
            signal_colors = []
            for signal in signals.values:
                if signal <= -0.6:
                    signal_colors.append(colors[0])  # Strong sell - red
                elif signal <= -0.2:
                    signal_colors.append(colors[1])  # Weak sell - orange
                elif signal <= 0.2:
                    signal_colors.append(colors[2])  # Neutral - gray
                elif signal <= 0.6:
                    signal_colors.append(colors[3])  # Weak buy - light green
                else:
                    signal_colors.append(colors[4])  # Strong buy - green

            # Plot signals as line and scatter
            ax.plot(
                signals.index, signals.values, linewidth=1, alpha=0.7, color="black"
            )
            scatter = ax.scatter(
                signals.index,
                signals.values,
                c=signal_colors,
                s=20,
                alpha=0.8,
                edgecolors="black",
                linewidths=0.5,
            )

            # Add horizontal reference lines
            ax.axhline(y=0, color="black", linestyle="-", alpha=0.3, linewidth=0.5)
            ax.axhline(y=0.5, color="green", linestyle="--", alpha=0.5, linewidth=0.5)
            ax.axhline(y=-0.5, color="red", linestyle="--", alpha=0.5, linewidth=0.5)

            # Formatting
            ax.set_ylabel(f"{ticker}\nSignal", fontsize=10)
            ax.set_ylim(-1.1, 1.1)
            ax.grid(True, alpha=0.3)

            # Add statistics to the plot
            mean_signal = signals.mean()
            std_signal = signals.std()
            ax.text(
                0.02,
                0.95,
                f"μ={mean_signal:.3f}, σ={std_signal:.3f}",
                transform=ax.transAxes,
                fontsize=8,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )

        # Set common x-axis label
        axes[-1].set_xlabel("Date", fontsize=12)

        # Add title and legend
        plt.suptitle(
            "Trading Signals Over Time\n(Green=Buy, Red=Sell, Gray=Neutral)",
            fontsize=14,
            fontweight="bold",
        )

        # Create custom legend
        from matplotlib.lines import Line2D

        legend_elements = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=colors[4],
                markersize=8,
                label="Strong Buy (>0.6)",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=colors[3],
                markersize=8,
                label="Weak Buy (0.2-0.6)",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=colors[2],
                markersize=8,
                label="Neutral (-0.2-0.2)",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=colors[1],
                markersize=8,
                label="Weak Sell (-0.6--0.2)",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=colors[0],
                markersize=8,
                label="Strong Sell (<-0.6)",
            ),
        ]

        fig.legend(
            handles=legend_elements, loc="upper right", bbox_to_anchor=(0.98, 0.98)
        )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Plot saved to {save_path}")
        else:
            plt.show()

    def equity_vs_benchmarks(self, benchmarks: List[str]) -> None:
        """Plot equity curve against benchmark assets."""
        equity = self.manager.get_results().get("equity_curve")
        if equity is None or equity.empty:
            print("No equity curve available. Run backtest first.")
            return

        _equity_vs_benchmark(
            equity=equity,
            data=self.aligned,
            benchmarks=benchmarks,
            initial_time=self.manager.portfolio.initial_time,
            title="Equity vs. (normalized) Benchmarks",
        )

    def holdings_over_time(self) -> None:
        """Plot portfolio allocation over time."""
        portfolio = self.manager.get_results()
        if not portfolio or "equity_curve" not in portfolio:
            print("No portfolio results available. Run backtest first.")
            return

        _holdings_over_time(
            equity=portfolio["equity_curve"],
            data=self.aligned,
            holdings=portfolio["holdings"],
            price="adjClose",
            title="Portfolio Allocation (% of Equity)",
        )

    def equity_vs_benchmarks_marked(
        self, benchmarks: List[str], history: List = []
    ) -> None:
        """Plot equity curve against benchmarks with trade markers."""
        equity = self.manager.get_results().get("equity_curve")
        if equity is None or equity.empty:
            print("No equity curve available. Run backtest first.")
            return

        _equity_vs_benchmark_marked(
            equity=equity,
            data=self.aligned,
            benchmarks=benchmarks,
            initial_time=self.manager.portfolio.initial_time,
            order_history=history,
            title="Equity vs. (normalized) Benchmarks",
        )

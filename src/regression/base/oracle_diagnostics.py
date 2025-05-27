import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from statsmodels.graphics.tsaplots import plot_acf

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parents[2]))
from regression.base.oracle import Oracle


def oracle_diagnostics(oracle: 'Oracle', data: dict[str, pd.Series], window: int) -> None:
    """Generate enhanced comparison plots including variance bands and market factor analysis."""
    actual_df = pd.DataFrame(data)
    preds_df, vol_bands_df, market_factor = oracle.regress(data, window=window)
    preds_df = pd.DataFrame(preds_df)
    vol_bands_df = pd.DataFrame(vol_bands_df)

    dates = actual_df.index
    tickers = actual_df.columns

    for ticker in tickers:
        # Create enhanced figure with variance-aware plots
        fig = plt.figure(figsize=(16, 16))
        gs = fig.add_gridspec(
            4, 2, height_ratios=[1, 1, 0.8, 0.8], hspace=0.35, wspace=0.25
        )

        # Enhanced plot layout:
        ax1 = fig.add_subplot(gs[0, :])  # Price predictions with confidence bands
        ax2 = fig.add_subplot(gs[1, 0])  # Volatility forecast accuracy
        ax3 = fig.add_subplot(gs[1, 1])  # Return distribution analysis
        ax4 = fig.add_subplot(gs[2, 0])  # Residuals with volatility context
        ax5 = fig.add_subplot(gs[2, 1])  # ACF of residuals
        ax6 = fig.add_subplot(gs[3, :])  # Volatility time series

        fig.suptitle(
            f"{oracle.name} Enhanced Analysis: {ticker}",
            fontsize=18,
            fontweight="bold",
        )

        # ===== PLOT 1: Price predictions with confidence bands =====
        ax1.plot(
            dates,
            actual_df[ticker],
            label=f"Actual {ticker}",
            linewidth=2.5,
            alpha=0.9,
            color="navy",
        )

        # Main prediction line
        ax1.plot(
            dates,
            preds_df[ticker],
            label=f"Predicted {ticker}",
            color="darkorange",
            linestyle="-",
            linewidth=2,
            alpha=0.8,
        )

        # Confidence bands (±1 and ±2 standard deviations)
        upper_1std = preds_df[ticker] + vol_bands_df[ticker]
        lower_1std = preds_df[ticker] - vol_bands_df[ticker]
        upper_2std = preds_df[ticker] + 2 * vol_bands_df[ticker]
        lower_2std = preds_df[ticker] - 2 * vol_bands_df[ticker]

        ax1.fill_between(
            dates,
            lower_2std,
            upper_2std,
            alpha=0.2,
            color="orange",
            label="±2σ Confidence",
        )
        ax1.fill_between(
            dates,
            lower_1std,
            upper_1std,
            alpha=0.3,
            color="orange",
            label="±1σ Confidence",
        )

        ax1.set_title(
            "Price Forecasts with Volatility-Based Confidence Intervals",
            fontsize=14,
            fontweight="bold",
        )
        ax1.set_xlabel("Time", fontsize=11)
        ax1.set_ylabel(f"{ticker} Price", fontsize=11)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)

        # ===== PLOT 2: Volatility forecast accuracy =====
        # Compare predicted volatility vs realized volatility
        actual_returns = actual_df[ticker].pct_change().dropna()
        realized_vol = (
            actual_returns.rolling(window=5).std().shift(-2)
        )  # Forward-looking realized vol
        predicted_vol = vol_bands_df[ticker] / 100  # Convert to decimal

        # Align series for comparison
        common_idx = realized_vol.dropna().index.intersection(list(predicted_vol.index))
        if len(common_idx) > 10:
            realized_aligned = realized_vol.loc[common_idx]
            predicted_aligned = predicted_vol.loc[common_idx]

            ax2.scatter(realized_aligned, predicted_aligned, alpha=0.6, s=25)

            # Add perfect prediction line
            min_vol = min(realized_aligned.min(), predicted_aligned.min())
            max_vol = max(realized_aligned.max(), predicted_aligned.max())
            ax2.plot(
                [min_vol, max_vol],
                [min_vol, max_vol],
                "r--",
                alpha=0.7,
                linewidth=1.5,
            )

            # Calculate and display correlation
            vol_corr, vol_p = pearsonr(realized_aligned, predicted_aligned)
            ax2.text(
                0.05,
                0.95,
                f"Vol Correlation: {vol_corr:.3f}\np-value: {vol_p:.3f}",
                transform=ax2.transAxes,
                fontsize=10,
                bbox=dict(
                    boxstyle="round,pad=0.3", facecolor="lightcyan", alpha=0.8
                ),
                verticalalignment="top",
            )

        ax2.set_title(
            "Volatility Forecast Accuracy", fontsize=12, fontweight="bold"
        )
        ax2.set_xlabel("Realized Volatility", fontsize=10)
        ax2.set_ylabel("Predicted Volatility", fontsize=10)
        ax2.grid(True, alpha=0.3)

        # ===== PLOT 3: Return distribution analysis =====
        if len(actual_returns) > 10:
            ax3.hist(
                actual_returns,
                bins=30,
                alpha=0.7,
                color="skyblue",
                density=True,
                label="Actual Returns",
            )

            # Overlay normal distribution based on mean predicted volatility
            mean_pred_vol = vol_bands_df[ticker].mean() / 100
            x_norm = np.linspace(actual_returns.min(), actual_returns.max(), 100)
            y_norm = (1 / (mean_pred_vol * np.sqrt(2 * np.pi))) * np.exp(
                -0.5 * (x_norm / mean_pred_vol) ** 2
            )
            ax3.plot(
                x_norm, y_norm, "orange", linewidth=2, label="Normal (Pred Vol)"
            )

        ax3.set_title(
            "Return Distribution vs Normal", fontsize=12, fontweight="bold"
        )
        ax3.set_xlabel("Returns", fontsize=10)
        ax3.set_ylabel("Density", fontsize=10)
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3)

        # ===== PLOT 4: Residuals with volatility context =====
        residuals = actual_df[ticker] - preds_df[ticker]
        standardized_residuals = residuals / vol_bands_df[ticker]

        ax4.scatter(
            residuals.index,
            standardized_residuals,
            c=vol_bands_df[ticker],
            cmap="viridis",
            s=25,
            alpha=0.7,
        )
        ax4.axhline(y=0, color="black", linestyle="--", alpha=0.7, linewidth=1.5)
        ax4.axhline(y=2, color="red", linestyle=":", alpha=0.7, label="±2σ bounds")
        ax4.axhline(y=-2, color="red", linestyle=":", alpha=0.7)

        cbar = plt.colorbar(ax4.collections[0], ax=ax4)
        cbar.set_label("Predicted Volatility", fontsize=9)

        ax4.set_title(
            "Standardized Residuals (Volatility-Adjusted)",
            fontsize=12,
            fontweight="bold",
        )
        ax4.set_xlabel("Time", fontsize=10)
        ax4.set_ylabel("Standardized Residuals", fontsize=10)
        ax4.legend(fontsize=9)
        ax4.grid(True, alpha=0.3)

        # ===== PLOT 5: ACF of residuals =====
        if len(residuals.dropna()) > 10:
            plot_acf(
                residuals.dropna(),
                zero=False,
                lags=min(40, len(residuals.dropna()) // 4),
                ax=ax5,
            )
            for line in ax5.lines:
                line.set_markersize(3)
            ax5.set_title(
                "Residual Autocorrelation", fontsize=12, fontweight="bold"
            )

        # ===== PLOT 6: Volatility time series =====
        ax6.plot(
            dates,
            vol_bands_df[ticker],
            color="red",
            linewidth=2,
            alpha=0.8,
            label="Predicted Volatility",
        )

        # Add realized volatility if we can compute it
        volatiliy_window = 20
        if len(actual_returns) > volatiliy_window:
            realized_vol_series = actual_returns.rolling(window=volatiliy_window).std() * 100
            realized_vol_series.dropna(inplace=True)
            ax6.plot(
                realized_vol_series.index,
                realized_vol_series,
                color="blue",
                linewidth=1.5,
                alpha=0.7,
                label=f"Realized Volatility ({volatiliy_window}-day)",
            )

        ax6.fill_between(dates, 0, vol_bands_df[ticker], alpha=0.3, color="red")
        ax6.set_title("Volatility Evolution", fontsize=12, fontweight="bold")
        ax6.set_xlabel("Time", fontsize=11)
        ax6.set_ylabel("Volatility (%)", fontsize=11)
        ax6.legend(fontsize=10)
        ax6.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()
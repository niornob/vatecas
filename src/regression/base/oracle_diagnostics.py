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


def oracle_diagnostics(
    oracle: "Oracle", data: dict[str, pd.Series], window: int
) -> None:
    """Generate enhanced comparison plots including variance bands and market factor analysis."""
    actual_df = pd.DataFrame(data)
    preds_df, vol_bands_df, market_factor = oracle.regress(data, window=window)
    preds_df = pd.DataFrame(preds_df)
    vol_bands_df = pd.DataFrame(vol_bands_df)

    dates = actual_df.index
    tickers = actual_df.columns

    # Define low data period (first 'window' periods where predictions are unreliable)
    low_data_period = window

    for ticker in tickers:
        # Calculate returns for both actual and predicted prices
        actual_returns = actual_df[ticker].pct_change().dropna()
        pred_returns = preds_df[ticker].pct_change().dropna()

        # Create enhanced figure with additional row for return analysis
        fig = plt.figure(figsize=(20, 24))  # Increased height for extra row
        gs = fig.add_gridspec(
            5, 2, height_ratios=[1, 1, 1, 0.8, 0.8], hspace=0.35, wspace=0.25
        )

        # Enhanced plot layout with new return analysis row:
        ax1 = fig.add_subplot(gs[0, :])  # Price predictions with confidence bands
        ax2 = fig.add_subplot(gs[1, 0])  # NEW: Return correlation scatter plot
        ax3 = fig.add_subplot(gs[1, 1])  # NEW: Sign correlation bar plot
        ax4 = fig.add_subplot(gs[2, 0])  # Volatility forecast accuracy
        ax5 = fig.add_subplot(gs[2, 1])  # Return distribution analysis
        ax6 = fig.add_subplot(gs[3, 0])  # Residuals with volatility context
        ax7 = fig.add_subplot(gs[3, 1])  # ACF of residuals
        ax8 = fig.add_subplot(gs[4, :])  # Volatility time series

        fig.suptitle(
            f"{oracle.name} Enhanced Analysis: {ticker}",
            fontsize=18,
            fontweight="bold",
        )

        # ===== PLOT 1: Price predictions with confidence bands =====
        # Your existing plotting code with added background coloring
        ax1.plot(
            dates,
            actual_df[ticker],
            label=f"Actual {ticker}",
            linewidth=1,
            alpha=0.9,
            color="navy",
        )

        # Main prediction line
        ax1.plot(
            dates,
            preds_df[ticker],
            label=f"Predicted {ticker}",
            color="red",
            linestyle="-",
            linewidth=1,
            alpha=0.8,
        )

        # Calculate returns (period-to-period changes)
        actual_returns = actual_df[ticker].diff().astype(float)  # This gives us the change from t to t+1
        pred_returns = preds_df[ticker].diff().astype(float)     # Same for predictions

        # Create boolean masks for vectorized operations (more efficient and avoids type issues)
        # Drop the first NaN value that diff() creates
        actual_returns_clean = actual_returns.dropna()
        pred_returns_clean = pred_returns.dropna()

        # Ensure both series have the same length after cleaning
        idx = actual_returns_clean.index.intersection(list(pred_returns_clean.index))
        actual_returns_clean = actual_returns_clean.loc[idx]
        pred_returns_clean = pred_returns_clean.loc[idx]

        # Calculate signs: True for positive/zero, False for negative
        actual_signs = actual_returns_clean >= 0
        pred_signs = pred_returns_clean >= 0

        # Find where signs agree (both positive or both negative)
        signs_agree = actual_signs == pred_signs

        # Add background coloring based on return sign agreement
        # We start from index 1 in the original dates because first return represents change from date[0] to date[1]
        for i, j, agree in zip(idx[:-1], idx[1:], signs_agree[1:]):            
            # Choose color based on whether the returns moved in the same direction
            if agree:
                # Green when both returns have same sign (both positive or both negative)
                color = 'lightgreen'
                alpha = 0.2
            else:
                # Red when returns have opposite signs
                color = 'lightcoral'
                alpha = 0.2
            
            # Add vertical span for this time period
            # We color from dates[date_idx-1] to dates[date_idx] since the return represents change between these points
            ax1.axvspan(i, j, color=color, alpha=alpha, zorder=0, linewidth=.5)

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
            alpha=0.5,
            color="orange",
            label="±1σ Confidence",
        )

        # Gray out low data period
        if low_data_period > 0:
            y_min, y_max = ax1.get_ylim()
            ax1.axvspan(
                dates[0],
                dates[min(low_data_period - 1, len(dates) - 1)],
                alpha=0.3,
                color="gray",
                label="Low Data Period",
            )

        ax1.set_title(
            "Price Forecasts with Volatility-Based Confidence Intervals",
            fontsize=14,
            fontweight="bold",
        )
        ax1.set_xlabel("Time", fontsize=11)
        ax1.set_ylabel(f"{ticker} Price", fontsize=11)

        # Add legend entries for the background coloring
        from matplotlib.patches import Patch
        legend_elements = ax1.get_legend_handles_labels()
        legend_elements[0].extend([
            Patch(facecolor='lightgreen', alpha=0.2, label='Returns aligned'),
            Patch(facecolor='lightcoral', alpha=0.2, label='Returns opposed')
        ])
        ax1.legend(handles=legend_elements[0], fontsize=10)

        ax1.grid(True, alpha=0.3)

        # ===== PLOT 2: Return correlation scatter plot =====
        # Align returns for comparison (both start from day 2 due to pct_change)
        common_idx = actual_returns.dropna().index.intersection(
            list(pred_returns.dropna().index)
        )
        if len(common_idx) > 10:
            actual_ret_aligned = actual_returns.loc[common_idx]
            pred_ret_aligned = pred_returns.loc[common_idx]

            # Create scatter plot
            ax2.scatter(
                actual_ret_aligned, pred_ret_aligned, alpha=0.6, s=25, color="blue"
            )

            # Add perfect prediction line
            min_ret = min(actual_ret_aligned.min(), pred_ret_aligned.min())
            max_ret = max(actual_ret_aligned.max(), pred_ret_aligned.max())
            ax2.plot(
                [min_ret, max_ret], [min_ret, max_ret], "r--", alpha=0.7, linewidth=1.5
            )

            # Calculate and display correlation
            ret_corr, ret_p = pearsonr(actual_ret_aligned, pred_ret_aligned)
            ax2.text(
                0.05,
                0.95,
                f"Return Correlation: {ret_corr:.3f}\np-value: {ret_p:.3f}",
                transform=ax2.transAxes,
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8),
                verticalalignment="top",
            )

        ax2.set_title("Predicted vs Actual Returns", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Actual Returns", fontsize=10)
        ax2.set_ylabel("Predicted Returns", fontsize=10)
        ax2.grid(True, alpha=0.3)

        # ===== PLOT 3: Sign correlation bar plot =====
        if len(common_idx) > 10:
            actual_ret_aligned = actual_returns.loc[common_idx]
            pred_ret_aligned = pred_returns.loc[common_idx]

            # Calculate sign agreement
            actual_signs = np.sign(actual_ret_aligned)
            pred_signs = np.sign(pred_ret_aligned)

            # Count combinations for bar plot
            pos_actual_pos_pred = np.sum((actual_signs > 0) & (pred_signs > 0))
            pos_actual_neg_pred = np.sum((actual_signs > 0) & (pred_signs < 0))
            neg_actual_pos_pred = np.sum((actual_signs < 0) & (pred_signs > 0))
            neg_actual_neg_pred = np.sum((actual_signs < 0) & (pred_signs < 0))

            # Create bar plot
            x_pos = [0.8, 1.2]  # Positions for +1 group
            x_neg = [-1.2, -0.8]  # Positions for -1 group

            # Bars for positive actual returns
            ax3.bar(
                x_pos[0],
                pos_actual_pos_pred,
                width=0.3,
                color="green",
                alpha=0.7,
                label="Pred +",
            )
            ax3.bar(
                x_pos[1],
                pos_actual_neg_pred,
                width=0.3,
                color="red",
                alpha=0.7,
                label="Pred -",
            )

            # Bars for negative actual returns
            ax3.bar(x_neg[0], neg_actual_neg_pred, width=0.3, color="green", alpha=0.7)
            ax3.bar(x_neg[1], neg_actual_pos_pred, width=0.3, color="red", alpha=0.7)

            # Calculate sign correlation
            sign_corr, sign_p = pearsonr(actual_signs, pred_signs)

            # Add text with correlation
            ax3.text(
                0.05,
                0.95,
                f"Sign Correlation: {sign_corr:.3f}\np-value: {sign_p:.3f}",
                transform=ax3.transAxes,
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
                verticalalignment="top",
            )

            # Formatting
            ax3.set_xticks([-1, 1])
            ax3.set_xticklabels(["Actual -", "Actual +"])
            ax3.set_ylabel("Count", fontsize=10)
            ax3.set_title(
                "Return Sign Prediction Accuracy", fontsize=12, fontweight="bold"
            )
            #ax3.legend(fontsize=9)
            ax3.grid(True, alpha=0.3)

        # ===== PLOT 4: Volatility forecast accuracy =====
        # Compare predicted volatility vs realized volatility
        actual_returns_full = actual_df[ticker].pct_change().dropna()
        realized_vol = (
            actual_returns_full.rolling(window=5).std().shift(-2)
        )  # Forward-looking realized vol
        predicted_vol = vol_bands_df[ticker].dropna() / 100  # Convert to decimal

        # Align series for comparison
        common_idx_vol = realized_vol.dropna().index.intersection(
            list(predicted_vol.index)
        )
        if len(common_idx_vol) > 10:
            realized_aligned = realized_vol.loc[common_idx_vol]
            predicted_aligned = predicted_vol.loc[common_idx_vol]

            ax4.scatter(realized_aligned, predicted_aligned, alpha=0.6, s=25)

            # Add perfect prediction line
            min_vol = min(realized_aligned.min(), predicted_aligned.min())
            max_vol = max(realized_aligned.max(), predicted_aligned.max())
            ax4.plot(
                [min_vol, max_vol],
                [min_vol, max_vol],
                "r--",
                alpha=0.7,
                linewidth=1.5,
            )

            # Calculate and display correlation
            vol_corr, vol_p = pearsonr(realized_aligned, predicted_aligned)
            ax4.text(
                0.05,
                0.95,
                f"Vol Correlation: {vol_corr:.3f}\np-value: {vol_p:.3f}",
                transform=ax4.transAxes,
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcyan", alpha=0.8),
                verticalalignment="top",
            )

        ax4.set_title("Volatility Forecast Accuracy", fontsize=12, fontweight="bold")
        ax4.set_xlabel("Realized Volatility", fontsize=10)
        ax4.set_ylabel("Predicted Volatility", fontsize=10)
        ax4.grid(True, alpha=0.3)

        # ===== PLOT 5: Return distribution analysis =====
        if len(actual_returns_full) > 10:
            ax5.hist(
                actual_returns_full,
                bins=100,
                alpha=0.7,
                color="skyblue",
                density=True,
                label="Actual Returns",
            )

            # Overlay normal distribution based on mean predicted volatility
            mean_pred_vol = vol_bands_df[ticker].mean() / 100
            x_norm = np.linspace(
                actual_returns_full.min(), actual_returns_full.max(), 100
            )
            y_norm = (1 / (mean_pred_vol * np.sqrt(2 * np.pi))) * np.exp(
                -0.5 * (x_norm / mean_pred_vol) ** 2
            )
            ax5.plot(x_norm, y_norm, "orange", linewidth=2, label="Normal (Pred Vol)")

        ax5.set_title("Return Distribution vs Normal", fontsize=12, fontweight="bold")
        ax5.set_xlabel("Returns", fontsize=10)
        ax5.set_ylabel("Density", fontsize=10)
        ax5.legend(fontsize=9)
        ax5.grid(True, alpha=0.3)

        # ===== PLOT 6: Residuals with volatility context =====
        residuals = actual_df[ticker] - preds_df[ticker]
        standardized_residuals = residuals / vol_bands_df[ticker]

        scatter = ax6.scatter(
            residuals.index,
            standardized_residuals,
            c=vol_bands_df[ticker],
            cmap="viridis",
            s=25,
            alpha=0.7,
        )
        ax6.axhline(y=0, color="black", linestyle="--", alpha=0.7, linewidth=1.5)
        ax6.axhline(y=2, color="red", linestyle=":", alpha=0.7, label="±2σ bounds")
        ax6.axhline(y=-2, color="red", linestyle=":", alpha=0.7)

        # Gray out low data period
        if low_data_period > 0:
            y_min, y_max = ax6.get_ylim()
            ax6.axvspan(
                dates[0],
                dates[min(low_data_period - 1, len(dates) - 1)],
                alpha=0.3,
                color="gray",
            )

        cbar = plt.colorbar(scatter, ax=ax6)
        cbar.set_label("Predicted Volatility", fontsize=9)

        ax6.set_title(
            "Standardized Residuals (Volatility-Adjusted)",
            fontsize=12,
            fontweight="bold",
        )
        ax6.set_xlabel("Time", fontsize=10)
        ax6.set_ylabel("Standardized Residuals", fontsize=10)
        ax6.legend(fontsize=9)
        ax6.grid(True, alpha=0.3)

        # ===== PLOT 7: ACF of residuals =====
        if len(residuals.dropna()) > 10:
            plot_acf(
                residuals.dropna(),
                zero=False,
                lags=min(40, len(residuals.dropna()) // 4),
                ax=ax7,
            )
            for line in ax7.lines:
                line.set_markersize(3)
            ax7.set_title("Residual Autocorrelation", fontsize=12, fontweight="bold")

        # ===== PLOT 8: Volatility time series =====
        ax8.plot(
            dates,
            vol_bands_df[ticker],
            color="red",
            linewidth=2,
            alpha=0.8,
            label="Predicted Volatility",
        )

        # Add realized volatility if we can compute it
        volatility_window = 5
        if len(actual_returns_full) > volatility_window:
            realized_vol_series = (
                actual_returns_full.rolling(window=volatility_window).std() * 100
            )
            realized_vol_series.dropna(inplace=True)
            ax8.plot(
                realized_vol_series.index,
                realized_vol_series,
                color="blue",
                linewidth=1.5,
                alpha=0.7,
                label=f"Realized Volatility ({volatility_window}-day)",
            )

        # Gray out low data period
        if low_data_period > 0:
            y_min, y_max = ax8.get_ylim()
            ax8.axvspan(
                dates[0],
                dates[min(low_data_period - 1, len(dates) - 1)],
                alpha=0.3,
                color="gray",
            )

        ax8.set_title("Volatility Evolution", fontsize=12, fontweight="bold")
        ax8.set_xlabel("Time", fontsize=11)
        ax8.set_ylabel("Volatility (%)", fontsize=11)
        ax8.legend(fontsize=10)
        ax8.grid(True, alpha=0.3)

        plt.show()

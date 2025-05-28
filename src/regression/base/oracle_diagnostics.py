import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from scipy.stats import pearsonr, linregress
from statsmodels.graphics.tsaplots import plot_acf
from typing import cast
from tqdm import tqdm

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parents[2]))
from regression.base.oracle import Oracle
from regression.volatility.GARCHpq import garch_series


def scatter_corr(
    X: pd.Series,
    Y: pd.Series,
    ax: Axes,
    correlation_of: str = "",
    plot_title: str = "",
    x_axis_label: str = "",
    y_axis_label: str = "",
):
    common_idx = X.dropna().index.intersection(list(Y.dropna().index))
    if len(common_idx) > 10:
        X_aligned = X.loc[common_idx]
        Y_aligned = Y.loc[common_idx]

        # Create scatter plot
        ax.scatter(X_aligned, Y_aligned, alpha=0.6, s=25)

        # Add perfect prediction line
        min_ret = min(X_aligned.min(), Y_aligned.min())
        max_ret = max(X_aligned.max(), Y_aligned.max())
        ax.plot([min_ret, max_ret], [min_ret, max_ret], "r--", alpha=0.7, linewidth=1)

        # Add MSE minimizing straight line
        slope, intercept, r_value, p_value, std_err = linregress(X_aligned, Y_aligned)
        reg_y = intercept + slope * np.array(list(X_aligned))
        ax.plot(
            X_aligned, reg_y, "b-", label=f"Regression: y={slope:.2f}x+{intercept:.2e}"
        )

        # draw the x-axis
        ax.axhline(y=cast(float, np.mean(Y)), color="black", linewidth=1.0, linestyle="--")

        # Calculate and display correlation
        ret_corr, ret_p = pearsonr(X_aligned, Y_aligned)
        ax.text(
            0.05,
            0.95,
            f"{correlation_of} Correlation: {ret_corr:.3f}\np-value: {ret_p:.3f}",
            transform=ax.transAxes,
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
            verticalalignment="top",
        )

    ax.set_title(plot_title, fontsize=12, fontweight="bold")
    ax.set_xlabel(x_axis_label, fontsize=10)
    ax.set_ylabel(y_axis_label, fontsize=10)
    ax.grid(True, alpha=0.3)


def oracle_diagnostics(
    oracle: "Oracle",
    data: dict[str, pd.Series],
    regress_window: int,
    smoothing_window: int = 1,
) -> None:
    """Generate enhanced comparison plots including variance bands and market factor analysis."""
    preds_df, vol_bands_df, market_factor = oracle.regress(data, window=regress_window)

    # convert data and prediction to dataframe for easier handling
    actual_df = pd.DataFrame(data)
    preds_df = pd.DataFrame(preds_df)
    vol_bands_df = pd.DataFrame(vol_bands_df)

    # replace data and prediction by their rolling means over smoothing_window days
    actual_df = actual_df.rolling(window=smoothing_window, min_periods=1).mean().dropna()
    preds_df = preds_df.rolling(window=smoothing_window, min_periods=1).mean().dropna()

    # compute volatiliy in data and in prediction for later comparison
    data_vol = {}
    pred_vol = {}
    for ticker in tqdm(data, desc=f"Computing volatility."):
        #data_vol[ticker] = garch_series(1e3 * actual_df[ticker].pct_change().dropna()) / 10
        pred_vol[ticker] = garch_series(1e3 * preds_df[ticker].pct_change().dropna()) / 10

    # is this how you change the variance when you take rolling average of the underlying series?
    vol_bands_df = vol_bands_df / smoothing_window

    dates = actual_df.index
    tickers = actual_df.columns

    # Define low data period (first 'window' periods where predictions are unreliable)
    low_data_period = regress_window

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
            f"{oracle.name} Analysis: {ticker} (applied {smoothing_window}-days rolling avg.)",
            fontsize=18,
            fontweight="bold",
        )

        # ===== PLOT 1: Price predictions with confidence bands =====
        # Your existing plotting code with added background coloring
        ax1.plot(
            actual_df[ticker].index,
            actual_df[ticker],
            label=f"Actual {ticker}",
            linewidth=1,
            alpha=0.9,
            color="navy",
        )

        # Main prediction line
        ax1.plot(
            preds_df[ticker].index,
            preds_df[ticker],
            label=f"Predicted {ticker}",
            color="red",
            linestyle="-",
            linewidth=1,
            alpha=0.8,
        )

        # Calculate returns (period-to-period changes)
        actual_returns = (
            actual_df[ticker].diff().astype(float)
        )  # This gives us the change from t to t+1
        pred_returns = preds_df[ticker].diff().astype(float)  # Same for predictions

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
                color = "lightgreen"
                alpha = 0.2
            else:
                # Red when returns have opposite signs
                color = "lightcoral"
                alpha = 0.2

            # Add vertical span for this time period
            # We color from dates[date_idx-1] to dates[date_idx] since the return represents change between these points
            ax1.axvspan(i, j, color=color, alpha=alpha, zorder=0, linewidth=0.5)

        # Confidence bands (±1 and ±2 standard deviations)
        # vol_bands_df contains volatility or std deviation of percentage returns
        # we want convert that to a band of price values.
        upper_1std = preds_df[ticker] * (1 + vol_bands_df[ticker] / 100)
        lower_1std = preds_df[ticker] * (1 - vol_bands_df[ticker] / 100)
        upper_2std = preds_df[ticker] * (1 + 2 * vol_bands_df[ticker] / 100)
        lower_2std = preds_df[ticker] * (1 - 2 * vol_bands_df[ticker] / 100)

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
        legend_elements[0].extend(
            [
                Patch(facecolor="lightgreen", alpha=0.2, label="Returns aligned"),
                Patch(facecolor="lightcoral", alpha=0.2, label="Returns opposed"),
            ]
        )
        ax1.legend(handles=legend_elements[0], fontsize=10)

        ax1.grid(True, alpha=0.3)

        # ===== PLOT 2: Return correlation scatter plot =====
        # Align returns for comparison (both start from day 2 due to pct_change)
        scatter_corr(
            X=actual_returns,
            Y=pred_returns,
            ax=ax2,
            correlation_of="Return",
            plot_title="Predicted vs Actual Returns",
            x_axis_label="Actual Returns",
            y_axis_label="Predicted Returns",
        )

        # ===== PLOT 3: Sign correlation bar plot =====
        common_idx = actual_returns.dropna().index.intersection(
            list(pred_returns.dropna().index)
        )
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
                0.35,
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
            # ax3.legend(fontsize=9)
            ax3.grid(True, alpha=0.3)

        # ===== PLOT 4: Volatility forecast accuracy =====
        # Compare predicted volatility vs realized volatility
        actual_returns_full = actual_df[ticker].pct_change().dropna()
        realized_vol = (
            actual_returns_full.rolling(window=5).std().shift(-2)
        )  # Forward-looking realized vol
        predicted_vol = vol_bands_df[ticker].dropna() / 100  # Convert to decimal

        scatter_corr(
            X=realized_vol,
            Y=predicted_vol,
            ax=ax4,
            correlation_of="Volatility",
            plot_title="Predicted vs Actual Volatility",
            x_axis_label="Actual Volatility",
            y_axis_label="Predicted Volatility"
        )

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
            pred_vol[ticker].index,
            pred_vol[ticker],
            color="red",
            linewidth=2,
            alpha=0.8,
            label="Predicted Volatility (GARCH)",
        )

        # Add realized volatility if we can compute it
        """
        ax8.plot(
            data_vol[ticker].index,
            data_vol[ticker],
            color="blue",
            linewidth=1.5,
            alpha=0.7,
            label=f"Realized Volatility",
        )
        """
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
                label=f"Realized Volatility (StDev: {volatility_window}-day)",
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

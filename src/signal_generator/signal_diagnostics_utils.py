import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from typing import List
import pandas as pd

def create_signal_colormap() -> LinearSegmentedColormap:
    """
    Returns a LinearSegmentedColormap that goes from dark red → white → dark green.
    """
    colors = [
        "darkred",
        "red",
        "lightcoral",
        "white",
        "lightgreen",
        "green",
        "darkgreen",
    ]
    return LinearSegmentedColormap.from_list("signal_strength", colors, N=100)


def plot_ticker_with_heatmap(
    ax: Axes,
    dates: List[pd.Timestamp],
    actual_prices: List[float],
    predicted_prices: np.ndarray,
    signals: np.ndarray,
    cmap: LinearSegmentedColormap
) -> None:
    """
    Draws, on a given Axes:
      1. A heatmap in the background whose color at each date is determined by 'signals'.
      2. A black line of actual_prices over time.
      3. A blue line of predicted_prices over time.
      4. A small text box in the top‐left corner showing μ, σ of 'signals' and MAE of predictions.

    Args:
      ax: a matplotlib Axes to draw into.
      dates: list of pd.Timestamp (length n_dates).
      actual_prices: list of float (length n_dates).
      predicted_prices: np.ndarray (length n_dates).
      signals: np.ndarray of shape (n_dates,). Values in [–1, +1].
      cmap: a LinearSegmentedColormap mapping [–1..+1] → RGBA.
    """
    n = len(dates)
    # 1) Compute heatmap grid
    all_prices = actual_prices + list(predicted_prices)
    p_min, p_max = min(all_prices), max(all_prices)
    padding = (p_max - p_min) * 0.1
    y_bottom, y_top = p_min - padding, p_max + padding

    x_coords = np.arange(n)
    y_coords = np.linspace(y_bottom, y_top, 100)
    X, Y = np.meshgrid(x_coords, y_coords)

    # Build Z so that each column is constant = signals[time]
    Z = np.zeros_like(X, dtype=float)
    for t_idx, sig in enumerate(signals):
        Z[:, t_idx] = sig

    # 2) Plot the heatmap
    im = ax.imshow(
        Z,
        extent=(0, n - 1, y_bottom, y_top),
        aspect="auto",
        cmap=cmap,
        vmin=-1,
        vmax=1,
        alpha=0.8,
        origin="lower",
    )

    # 3) Overlay actual price (black) and predicted price (blue)
    ax.plot(range(n), actual_prices, color="black", linewidth=1, alpha=0.9, label="Actual Price")
    ax.plot(range(n), predicted_prices, color="blue", linewidth=1, alpha=0.9, linestyle="-", label="Predicted Price")

    # 4) Format x‐axis using helper
    format_axes_dates(ax, dates)

    # 5) Y‐axis label
    ax.set_ylabel("Price")
    ax.set_ylim(y_bottom, y_top)
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 6) Text box with signal stats & prediction error
    pred_error = float(np.mean(np.abs(np.array(actual_prices) - predicted_prices)))
    sig_mean, sig_std = float(np.mean(signals)), float(np.std(signals))
    stats_text = f"Signals: μ={sig_mean:.3f}, σ={sig_std:.3f}\nMAE: {pred_error:.3f}"
    ax.text(
        0.02,
        0.98,
        stats_text,
        transform=ax.transAxes,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        verticalalignment="top",
        fontsize=9,
    )

    # 7) Attach colorbar on the right of the entire figure if needed by the caller
    #    (Often, callers will create a separate colorbar, so we omit it here.)

def format_axes_dates(ax: Axes, dates: List[pd.Timestamp], max_labels: int = 10) -> None:
    """
    Sets x‐ticks & x‐ticklabels on `ax` so they show a subset of `dates`.
    Rotates them for readability.

    Args:
      ax: the Axes whose x‐axis you want to format.
      dates: full list of pd.Timestamp objects (length n).
      max_labels: maximum number of date labels on the x‐axis.
    """
    n = len(dates)
    step = max(1, n // max_labels)
    tick_positions = list(range(0, n, step))
    tick_labels = [dates[i].strftime("%Y-%m-%d") for i in tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")
    ax.set_xlabel("Date")

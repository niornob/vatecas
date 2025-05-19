"""
Graphing utilities intended to be used mainly by other modules.
"""
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import cm, rcParams
from matplotlib.colors import ListedColormap
from cycler import cycler

"""
Colormap Name	Number of Colors	        Description
Accent	                8           Bold colors with high contrast
Dark2	                8	        Darker hues, good for visibility
Paired	                12	        Pairs of light/dark complementary colors
Pastel1	                9	        Soft, pale pastels
Pastel2	                8	        Slightly stronger pastels
Set1	                9	        Bright and saturated, not colorblind-safe
Set2	                8	        Muted but clear
Set3	                12	        Bright and diverse, useful for many classes
tab10	                10	        Default Matplotlib color cycle
tab20	                20	        Extension of tab10 with more variation
tab20b	                20	        Blues-based variation of tab20
tab20c	                20	        Cooler/greener variation of tab20
"""
cmap = cm.get_cmap('tab20')
if isinstance(cmap, ListedColormap):
    colors = cmap.colors  # Now recognized by Pylance
else:
    raise TypeError("Expected a ListedColormap with discrete colors.")
rcParams['axes.prop_cycle'] = cycler(color=colors)

def _equity_vs_benchmark(
    equity: pd.Series,
    data: dict[str, pd.DataFrame],
    benchmarks: list[str],
    initial_time: pd.Timestamp,
    price: str = "adjClose",
    title: str = "",
) -> None:
    """
    Plots equity curve along with benchmark price series normalized to match equity
    at the corresponding start time by converting pandas objects to NumPy arrays.

    Parameters:
        equity: Time-indexed series of portfolio equity.
        data: Dictionary mapping ticker symbols to time-indexed DataFrames with at least a column `price`.
        benchmarks: List of ticker symbols (keys of `data`) to use as benchmarks.
        price: Column name in each DataFrame to use for benchmark price (default: 'adjClose').
    """
    plt.figure(figsize=(12, 6))

    # Ensure we have sorted, time-indexed data
    equity = equity.sort_index()

    # Convert index and values to NumPy arrays for type compatibility
    x_eq = equity.index.to_numpy()
    y_eq = equity.to_numpy()
    plt.plot(x_eq, y_eq, label="Equity", color="black", linewidth=2)

    # Plot each benchmark, normalized
    for ticker in benchmarks:
        df = data[ticker].sort_index().loc[initial_time:]
        other_idx_list = df.index.to_list()
        common_idx = equity.index.intersection(other_idx_list)
        if common_idx.empty:
            continue

        # pick the first common timestamp
        t0 = common_idx[0]
        equity_t0 = equity.loc[t0]
        price_t0 = df.loc[t0, price]
        if price_t0 == 0:
            continue

        # normalization and re-alignment
        normalized = df[price] * (equity_t0 / price_t0)
        normalized = normalized.reindex(common_idx)

        # again convert to NumPy arrays
        x_bm = common_idx.to_numpy()
        y_bm = normalized.to_numpy()
        plt.plot(x_bm, y_bm, label=ticker, linestyle="--")

    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def _holdings_over_time(
    equity: pd.Series,
    data: dict[str, pd.DataFrame],
    holdings: dict[str, pd.Series],
    price: str = "adjClose",
    title: str = "",
):
    """
    Plot the percentage of portfolio equity invested in each ticker over time.

    Parameters
    ----------
    equity : pd.Series
        Time-indexed portfolio equity curve.
    data : Dict[str, pd.DataFrame]
        Dict of time-indexed DataFrames for each ticker, must contain `price` column.
    holdings : Dict[str, pd.Series]
        Dict of time-indexed Series giving number of shares held per ticker.
    price : str
        Column name in each DataFrame to use for valuation (e.g. "Close", "adjClose").
    """
    # Align all indexes
    idx = equity.index

    # Build a DataFrame of position values
    value_df = pd.DataFrame(index=idx)
    for ticker, h in holdings.items():
        # align holdings and prices to the equity index
        h_aligned = h.reindex(idx).ffill().fillna(0)
        p = data[ticker][price].reindex(idx).ffill().fillna(0)
        value_df[ticker] = h_aligned * p

    # Compute percentages
    pct = value_df.div(equity, axis=0).fillna(0)

    # Plot as stacked area
    ax = pct.plot.area(figsize=(12, 4), title=title, linewidth=0)
    ax.set_ylabel("Fraction of Portfolio")
    ax.set_xlabel("Date")
    ax.set_ylim(0, 1)
    ax.legend(title="Ticker", loc="upper left", bbox_to_anchor=(1.0, 1.0))
    plt.tight_layout()
    plt.grid(True)
    plt.show()

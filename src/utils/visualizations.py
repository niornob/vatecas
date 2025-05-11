"""
    Graphing utilities intended to be used mainly by other modules.
"""

import matplotlib.pyplot as plt
import pandas as pd

def _equity_vs_benchmark(
    equity: pd.Series,
    data: dict[str, pd.DataFrame],
    benchmarks: list[str],
    initial_time: pd.Timestamp,
    price: str = 'adjClose',
    title: str = ''
) -> None:
    """
    Plots equity curve along with benchmark price series normalized to match equity at the corresponding start time.

    Parameters:
        equity: Time-indexed series of portfolio equity.
        data: Dictionary mapping ticker symbols to time-indexed DataFrames with at least a column `price`.
        benchmarks: List of ticker symbols (keys of `data`) to use as benchmarks.
        price: Column name in each DataFrame to use for benchmark price (default: 'adjClose').
    """
    plt.figure(figsize=(12, 6))
    
    # Plot equity
    equity = equity.sort_index()
    plt.plot(equity.index, equity.values, label='Equity', color='black', linewidth=2)

    # Plot each benchmark, normalized
    for ticker in benchmarks:
        df = data[ticker].sort_index().loc[initial_time:]
        if price not in df.columns:
            raise ValueError(f"Price column '{price}' not found in data for ticker '{ticker}'.")

        # Find the first common index between equity and benchmark
        common_idx = equity.index.intersection(df.index)
        if common_idx.empty:
            print(f"Skipping {ticker}: no overlapping time index with equity.")
            continue

        t0 = common_idx[0]
        equity_t0 = equity.loc[t0]
        price_t0 = df.loc[t0, price]

        if price_t0 == 0:
            print(f"Skipping {ticker}: price at t0 is zero.")
            continue

        # Normalize benchmark to match equity at t0
        normalized_price = df[price] * (equity_t0 / price_t0)
        plt.plot(df.index, normalized_price, label=f'{ticker}', linestyle='--')

    plt.xlabel('Date')
    plt.ylabel('Value')
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
    title: str = ''
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
    ax = pct.plot.area(
        figsize=(12, 6),
        title=title,
        linewidth=0
    )
    ax.set_ylabel("Fraction of Portfolio")
    ax.set_xlabel("Date")
    ax.set_ylim(0, 1)
    ax.legend(title="Ticker", loc='upper left', bbox_to_anchor=(1.0, 1.0))
    plt.tight_layout()
    plt.grid(True)
    plt.show()
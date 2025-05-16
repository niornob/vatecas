import sys
import json
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import pickle
import os

# Determine base directory (directory of this script)
BASE_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = BASE_DIR.parents[1]  # vatecas/src
DATA_DIR = PACKAGE_ROOT / "data"
SRC_DIR = PACKAGE_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Relative paths under the recommender package
PORTFOLIO_PATH = BASE_DIR / "portfolio" / "portfolio.json"
RECOMMENDATION_PATH = BASE_DIR / "portfolio" / "recommendation.json"

# Import platform classes
from investor.portfolio import Portfolio, FixedFractionalSizing
from signal_modules.signal_registry import SignalRegistry


def load_portfolio(path: Path) -> Portfolio:
    """
    Read portfolio configuration from JSON and instantiate Portfolio.
    """
    with path.open("r") as f:
        cfg = json.load(f)

    frac = cfg.get("sizing_fraction", 0.1)

    # Build Portfolio
    portfolio = Portfolio(
        capital=cfg.get("cash", 0.0),
        max_position_size=cfg.get("max_position_size", 0.0),
        sizing_model=FixedFractionalSizing(fraction=frac),
        commission=cfg.get("commission", 0.0),
        slippage=cfg.get("slippage", 0.0),
        initial_time=pd.to_datetime(
            cfg.get("initial_time", "2020-01-01T12:00:00+00:00")
        ),
        trades_per_period=cfg.get("trades_per_period", 0),
        period_length_days=cfg.get("period_length_days", 5),
        tau_max=cfg.get("tau_max", 1.0),
        tau_min=cfg.get("tau_min", 0.1),
        universe=cfg.get("universe", []),
    )

    # Restore positions and last_trade_times
    portfolio.positions = cfg.get("positions", {})
    raw = cfg.get("last_trade_times", {})
    portfolio.last_trade_times = {
        tk: portfolio.initial_time if tk not in raw else pd.to_datetime(raw[tk])
        for tk in portfolio.positions.keys()
    }
    return portfolio


def load_data(data_dir: Path, universe: list[str]) -> dict[str, pd.DataFrame]:
    """
    Load per-ticker Parquet files from data_dir into a dict of DataFrames.
    Expects files named <TICKER>.parquet with tz-aware DateTime index.
    """
    data: dict[str, pd.DataFrame] = {}
    for path in data_dir.glob("*.parquet"):
        ticker = path.stem.upper()
        if ticker not in universe:
            continue
        df = pd.read_parquet(path)
        # Ensure index is datetime and tz-aware
        if not isinstance(df.index, pd.DatetimeIndex) or df.index.tz is None:
            df.index = pd.to_datetime(df.index).tz_localize("UTC")
        data[ticker] = df
    return data


def generate_recommendations(
    portfolio: Portfolio, data: dict[str, pd.DataFrame], current_time: pd.Timestamp
) -> dict[str, dict]:
    """
    Generate trade recommendations for each ticker using provided historical DataFrames.
    """
    # Prepare price map from the latest adjClose
    price_map = {
        tk: df["adjClose"].iloc[-1]
        for tk, df in data.items()
        if "adjClose" in df.columns and not df.empty
    }

    # Instantiate and run signal module
    signal_registry = SignalRegistry()
    model = signal_registry.get("Kalman")()
    raw_signals = model.generate_signals(data)  # Mapping[str, pd.Series] per ticker
    raw_signals = {tk: sig.iloc[-1] for tk, sig in raw_signals.items() if not sig.empty}
    confidences = {tk: 2 * sig - 1 for tk, sig in raw_signals.items()}
    confidences = dict(sorted(confidences.items(), key=lambda kv: (kv[1] >= 0, -kv[1] if kv[1] >= 0 else kv[1])))

    recs = portfolio.generate_order(confidences, price_map, current_time)

    return recs


def main():
    # Load portfolio state
    portfolio = load_portfolio(PORTFOLIO_PATH)

    # Load market data from Parquet files
    data = load_data(DATA_DIR, universe=portfolio.universe)

    timeline = pd.DatetimeIndex(
        pd.to_datetime(sorted(set().union(*[df.index for df in data.values()])))
    )
    aligned = {tk: df.reindex(timeline).ffill() for tk, df in data.items()}
    timeline = timeline[timeline >= portfolio.initial_time]
    aligned = {tk: df.loc[portfolio.initial_time :] for tk, df in aligned.items()}

    # Execution timestamp (UTC)
    now = pd.Timestamp(datetime.now(timezone.utc))

    # Generate recommendations using in-memory data
    orders = generate_recommendations(portfolio, aligned, now)
    recs_with_closing_price = {
        tk: {"size": order['size'], "closing_price": order['price']}
        for tk, order in orders.items()
    }

    print(recs_with_closing_price)

    # Write recommendation.json
    out = {"timestamp": now.isoformat(), "signals": recs_with_closing_price}
    with RECOMMENDATION_PATH.open("w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()

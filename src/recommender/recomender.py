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
from investor.portfolio_manager import PortfolioManager
from investor.atomic_types import Signal, Order, Position
from investor._sizing import FixedFractionalSizing
from signal_modules.signal_registry import SignalRegistry


def load_portfolio(path: Path) -> tuple[PortfolioManager, list[str]]:
    """
    Read portfolio configuration from JSON and instantiate Portfolio.
    """
    with path.open("r") as f:
        cfg = json.load(f)

    frac = cfg.get("sizing_fraction", 0.1)
    initial_time = pd.Timestamp(cfg.get("initial_time", "2000-01-01T12:00:00+00:00"))
    positions = {
        tk: Position(ticker=tk, size=sz, last_trade_time=initial_time)
        for tk, sz in cfg.get("positions", {}).items()
    }

    # Build Portfolio
    manager = PortfolioManager(
        initial_capital=cfg.get("cash", 0.0),
        initial_positions=positions,
        initial_time=initial_time,
        max_position_size=float(cfg.get("max_position_size", "inf")),
        commission=cfg.get("commission", 0.0),
        slippage=cfg.get("slippage", 0.0),
        sizing_model=FixedFractionalSizing(fraction=frac),
        trades_per_period=cfg.get("trades_per_period", 1),
        period_length_days=cfg.get("period_length_days", 5),
        tau_max=cfg.get("tau_max", 1.0),
        tau_min=cfg.get("tau_min", 0.1),
    )

    universe: list[str] = cfg.get("universe", [])

    return manager, universe


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
    manager: PortfolioManager,
    universe: list[str],
    data: dict[str, pd.DataFrame],
    current_time: pd.Timestamp,
) -> dict[str, Order]:
    """
    Generate trade recommendations for each ticker using provided historical DataFrames.
    """
    data = {tk: df for tk, df in data.items() if tk in universe}

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
    confidences = dict(
        sorted(
            confidences.items(),
            key=lambda kv: (kv[1] >= 0, -kv[1] if kv[1] >= 0 else kv[1]),
        )
    )
    signals: dict[str, Signal] = {
        tk: Signal(ticker=tk, value=sig, price=price_map[tk], timestamp=current_time)
        for tk, sig in confidences.items()
    }

    recs = manager.portfolio.order_generator.generate_orders(
        signals=signals,
        positions=manager.portfolio.positions,
        cash=manager.portfolio.cash,
        current_prices=price_map,
    )

    return recs


def main():
    # Load portfolio state
    manager, universe = load_portfolio(PORTFOLIO_PATH)

    # Load market data from Parquet files
    data = load_data(DATA_DIR, universe=universe)

    timeline = pd.DatetimeIndex(
        pd.to_datetime(sorted(set().union(*[df.index for df in data.values()])))
    )

    """
        L is the maximum number of past days, from which historica data is used to make
        signals at preset time. its minimum value depends on the type of signal generator.
        for Kalman
        L >= max(minimum num of assets in the universe, process_window (20 by default))
    """
    L = 1000
    aligned = {tk: df.reindex(timeline).ffill()  for tk, df in data.items()}

    # Execution timestamp (UTC)
    now = pd.Timestamp(datetime.now(timezone.utc))

    # Generate recommendations using in-memory data
    orders = generate_recommendations(
        manager=manager, universe=universe, data=aligned, current_time=now
    )

    recs_with_closing_price = {
        order.ticker: {"size": order.size, "closing_price": order.price}
        for order in orders.values()
    }

    print(recs_with_closing_price)

    # Write recommendation.json
    out = {"timestamp": now.isoformat(), "signals": recs_with_closing_price}
    with RECOMMENDATION_PATH.open("w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()

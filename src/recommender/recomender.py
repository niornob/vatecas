import sys
import json
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import numpy as np

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
from portfolio_management.portfolio_manager import PortfolioManager
from portfolio_management.atomic_types import Signal, Order, Position
from portfolio_management._sizing import FixedFractionalSizing
from signal_generator.signal_module import SignalModule
from signal_generator.regression.Kalman_filter import KalmanFilter


def load_portfolio(path: Path) -> tuple[PortfolioManager, list[str]]:
    """
    Read portfolio configuration from JSON and instantiate Portfolio.
    """
    with path.open("r") as f:
        cfg = json.load(f)

    frac = cfg.get("sizing_fraction", 0.1)
    initial_time = pd.Timestamp(cfg.get("initial_time", "2000-01-01T12:00:00+00:00"))
    last_trade_times = cfg.get("last_trade_times")
    positions = {
        tk: Position(
            ticker=tk,
            size=sz,
            last_trade_time=(
                pd.Timestamp(last_trade_times[tk])
                if tk in last_trade_times
                else initial_time
            ),
        )
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

    print("cash: ", manager.portfolio.cash)
    for pos in manager.portfolio.positions.values():
        print(pos.ticker, ":", pos.size, " (", pos.last_trade_time, ")")
    print("====================================")

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
    len_history = 40
    assert all(
        np.array([len(df) >= len_history for df in data.values()])
    ), f"not enough data for lookback period ({len_history})"

    all_dates = set()
    for df in data.values():
        all_dates.update(df.index)

    timeline = pd.DatetimeIndex(sorted(all_dates))

    data_aligned = {}
    for tk, df in data.items():
        df_aligned = df.reindex(timeline)
        df_aligned = df_aligned.ffill().bfill().fillna(0)

        data_aligned[tk] = df_aligned

    closing_data: dict[str, pd.Series] = {
        tk: df.iloc[-len_history:]["adjClose"]
        for tk, df in data_aligned.items()
        if tk in universe
    }

    # Prepare price map from the latest adjClose
    price_map: dict[str, float] = {
        tk: series.iloc[-1] for tk, series in closing_data.items()
    }

    # Instantiate and run signal module
    """
    completely empirically, low process_noise and high observation_noise_scale
    produces high gain. so keep it as it is until an explanation is found.
    """
    regressor = KalmanFilter(process_noise=1e3, obs_noise_scale=1e-3)
    model = SignalModule(oracle=regressor, smoothing_window=1, market_vol_target=5)

    raw_signals = model.generate_signals(closing_data)
    signals: dict[str, Signal] = {
        tk: Signal(ticker=tk, value=sig, price=price_map[tk], timestamp=current_time)
        for tk, sig in zip(raw_signals.tickers, raw_signals.signals)
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

    """
        L is the maximum number of past days, from which historica data is used to make
        signals at preset time. its minimum value depends on the type of signal generator.
        for Kalman
        L >= max(minimum num of assets in the universe, process_window (20 by default))
    """

    # Execution timestamp (UTC)
    now = pd.Timestamp(datetime.now(timezone.utc))

    # Generate recommendations using in-memory data
    orders = generate_recommendations(
        manager=manager, universe=universe, data=data, current_time=now
    )

    recs_with_closing_price = {
        order.ticker: {"size": order.size, "closing_price": order.price}
        for order in orders.values()
    }

    print("Recommendations:", recs_with_closing_price)

    # Write recommendation.json
    out = {"timestamp": now.isoformat(), "signals": recs_with_closing_price}
    with RECOMMENDATION_PATH.open("w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()

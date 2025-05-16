from dataclasses import dataclass
from typing import Mapping, Sequence, Tuple, Dict, List
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from investor.portfolio import Portfolio
from signal_modules.base import SignalModule

from utils.visualizations import _equity_vs_benchmark, _holdings_over_time


class BacktestEngine:
    def __init__(
        self,
        data: Mapping[str, pd.DataFrame],
        modules_weights: Sequence[Tuple[SignalModule, float]],
        portfolio: Portfolio,
    ):
        self.data = {tk: df.copy() for tk, df in data.items()}
        self.timeline = pd.DatetimeIndex(
            pd.to_datetime(
                sorted(set().union(*[df.index for df in self.data.values()]))
            )
        )
        self.aligned = {
            tk: df.reindex(self.timeline).ffill() for tk, df in self.data.items()
        }
        self.timeline = self.timeline[self.timeline >= portfolio.initial_time]
        self.aligned = {
            tk: df.loc[portfolio.initial_time :] for tk, df in self.aligned.items()
        }
        if not all(w >= 0 for _, w in modules_weights):
            raise ValueError("Non-sensical weight(s)")
        total_w = sum(w for _, w in modules_weights)
        self.modules_weights = (
            [(m, w / total_w) for m, w in modules_weights if w != 0]
            if total_w != 0
            else []
        )
        self.portfolio = portfolio

    def run(self) -> None:
        # Precompute weighted signal history
        raw: Dict[str, List[pd.Series]] = {}
        for module, weight in self.modules_weights:
            out = module.generate_signals(self.aligned)
            for tk, series in out.items():
                raw.setdefault(tk, []).append((2 * series - 1) * weight)
        net = {tk: pd.concat(lst).groupby(level=0).sum() for tk, lst in raw.items()}

        for t in tqdm(self.timeline, desc="Backtesting"):
            # extract and filter signals at t
            signals: Dict[str, float] = {}
            for tk, s in net.items():
                if t in s.index:
                    sig = s.loc[t]
                    # apply portfolio threshold logic
                    if abs(sig) >= self.portfolio.threshold(tk, t):
                        signals[tk] = sig
            # get open prices
            prices = {tk: self.aligned[tk]["adjOpen"].loc[t] for tk in signals}
            # execute and record trades
            self.portfolio.execute_orders(signals, prices, t)
            # record portfolio state
            price_map = {tk: self.aligned[tk]["adjClose"].loc[t] for tk in self.data}
            self.portfolio.record_state(price_map, t)

        return

    """
    The rest are plotting utilities.
    """

    def equity_vs_benchmarks(self, benchmarks: list[str]):
        _equity_vs_benchmark(
            equity=self.portfolio.get_results()["equity_curve"],
            data=self.aligned,
            benchmarks=benchmarks,
            initial_time=self.portfolio.initial_time,
            title="Equity vs. (normalized) Benchmarks",
        )

    def holdings_over_time(self):
        portfolio = self.portfolio.get_results()
        _holdings_over_time(
            equity=portfolio["equity_curve"],
            data=self.aligned,
            holdings=portfolio["holdings"],
            price="adjClose",
            title="Portfolio Allocation (% of Equity)",
        )

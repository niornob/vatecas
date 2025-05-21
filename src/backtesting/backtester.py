from typing import Mapping, Sequence, Tuple, Dict, List
import pandas as pd
from tqdm import tqdm

from investor.portfolio_manager import PortfolioManager
from signal_modules.base import SignalModule

from utils.visualizations import _equity_vs_benchmark, _holdings_over_time


class BacktestEngine:
    def __init__(
        self,
        data: Mapping[str, pd.DataFrame],
        modules_weights: Sequence[Tuple[SignalModule, float]],
        manager: PortfolioManager,
    ):
        """
        create a unified timeline by taking union of timelines from all tickers.
        align dataframes for all tickers to the unified timeline.
        """
        self.data = {tk: df.copy() for tk, df in data.items()}
        self.timeline = pd.DatetimeIndex(
            pd.to_datetime(
                sorted(set().union(*[df.index for df in self.data.values()]))
            )
        )
        self.aligned = {
            tk: df.reindex(self.timeline).ffill() for tk, df in self.data.items()
        }

        if not all(w >= 0 for _, w in modules_weights):
            raise ValueError("Non-sensical weight(s)")
        total_w = sum(w for _, w in modules_weights)
        self.modules_weights = (
            [(m, w / total_w) for m, w in modules_weights if w != 0]
            if total_w != 0
            else []
        )
        """
        timeline starts from initial portfolio time.
        aligned data keeps L days of extra history.
        the minimum value of L depends on the type of signal generator.
        for Kalman L >= max(minimum num of assets in the universe, process_window (20 by default)).
        """
        self.L = pd.Timedelta(days=50)
        self.timeline = self.timeline[self.timeline >= manager.portfolio.initial_time]
        self.aligned = {
            tk: df.loc[manager.portfolio.initial_time - self.L:] for tk, df in self.aligned.items()
        }
        
        self.manager = manager

    def run(self) -> None:
        for today in tqdm(self.timeline, desc="Backtesting"):
            raw: Dict[str, List[float]] = {}
            """
            get rid of future data.
            orders will be generated today,
            based on yesterday's closing prices.
            """
            yesterday = today - pd.Timedelta(days=1)
            truncated_data = {tk: df.loc[yesterday - self.L:yesterday] for tk, df in self.aligned.items()}
            for module, weight in self.modules_weights:
                out = module.generate_signals(truncated_data)
                for tk, sig in out.items():
                    raw.setdefault(tk, []).append((2 * sig - 1) * weight)
            signals: dict[str, float] = {tk: sum(lst) for tk, lst in raw.items()}

            # get today's opening prices
            opening_prices: dict[str, float] = {tk: self.aligned[tk]["adjOpen"].loc[today] for tk in signals}
            # execute and record trades at today's opening prices
            self.manager.process_signals(signals, opening_prices, today)

        return

    """
    The rest are plotting utilities.
    """

    def equity_vs_benchmarks(self, benchmarks: list[str]):
        #print(self.aligned)
        equity = self.manager.get_results()["equity_curve"]
        _equity_vs_benchmark(
            equity=equity,
            data=self.aligned,
            benchmarks=benchmarks,
            initial_time=self.manager.portfolio.initial_time,
            title="Equity vs. (normalized) Benchmarks",
        )

    def holdings_over_time(self):
        portfolio = self.manager.get_results()
        _holdings_over_time(
            equity=portfolio["equity_curve"],
            data=self.aligned,
            holdings=portfolio["holdings"],
            price="adjClose",
            title="Portfolio Allocation (% of Equity)",
        )

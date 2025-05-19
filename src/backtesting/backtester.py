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

        
        # unnecessary to compute signals for all of data
        # trucate to early enough so that at the provided initial_time
        # there's good signal.
        if not all(w >= 0 for _, w in modules_weights):
            raise ValueError("Non-sensical weight(s)")
        total_w = sum(w for _, w in modules_weights)
        self.modules_weights = (
            [(m, w / total_w) for m, w in modules_weights if w != 0]
            if total_w != 0
            else []
        )
        """
        trucate timeline, aligned data, and precomputed signals
        according to provided initial_time.
        """
        self.timeline = self.timeline[self.timeline >= manager.portfolio.initial_time]
        self.aligned = {
            tk: df.loc[manager.portfolio.initial_time :] for tk, df in self.aligned.items()
        }
        
        self.manager = manager

    def run(self) -> None:
        """
        L is the maximum number of past days, from which historica data is used to make
        signals at preset time. its minimum value depends on the type of signal generator.
        for Kalman
        L >= max(minimum num of assets in the universe, process_window (20 by default))
        """
        L = pd.Timedelta(days=50)
        for t in tqdm(self.timeline, desc="Backtesting"):
            raw: Dict[str, List[pd.Series]] = {}
            truncated_data = {tk: df.loc[t - L:t] for tk, df in self.aligned.items()}
            for module, weight in self.modules_weights:
                out = module.generate_signals(truncated_data)
                for tk, series in out.items():
                    raw.setdefault(tk, []).append((2 * series - 1) * weight)
            net: dict[str, pd.Series] = {tk: pd.concat(lst).groupby(level=0).sum() for tk, lst in raw.items()}

            # extract and filter signals at t
            signals: dict[str, float] = {tk: s.loc[t] for tk, s in net.items()}
            # get open prices
            opening_prices: dict[str, float] = {tk: self.aligned[tk]["adjOpen"].loc[t] for tk in signals}
            # execute and record trades
            self.manager.process_signals(signals, opening_prices, t)
            """
            orders are processed and portfolio snapshots are taken at market open.
            we may want to allow taking snapshots at market close.
            """

        return

    """
    The rest are plotting utilities.
    """

    def equity_vs_benchmarks(self, benchmarks: list[str]):
        #print(self.aligned)
        equity = self.manager.get_results()["equity_curve"]
        print(equity)
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

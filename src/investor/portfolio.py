import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Mapping, Sequence, Tuple, Dict, List
import copy
from math import floor


class SizingModel(ABC):
    """
    Abstract base class for position sizing logic.
    Subclasses define how confidence signals translate into number of shares.
    """
    @abstractmethod
    def size_position(self, signal: float, price: float, portfolio: "Portfolio") -> float:
        """
        signal: in the range [-1, 1]. <0 is a sell and >0 is a buy.
        """
        pass


class FixedFractionalSizing(SizingModel):
    """
    Allocates a fixed fraction of capital scaled by signal strength,
    and converts the resulting dollar amount into number of shares.
    """
    def __init__(self, fraction: float):
        if not 0 <= fraction <= 1:
            raise ValueError("Fraction must be between 0 and 1")
        self.fraction = fraction

    def size_position(self, signal: float, price: float, portfolio: "Portfolio") -> float:
        if price <= 0:
            return 0.0
        # use current cash for buys, current position value for sells
        base = portfolio.cash if signal > 0 else portfolio.pos_value
        dollar_amount = self.fraction * base * signal
        limit = portfolio.max_position_size
        dollar_amount = min(abs(dollar_amount), limit) * (1 if dollar_amount > 0 else -1)
        # assuming fractional shares can not be traded
        return round(dollar_amount / price)


@dataclass
class Portfolio:
    capital: float = 100000.0
    max_position_size: float = 0
    sizing_model: SizingModel = FixedFractionalSizing(fraction=.1)
    commission: float = 0.0
    slippage: float = 0.0
    universe: list[str] = field(default_factory=list)
    initial_time: pd.Timestamp = pd.Timestamp('2000-01-01T12', tz='UTC')

    # Frequency control parameters:
    trades_per_period: int = 3             # desired trades per period
    period_length_days: int = 15            # period in trading days
    tau_max: float = 1.0                   # threshold immediately after trade
    tau_min: float = 0.1                   # threshold after full decay
    """
    Setting decay to exponential requires setting tau_min to a small nonzero value.
    """
    decay: str = 'linear'                  # 'linear' or 'exponential'

    # Internal state:
    positions: dict[str, float] = field(default_factory=dict)
    last_trade_times: dict[str, pd.Timestamp] = field(default_factory=dict)
    # time-series records
    order_history: List[dict] = field(default_factory=list)
    timestamps: List[pd.Timestamp] = field(default_factory=list)
    equity_vals: List[float] = field(default_factory=list)
    cash_vals: List[float] = field(default_factory=list)
    holdings_vals: dict[str, List[float]] = field(default_factory=dict)

    def __post_init__(self):
        self.initial_cash: float = self.capital
        self.cash: float = self.capital
        self.positions = {} if self.positions is None else self.positions.copy()
        self.pos_value: float = 0.0
        self.order_history = []
        self.timestamps = []
        self.equity_vals = []
        self.cash_vals = []
        self.holdings_vals = {}
        # initialize last_trade_times far in the past
        self.last_trade_times = {}
        self._compute_decay_constant()

    def _compute_decay_constant(self):
        # average inter-trade interval
        self.dt_avg = self.period_length_days / max(self.trades_per_period, 1)
        if self.decay == 'exponential' and self.tau_min > 0:
            self.lambda_ = -np.log(self.tau_min / self.tau_max) / self.dt_avg
        else:
            self.lambda_ = -float('inf')

    def _threshold(self, ticker: str, current_time: pd.Timestamp) -> float:
        # compute days since last trade for this ticker
        last = self.last_trade_times.get(ticker, self.initial_time)
        delta = (current_time - last) / pd.Timedelta(days=1)
        if self.decay == 'linear':
            frac = min(delta / self.dt_avg, 1.0)
            return self.tau_max - (self.tau_max - self.tau_min) * frac
        else:  # exponential
            return self.tau_min + (self.tau_max - self.tau_min) * np.exp(-self.lambda_ * delta)
        
    def timed_size_position(self, ticker: str, signal: float, price: float, current_time: pd.Timestamp):
        thr = self._threshold(ticker, current_time)
        if abs(signal) < thr:
            return 0
        else:
            return self.sizing_model.size_position(signal, price, self)
        
    def generate_order(self, signals: Dict[str, float], prices: Dict[str, float], timestamp: pd.Timestamp):
        #print(signals)
        portfolio = copy.deepcopy(self)
        order = {}
        # execute sells first. then execute buys in descending order of signal strength.
        sorted_signals = sorted(signals.items(), key=lambda kv: (kv[1] >= 0, -kv[1] if kv[1] >= 0 else kv[1]))
        for ticker, signal in sorted_signals:
            # print(ticker, " ", portfolio.cash)
            price = prices.get(ticker)
            if price is None or price <= 0:
                continue
            size = portfolio.timed_size_position(ticker, signal, price, timestamp)
            if abs(size) < 1:
                continue
            current = portfolio.positions.get(ticker, 0.0)
            # long-only enforcement
            if size < 0:
                size = max(-current, size)
                if size == 0:
                    continue
            
            # compute signed dollar amount
            notional = price * size  
            # slippage is always positive
            slippage_cost = abs(notional) * portfolio.slippage  
            commission = portfolio.commission

            # unified total_cost: for buys (+notional) it’s positive, for sells (–notional) it’s negative
            total_cost = notional + slippage_cost + commission

            # if it’s a buy and we can’t afford it, shrink size to exactly exhaust cash–commission–1
            if size > 0 and total_cost > portfolio.cash:
                size = max(floor((portfolio.cash - commission - 1) / (price * (1 + portfolio.slippage))), 0)
                # re-compute everything with the new size
                notional = price * size
                slippage_cost = abs(notional) * portfolio.slippage
                total_cost = notional + slippage_cost + commission

            # if after adjustment there’s nothing to do, skip
            if size == 0:
                continue

            # now apply the trade: subtracting total_cost from cash
            # a positive total_cost (buy) reduces cash
            # a negative total_cost (sell) increases cash
            portfolio.cash -= total_cost

            order[ticker] = {
                'size': size, 
                'total_cost': total_cost, 
                'price': price, 
                'slippage': slippage_cost, 
                'commission': commission
            }
        return order

    def execute_orders(self, signals: Dict[str, float], prices: Dict[str, float], timestamp: pd.Timestamp):
        orders = self.generate_order(signals, prices, timestamp)
        for tk, order in orders.items():
            current = self.positions.get(tk, 0.0)
            self.positions[tk] = current + order['size']
            self.cash = self.cash - order['total_cost']
            self.last_trade_times[tk] = timestamp
            self.order_history.append({
                'ticker': tk,
                'timestamp': timestamp.isoformat(),
                'size': order['size'],
                'price': order['slippage'],
                'commission': order['commission']
            })

    def record_state(self, price_map: Dict[str, float], timestamp: pd.Timestamp):
        self.timestamps.append(timestamp)
        self.pos_value = sum(self.positions.get(tk, 0.0) * price_map.get(tk, 0.0)
                             for tk in price_map)
        eq = self.cash + self.pos_value
        self.equity_vals.append(eq)
        self.cash_vals.append(self.cash)
        for tk in price_map:
            if tk not in self.holdings_vals:
                self.holdings_vals[tk] = [0.0] * (len(self.timestamps)-1)
            self.holdings_vals[tk].append(self.positions.get(tk, 0.0))
        for tk, vals in self.holdings_vals.items():
            if len(vals) < len(self.timestamps):
                vals.append(0.0)

    def get_results(self) -> Mapping:
        # Handle empty portfolio (no recorded states)
        if not self.timestamps:
            empty_series = pd.Series(dtype=float)
            return {
                'equity_curve': empty_series,
                'cash_curve': empty_series,
                'positions_value_curve': empty_series,
                'returns': empty_series,
                'cumulative_return': 0.0,
                'sharpe_ratio': float('nan'),
                'max_drawdown': 0.0,
                'trades': self.order_history.copy(),
                'holdings': {},
                'final_cash': self.cash,
                'final_positions': self.positions.copy()
            }
        # Build pandas Series/DataFrames from history lists
        idx = pd.Index(self.timestamps)
        equity = pd.Series(self.equity_vals, index=idx)
        cash = pd.Series(self.cash_vals, index=idx)
        pos_val = equity - cash
        returns = equity.pct_change().fillna(0)
        # Compute metrics with safety checks
        cumulative = (equity.iloc[-1] / equity.iloc[0] - 1) if equity.iloc[0] != 0 else float('nan')
        sharpe = (returns.mean() * (252**0.5) / returns.std()) if returns.std() else float('nan')
        max_dd = ((equity - equity.cummax()) / equity.cummax()).min() if not equity.empty else 0.0
        holdings = {tk: pd.Series(vals, index=idx)
                    for tk, vals in self.holdings_vals.items()}
        return {
            'equity_curve': equity,
            'cash_curve': cash,
            'positions_value_curve': pos_val,
            'returns': returns,
            'cumulative_return': cumulative,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'trades': self.order_history.copy(),
            'holdings': holdings,
            'final_cash': self.cash,
            'final_positions': self.positions.copy()
        }

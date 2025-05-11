# portfolio.py
import pandas as pd
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
from typing import Dict, List, Mapping, Optional, Type
import yaml
from datetime import datetime

@dataclass
class Order:
    ticker: str
    timestamp: pd.Timestamp
    size: float  # positive for buy, negative for sell
    price: float
    slippage: float
    commission: float

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
        #print(signal, " ", price, " ", base, " ", dollar_amount, " ", self.fraction)
        return dollar_amount / price


@dataclass
class Portfolio:
    capital: float = 100000.0
    max_position_size: float = 0
    sizing_model: SizingModel = FixedFractionalSizing(fraction=.1)
    commission: float = 0.0
    slippage: float = 0.0
    initial_time: pd.Timestamp = pd.Timestamp('2000-01-01T12', tz='UTC')
    positions: dict[str, float] = None

    def __post_init__(self):
        # record initial parameters
        self.initial_cash: float = self.capital
        self.cash: float = self.capital
        self.positions = {} if self.positions is None else self.positions.copy()
        self.pos_value: float = 0.0
        self.order_history: List[Dict] = []
        self.timestamps: List[pd.Timestamp] = []
        self.equity_vals: List[float] = []
        self.cash_vals: List[float] = []
        self.holdings_vals: Dict[str, List[float]] = {}

    def execute_orders(self, signals: Dict[str, float], prices: Dict[str, float], timestamp: pd.Timestamp):
        for ticker, signal in sorted(signals.items(), key=lambda kv: -abs(kv[1])):
            price = prices.get(ticker)
            if price is None or price <= 0:
                continue
            size = self.sizing_model.size_position(signal, price, self)
            #print(ticker, " ", signal, " ", price, " ", size)
            if size == 0:
                continue
            current = self.positions.get(ticker, 0.0)
            # long-only enforcement
            if size < 0:
                size = max(-current, size)
                if size == 0:
                    continue
            value = price * size
            slippage_cost = abs(value) * self.slippage
            commission = self.commission
            if size > 0:
                total_cost = value + slippage_cost + commission
                if total_cost > self.cash:
                    continue
                self.cash -= total_cost
            else:
                proceeds = abs(value) - slippage_cost - commission
                self.cash += proceeds
            # update position and history
            self.positions[ticker] = current + size
            self.order_history.append({
                'ticker': ticker,
                'timestamp': timestamp.isoformat(),
                'size': size,
                'price': price,
                'slippage': slippage_cost,
                'commission': commission
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

    def to_config(self) -> dict:
        sm = self.sizing_model
        sm_cfg = {'type': sm.__class__.__name__}
        if isinstance(sm, FixedFractionalSizing):
            sm_cfg['fraction'] = sm.fraction
        return {
            'capital': self.initial_cash,
            'max_position_size': self.max_position_size,
            'commission': self.commission,
            'slippage': self.slippage,
            'initial_time': self.initial_time.isoformat() if self.initial_time else None,
            'sizing_model': sm_cfg,
            'positions': self.positions
        }

    def save_config(self, path: str):
        with open(path, 'w') as f:
            yaml.safe_dump(self.to_config(), f)



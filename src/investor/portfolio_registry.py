# portfolio_registry.py
from pathlib import Path
import pandas as pd
import yaml

from .portfolio import (
    PortfolioManager,
    FixedFractionalSizing,
    SizingModel,
    Position,
)


class PortfolioRegistry:
    def __init__(self, config_path: str = ""):
        if config_path:
            self.config_path = Path(config_path)
        else:
            base = Path(__file__).resolve().parents[0]
            self.config_path = base / "portfolios.yaml"
        self.configs = yaml.safe_load(self.config_path.read_text()) or {}

    def available(self) -> list[str]:
        return list(self.configs.keys())

    def get(self, name: str) -> PortfolioManager:
        if name not in self.configs:
            raise KeyError(f"Portfolio '{name}' not found.")
        cfg = self.configs[name]

        sm_cfg = cfg["sizing_model"]
        if sm_cfg["type"] == "FixedFractionalSizing":
            sizing: SizingModel = FixedFractionalSizing(sm_cfg["fraction"])
        else:
            raise ValueError(f"Unknown sizing model {sm_cfg['type']}")

        initial_time = pd.Timestamp(
            cfg.get("initial_time", "2000-01-01T12:00:00+00:00")
        )
        positions: dict[str, Position] = {
            tk: Position(ticker=tk, size=pos, last_trade_time=initial_time)
            for tk, pos in cfg.get("positions", {}).items()
        }

        manager = PortfolioManager(
            initial_capital=float(cfg["capital"]),
            initial_positions=positions,
            initial_time=initial_time,
            max_position_size=float(cfg["max_position_size"]),
            commission=cfg.get("commission", 0.0),
            slippage=cfg.get("slippage", 0.0),
            sizing_model=sizing,
            trades_per_period=cfg.get("trades_per_period", 1),
            period_length_days=cfg.get("period_length_days", 5),
            tau_max=cfg.get("tau_max", 1.0),
            tau_min=cfg.get("tau_min", 0.1),
            decay=cfg.get("decay", "linear"),
        )

        return manager

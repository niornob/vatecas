# portfolio_registry.py
from pathlib import Path
import yaml
from typing import Dict
import pandas as pd

from .deprecated_portfolio import Portfolio, FixedFractionalSizing, SizingModel


class PortfolioRegistry:
    def __init__(self, config_path: str = ""):
        if config_path:
            self.config_path = Path(config_path)
        else:
            base = Path(__file__).resolve().parents[2] / "config"
            self.config_path = base / "portfolios.yaml"
        self.configs = yaml.safe_load(self.config_path.read_text()) or {}

    def available(self) -> list[str]:
        return list(self.configs.keys())

    def get(self, name: str) -> Portfolio:
        if name not in self.configs:
            raise KeyError(f"Portfolio '{name}' not found.")
        cfg = self.configs[name]
        sm_cfg = cfg["sizing_model"]
        if sm_cfg["type"] == "FixedFractionalSizing":
            sizing: SizingModel = FixedFractionalSizing(sm_cfg["fraction"])
        else:
            raise ValueError(f"Unknown sizing model {sm_cfg['type']}")
        initial_time = (
            pd.Timestamp(cfg["initial_time"])
            if cfg.get("initial_time")
            else pd.Timestamp("2000-01-01 00:00", tz="UTC")
        )
        return Portfolio(
            capital=cfg["capital"],
            max_position_size=cfg["max_position_size"],
            sizing_model=sizing,
            commission=cfg.get("commission", 0.0),
            slippage=cfg.get("slippage", 0.0),
            initial_time=initial_time,
            positions=cfg.get("positions", {}),
        )

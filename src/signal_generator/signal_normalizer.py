"""
This module provides ways to normalize raw singnals.
raw signals are valued between -1 and +1 for each asset.
SignalNormalizer implements method to modify and stretch these signals across [-1,1].
overall market volatility and a target volatility can be used.
"""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from scipy.special import expit
from scipy.stats import rankdata


class SignalNormalizer(ABC):
    def __init__(self, name: str = "GenericNormalizer"):
        self.name = name

    @abstractmethod
    def normalize_signals(
        self,
        raw_signals: np.ndarray,
        market_volatility: Optional[float],
        target_volatility: Optional[float],
    ) -> np.ndarray: ...


class NormalizerRank(SignalNormalizer):
    def normalize_signals(
        self,
        raw_signals: np.ndarray,
        market_volatility: Optional[float] = None,
        target_volatility: Optional[float] = None,
    ) -> np.ndarray:
        rank = rankdata(raw_signals)
        rank = rank / len(rank)
        rank = 2 * rank - 1

        return rank


class NormalizerVolAdj(SignalNormalizer):
    def normalize_signals(
        self,
        raw_signals: np.ndarray,
        market_volatility: float,
        target_volatility: float,
    ) -> np.ndarray:

        n = max(np.abs(raw_signals))

        assert n > 0, "all raw_signals are zero. this should not happen."

        vol_scaling = np.clip(target_volatility / market_volatility, 0, 1)
        # print(vol_scaling)
        raw_raw_signals = (
            (1 - 1e-5) * (2 * expit(raw_signals / n) - 1) / (expit(1.0) - expit(-1.0))
        )

        return vol_scaling * raw_raw_signals

class NormalizerTrivial(SignalNormalizer):
    def normalize_signals(
        self,
        raw_signals: np.ndarray,
        market_volatility: Optional[float] = None,
        target_volatility: Optional[float] = None,
    ) -> np.ndarray:

        n = max(np.abs(raw_signals))
       
        return raw_signals / n

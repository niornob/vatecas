from typing import Type
from .base import SignalModule

from .KalmanNew import UKFSignalModule
#from .Kalman import UKFSignalModule


# Central registry: name -> SignalModule subclass
_SIGNAL_REGISTRY: dict[str, Type[SignalModule]] = {}

def register_signal(name: str, cls: Type[SignalModule]) -> None:
    if name in _SIGNAL_REGISTRY:
        raise ValueError(f"Signal name '{name}' is already registered.")
    if not issubclass(cls, SignalModule):
        raise TypeError(f"Class {cls.__name__} must inherit from SignalModule.")
    _SIGNAL_REGISTRY[name] = cls

# Register known signals at import time
register_signal("Kalman", UKFSignalModule)


class SignalRegistry:
    """
    Read-only interface to the global signal registry.
    """

    def __init__(self):
        self._registry = _SIGNAL_REGISTRY  # reference, not a copy

    def get(self, name: str) -> Type[SignalModule]:
        if name not in self._registry:
            raise KeyError(f"Signal '{name}' not found in registry.")
        return self._registry[name]

    def available_signals(self) -> list[str]:
        return list(self._registry.keys())

    def as_dict(self) -> dict[str, str]:
        return {name: cls.__name__ for name, cls in self._registry.items()}


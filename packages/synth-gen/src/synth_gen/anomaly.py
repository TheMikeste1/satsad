from abc import ABC, abstractmethod
from typing import Literal

import numpy as np


class Anomaly(ABC):
    layer: Literal["system", "observation"]

    @abstractmethod
    def apply(self, signal: np.float64, rng: np.random.Generator) -> np.float64: ...

    @abstractmethod
    def should_apply(self, t: float, rng: np.random.Generator) -> bool: ...

    def step(self, rng: np.random.Generator): ...

    @abstractmethod
    def _sub_id(self) -> tuple[str, ...]: ...

    def exclusive(self) -> bool:
        """If no other anomalies can be applied at this layer at this timestep. All other anomalies should be ignored this step."""
        return False

    def reset(self):
        pass

    def id(self) -> str:
        return ".".join((self.layer, *self._sub_id()))


class SystemAnomaly(Anomaly):
    layer = "system"


class ObservationAnomaly(Anomaly):
    layer = "observation"


class LevelShift(SystemAnomaly):
    def __init__(self, magnitude: float, p: float):
        self.magnitude = magnitude
        self.p = p
        self._active = False

    def _sub_id(self) -> tuple[str, ...]:
        return ("level_shift",)

    def should_apply(self, t: float, rng: np.random.Generator) -> bool:
        _ = t
        if not self._active and rng.random() < self.p:
            self._active = True
        return self._active

    def apply(self, signal: np.float64, rng: np.random.Generator) -> np.float64:
        _ = rng
        return signal + self.magnitude

    def reset(self):
        self._active = False


class LinearDrift(SystemAnomaly):
    def __init__(self, slope: float, p_start: float, p_stop: float):
        self.slope = slope
        self.p_start = p_start
        self.p_stop = p_stop
        self._active = False
        self._t = 0.0

    def _sub_id(self) -> tuple[str, ...]:
        return ("drift", "linear")

    def should_apply(self, t: float, rng: np.random.Generator) -> bool:
        _ = t
        if not self._active and rng.random() < self.p_start:
            self._active = True
            self._t = 0
        return self._active

    def apply(self, signal: np.float64, rng: np.random.Generator) -> np.float64:
        _ = rng
        return signal + self.slope * self._t

    def step(self, rng: np.random.Generator):
        if self._active:
            self._t += 1
            self._active = rng.random() < self.p_stop

    def reset(self):
        self._active = False


class Spike(ObservationAnomaly):
    def __init__(self, magnitude: float, p: float):
        self.magnitude = magnitude
        self.p = p

    def _sub_id(self) -> tuple[str, ...]:
        return ("spike",)

    def should_apply(self, t: float, rng: np.random.Generator) -> bool:
        _ = t
        return rng.random() < self.p

    def apply(self, signal: np.float64, rng: np.random.Generator) -> np.float64:
        return signal + rng.choice([-1, 1]) * self.magnitude


class Dropout(ObservationAnomaly):
    def __init__(self, p: float, fill_value: float = 0.0):
        self.p = p
        self.fill_value = fill_value

    def _sub_id(self) -> tuple[str, ...]:
        return ("dropout",)

    def should_apply(self, t: float, rng: np.random.Generator) -> bool:
        _ = t
        return rng.random() < self.p

    def apply(self, signal: np.float64, rng: np.random.Generator) -> np.float64:
        _ = signal
        _ = rng
        return np.float64(self.fill_value)

    def exclusive(self) -> bool:
        return True


class NoiseInflation(ObservationAnomaly):
    def __init__(self, sigma: float, p: float):
        self.sigma = sigma
        self.p = p

    def _sub_id(self) -> tuple[str, ...]:
        return ("noise_inflation",)

    def should_apply(self, t: float, rng: np.random.Generator) -> bool:
        _ = t
        return rng.random() < self.p

    def apply(self, signal: np.float64, rng: np.random.Generator) -> np.float64:
        return signal + rng.normal(0, self.sigma)

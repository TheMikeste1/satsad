from collections.abc import Sequence
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .anomaly import ObservationAnomaly, SystemAnomaly
from .regime import Regime, RegimeController


@dataclass(frozen=True)
class Mode:
    amplitude: float
    period: float
    phase: float

    def sample(self, t: np.float64 | float) -> np.float64:
        return self.amplitude * np.sin(2 * np.pi / self.period * t + self.phase, dtype=np.float64)


@dataclass(frozen=True)
class Sample:
    clean: np.float64
    observed: np.float64
    regime: int
    system_anomalies: tuple[str, ...]
    observation_anomalies: tuple[str, ...]


def generate_dataset(
    regimes: Sequence[Regime],
    system_anomalies: Sequence[SystemAnomaly],
    observation_anomalies: Sequence[ObservationAnomaly],
    t_start: int,
    count: int,
    *,
    rng: Optional[np.random.Generator] = None,
) -> list[Sample]:
    if rng is None:
        rng = np.random.default_rng()

    controller = RegimeController(regimes, system_anomalies, observation_anomalies)
    controller.start(rng)
    samples = [controller.step(t_start + i, rng) for i in range(count)]
    return samples

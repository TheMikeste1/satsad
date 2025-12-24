from collections.abc import Sequence
from dataclasses import dataclass
from typing import Optional

import numpy as np
from tqdm import tqdm

from synth_gen import regime

from .anomaly import ObservationAnomaly, SystemAnomaly
from .sample import Sample


@dataclass(frozen=True)
class Mode:
    amplitude: float
    period: float
    phase: float

    def sample(self, t: np.float64 | float) -> np.float64:
        return self.amplitude * np.sin(2 * np.pi / self.period * t + self.phase, dtype=np.float64)


def generate_dataset(
    regimes: Sequence[regime.Regime],
    global_system_anomalies: Sequence[SystemAnomaly],
    global_observation_anomalies: Sequence[ObservationAnomaly],
    t_start: int,
    count: int,
    *,
    rng: Optional[np.random.Generator] = None,
) -> list[Sample]:
    if rng is None:
        rng = np.random.default_rng()

    controller = regime.Controller(regimes, global_system_anomalies, global_observation_anomalies)
    controller.start(rng)
    samples = []
    for i in tqdm(range(count)):
        sample = controller.step(t_start + i, rng)
        samples.append(sample)

    return samples

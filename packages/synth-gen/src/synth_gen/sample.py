from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Sample:
    clean: np.float64
    observed: np.float64
    regime: int
    system_anomalies: tuple[str, ...]
    observation_anomalies: tuple[str, ...]

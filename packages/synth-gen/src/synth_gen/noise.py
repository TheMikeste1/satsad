from dataclasses import dataclass
from typing import Protocol

import numpy as np


class Noise(Protocol):
    def sample(self, rng: np.random.Generator) -> np.float64: ...


@dataclass(frozen=True)
class Gaussian(Noise):
    center: float
    std_dev: float

    def sample(self, rng: np.random.Generator) -> np.float64:
        return np.float64(rng.normal(self.center, self.std_dev))


@dataclass(frozen=True)
class Uniform(Noise):
    low: float
    high: float

    def sample(self, rng: np.random.Generator) -> np.float64:
        return np.float64(rng.uniform(self.low, self.high))


@dataclass(frozen=True)
class Impulsive(Noise):
    chance: float
    noise: Noise

    def __post_init__(self):
        if self.chance < 0 or 1 < self.chance:
            raise ValueError("chance must be in [0, 1]")

    def sample(self, rng: np.random.Generator) -> np.float64:
        rand = rng.random()
        if rand < self.chance:
            return np.float64(self.noise.sample(rng))
        return np.float64(0)

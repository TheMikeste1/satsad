from dataclasses import dataclass
from typing import Protocol

import numpy as np

from ._typing import VecF64


class Noise(Protocol):
    def __call__(self, rng: np.random.Generator, num_elements: int) -> VecF64: ...


@dataclass(frozen=True)
class Gaussian(Noise):
    center: float
    std_dev: float

    def __call__(self, rng: np.random.Generator, num_elements: int) -> VecF64:
        return rng.normal(self.center, self.std_dev, num_elements)


@dataclass(frozen=True)
class Uniform(Noise):
    low: float
    high: float

    def __call__(self, rng: np.random.Generator, num_elements: int) -> VecF64:
        return rng.uniform(self.low, self.high, num_elements)


@dataclass(frozen=True)
class Impulsive(Noise):
    chance: float
    noise: Noise

    def __post_init__(self):
        if self.chance < 0 or 1 < self.chance:
            raise ValueError("chance must be in [0, 1]")

    def __call__(self, rng: np.random.Generator, num_elements: int) -> VecF64:
        rand = rng.random(num_elements)
        np.less(rand, self.chance, out=rand)
        noise = self.noise(rng, num_elements)
        return np.multiply(rand, noise, out=rand)

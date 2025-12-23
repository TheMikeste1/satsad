from dataclasses import dataclass
from typing import Optional, Protocol, overload

import numpy as np

from ._typing import VecF64


class Noise(Protocol):
    @overload
    def __call__(self, rng: np.random.Generator) -> np.float64: ...
    @overload
    def __call__(self, rng: np.random.Generator, num_elements: int) -> VecF64: ...


@dataclass(frozen=True)
class Gaussian(Noise):
    center: float
    std_dev: float

    @overload
    def __call__(self, rng: np.random.Generator) -> np.float64: ...
    @overload
    def __call__(self, rng: np.random.Generator, num_elements: int) -> VecF64: ...
    def __call__(self, rng: np.random.Generator, num_elements: Optional[int] = None) -> np.float64 | VecF64:
        return rng.normal(self.center, self.std_dev, num_elements)


@dataclass(frozen=True)
class Uniform(Noise):
    low: float
    high: float

    @overload
    def __call__(self, rng: np.random.Generator) -> np.float64: ...
    @overload
    def __call__(self, rng: np.random.Generator, num_elements: int) -> VecF64: ...
    def __call__(self, rng: np.random.Generator, num_elements: Optional[int] = None) -> np.float64 | VecF64:
        return rng.uniform(self.low, self.high, num_elements)


@dataclass(frozen=True)
class Impulsive(Noise):
    chance: float
    noise: Noise

    def __post_init__(self):
        if self.chance < 0 or 1 < self.chance:
            raise ValueError("chance must be in [0, 1]")

    @overload
    def __call__(self, rng: np.random.Generator) -> np.float64: ...
    @overload
    def __call__(self, rng: np.random.Generator, num_elements: int) -> VecF64: ...
    def __call__(self, rng: np.random.Generator, num_elements: Optional[int] = None) -> np.float64 | VecF64:
        rand = rng.random(num_elements)
        np.less(rand, self.chance, out=rand)
        if num_elements is None:
            noise = self.noise(rng)
        else:
            noise = self.noise(rng, num_elements)
        return np.multiply(rand, noise, out=rand)

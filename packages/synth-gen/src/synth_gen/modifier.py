from dataclasses import dataclass
from typing import Protocol

import numpy as np

from ._typing import VecF64


class Modifier(Protocol):
    def __call__(self, t: VecF64, signal: VecF64) -> VecF64: ...


@dataclass(frozen=True)
class Amplify(Modifier):
    by: float

    def __call__(self, t: VecF64, signal: VecF64) -> VecF64:
        _ = t
        return np.multiply(signal, self.by, out=signal)


@dataclass(frozen=True)
class ChangingAmplify(Modifier):
    rate_of_change: float
    by: float = 1

    def __call__(self, t: VecF64, signal: VecF64) -> VecF64:
        rates = t * self.rate_of_change
        np.add(rates, self.by, out=rates)
        return np.multiply(signal, rates, out=signal)


@dataclass(frozen=True)
class Clip(Modifier):
    min: float
    max: float

    def __post_init__(self):
        if self.min > self.max:
            raise ValueError("min cannot be greater than max")

    def __call__(self, t: VecF64, signal: VecF64) -> VecF64:
        _ = t
        return np.clip(signal, self.min, self.max, out=signal)


type Saturate = Clip

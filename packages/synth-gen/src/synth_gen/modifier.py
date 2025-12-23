from dataclasses import dataclass
from typing import Protocol, overload

import numpy as np

from ._typing import VecF64


class Modifier(Protocol):
    @overload
    def __call__(self, t: VecF64, signal: VecF64) -> VecF64: ...
    @overload
    def __call__(self, t: float, signal: float) -> np.float64: ...


@dataclass(frozen=True)
class Amplify(Modifier):
    by: float

    @overload
    def __call__(self, t: VecF64, signal: VecF64) -> VecF64: ...
    @overload
    def __call__(self, t: float, signal: float) -> np.float64: ...

    def __call__(self, t: VecF64 | int | float, signal: VecF64 | float) -> VecF64 | np.float64:
        _ = t
        out = signal if isinstance(signal, np.ndarray) else None
        a = np.multiply(signal, self.by, out=out, dtype=np.float64)
        return a


@dataclass(frozen=True)
class ChangingAmplify(Modifier):
    rate_of_change: float
    by: float = 1

    @overload
    def __call__(self, t: VecF64, signal: VecF64) -> VecF64: ...
    @overload
    def __call__(self, t: float, signal: float) -> np.float64: ...

    def __call__(self, t: VecF64 | int | float, signal: VecF64 | float) -> VecF64 | np.float64:
        rates = t * self.rate_of_change
        out = signal if isinstance(signal, np.ndarray) else None
        np.add(rates, self.by, out=out)
        return np.multiply(signal, rates, out=out)


@dataclass(frozen=True)
class Clip(Modifier):
    min: float
    max: float

    def __post_init__(self):
        if self.min > self.max:
            raise ValueError("min cannot be greater than max")

    @overload
    def __call__(self, t: VecF64, signal: VecF64) -> VecF64: ...
    @overload
    def __call__(self, t: float, signal: float) -> np.float64: ...

    def __call__(self, t: VecF64 | int | float, signal: VecF64 | float) -> VecF64 | np.float64:
        _ = t
        out = signal if isinstance(signal, np.ndarray) else None
        return np.clip(signal, self.min, self.max, out=out)


type Saturate = Clip

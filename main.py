from collections.abc import Iterable
from dataclasses import dataclass
from functools import reduce
from typing import Protocol

import numpy as np
import plotly.express as px

type VecF64 = np.ndarray[tuple[int], np.dtype[np.float64]]


@dataclass(frozen=True)
class Mode:
    amplitude: float
    period: float
    phase: float


class SignalModifier(Protocol):
    def __call__(self, t: VecF64, signal: VecF64) -> VecF64: ...


@dataclass(frozen=True)
class Amplify(SignalModifier):
    by: float

    def __call__(self, t: VecF64, signal: VecF64) -> VecF64:
        _ = t
        return np.multiply(signal, self.by, out=signal)


@dataclass(frozen=True)
class ChangingAmplify(SignalModifier):
    rate_of_change: float
    by: float = 1

    def __call__(self, t: VecF64, signal: VecF64) -> VecF64:
        rates = t * self.rate_of_change
        np.add(rates, self.by, out=rates)
        return np.multiply(signal, rates, out=signal)


@dataclass(frozen=True)
class Clip(SignalModifier):
    min: float
    max: float

    def __post_init__(self):
        if self.min > self.max:
            raise ValueError("min cannot be greater than max")

    def __call__(self, t: VecF64, signal: VecF64) -> VecF64:
        _ = t
        return np.clip(signal, self.min, self.max, out=signal)


type Saturate = Clip


def apply_mode(t: VecF64, mode: Mode) -> VecF64:
    return mode.amplitude * np.sin(2 * np.pi / mode.period * (t + mode.phase))


def compose_signal(
    t: VecF64,
    modes: Mode | Iterable[Mode],
):
    if isinstance(modes, Mode):
        modes = [modes]

    values = np.sum(tuple(apply_mode(t, m) for m in modes), axis=0)
    assert isinstance(values, np.ndarray), "Modes can't be empty"
    return values


def main():
    mode = Mode(2, 1, 0.5)
    modulators: list[SignalModifier] = [
        Clip(-1, 1),
    ]

    t = np.arange(100, dtype=np.float64)
    # t = np.linspace(0, 1, 10_000 * 10, dtype=np.float64)
    t.setflags(write=False)
    signal = compose_signal(t, mode)
    signal = reduce(lambda sig, mod: mod(t, sig), modulators, signal)
    print(signal)

    fig = px.line(x=t, y=signal)
    fig.show()


if __name__ == "__main__":
    main()

from collections.abc import Iterable
from dataclasses import dataclass
from functools import reduce
from typing import Optional, Protocol

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


class Noise(Protocol):
    def __call__(self, rng: np.random.Generator, num_elements: int) -> VecF64: ...


@dataclass(frozen=True)
class GaussianNoise(Noise):
    center: float
    std_dev: float

    def __call__(self, rng: np.random.Generator, num_elements: int) -> VecF64:
        return rng.normal(self.center, self.std_dev, num_elements)


@dataclass(frozen=True)
class UniformNoise(Noise):
    low: float
    high: float

    def __call__(self, rng: np.random.Generator, num_elements: int) -> VecF64:
        return rng.uniform(self.low, self.high, num_elements)


@dataclass(frozen=True)
class ImpulsiveNoise(Noise):
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


def apply_mode(t: VecF64, mode: Mode) -> VecF64:
    return mode.amplitude * np.sin(2 * np.pi / mode.period * t + mode.phase)


def compose_signal(
    t: VecF64,
    modes: Mode | Iterable[Mode],
):
    if isinstance(modes, Mode):
        modes = [modes]

    values = np.sum(tuple(apply_mode(t, m) for m in modes), axis=0)
    assert isinstance(values, np.ndarray), "Modes can't be empty"
    return values


def generate_signal(t: VecF64, modes: Mode | Iterable[Mode], modifiers: list[SignalModifier], noise: list[Noise], rng: Optional[np.random.Generator] = None):
    signal = compose_signal(t, modes)
    signal = reduce(lambda sig, mod: mod(t, sig), modifiers, signal)

    # Current regime's "clean" signal is complete; add noise
    if rng is None:
        rng = np.random.default_rng()
    signals = [signal] + [n(rng, signal.shape[0]) for n in noise]
    noised_signal = np.sum(signals, axis=0)
    return signal, noised_signal


def main():
    rng = np.random.default_rng()

    t = np.arange(1_000, dtype=np.float64)
    t.setflags(write=False)
    mode = Mode(2, 100, 1)
    modifiers: list[SignalModifier] = [
        Clip(-1, 1),
    ]
    noise: list[Noise] = [ImpulsiveNoise(0.05, UniformNoise(-1, 1))]
    signal = generate_signal(t, mode, modifiers, noise, rng=rng)

    fig = px.line(x=t, y=list(signal))
    fig.show()


if __name__ == "__main__":
    main()

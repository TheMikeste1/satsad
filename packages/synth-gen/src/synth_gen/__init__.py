from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import reduce
from typing import Optional, overload

import numpy as np

from ._typing import VecF64, VecU16
from .modifier import Modifier
from .noise import Noise
from .regime import Regime, RegimeController


@dataclass(frozen=True)
class Mode:
    amplitude: float
    period: float
    phase: float

    @overload
    def generate(self, t: VecF64) -> VecF64: ...
    @overload
    def generate(self, t: np.float64 | float) -> np.float64: ...

    def generate(self, t: VecF64 | np.float64 | float) -> VecF64 | np.float64:
        return self.amplitude * np.sin(2 * np.pi / self.period * t + self.phase, dtype=np.float64)


@dataclass(frozen=True)
class Sample:
    clean: np.float64
    observed: np.float64
    regime: int
    anomaly: bool


def generate_signal(
    t: VecF64, modes: Mode | Iterable[Mode], modifiers: list[Modifier], noise: list[Noise], *, rng: Optional[np.random.Generator] = None
) -> tuple[VecF64, VecF64]:
    if isinstance(modes, Mode):
        modes = [modes]

    signal = np.sum(tuple(m.generate(t) for m in modes), axis=0)
    assert isinstance(signal, np.ndarray), "Modes can't be empty"
    signal = reduce(lambda sig, mod: mod(t, sig), modifiers, signal)

    # "Clean" signal is complete; add noise
    if rng is None:
        rng = np.random.default_rng()
    signals = [signal] + [n(rng, signal.shape[0]) for n in noise]
    noised_signal = np.sum(signals, axis=0)
    assert len(signal) == len(noised_signal)
    return signal, noised_signal


def generate_regimes(regimes: Sequence[Regime], t_start: int, count: int, *, rng: Optional[np.random.Generator] = None) -> list[Sample]:
    if rng is None:
        rng = np.random.default_rng()

    controller = RegimeController(regimes)
    controller.start(rng)
    samples = []
    for i in range(count):
        (c, o), rm = controller.step(t_start + i, rng)
        samples.append(Sample(c, o, rm, False))
    return samples

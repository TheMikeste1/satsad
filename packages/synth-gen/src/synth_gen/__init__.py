from collections.abc import Iterable
from dataclasses import dataclass
from functools import reduce
from typing import Optional

import numpy as np

from ._typing import VecF64, VecU16
from .modifier import Modifier
from .noise import Noise

__all__ = ["Mode", "Regime", "generate_signal", "generate_regime", "generate_regimes"]


@dataclass(frozen=True)
class Mode:
    amplitude: float
    period: float
    phase: float

    def generate(self, t: VecF64) -> VecF64:
        return self.amplitude * np.sin(2 * np.pi / self.period * t + self.phase)


@dataclass(frozen=True)
class Regime:
    modes: Mode | list[Mode]
    modifiers: list[Modifier]
    noise: list[Noise]
    length_distribution: int  # TODO: Replace with distribution object

    def __post_init__(self):
        if isinstance(self.length_distribution, int) and self.length_distribution <= 0:
            raise ValueError("length must be > 0")

    def length(self, rng: np.random.Generator) -> int:
        _ = rng
        return self.length_distribution


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


def generate_regime(regime: Regime, t_start: int, *, rng: Optional[np.random.Generator] = None) -> tuple[VecF64, VecF64]:
    if rng is None:
        rng = np.random.default_rng()
    length = regime.length(rng)
    t = np.arange(t_start, t_start + length, dtype=np.float64)
    return generate_signal(t, regime.modes, regime.modifiers, regime.noise)


def generate_regimes(regimes: list[Regime], t_start: int, *, rng: Optional[np.random.Generator] = None) -> tuple[VecF64, VecF64, VecU16]:
    if rng is None:
        rng = np.random.default_rng()

    segments = []
    noised_segments = []
    regime_masks = []
    for i, regime in enumerate(regimes):
        segment, noised_segment = generate_regime(regime, t_start, rng=rng)
        segments.append(segment)
        noised_segments.append(noised_segment)
        regime_masks.append(np.repeat(np.uint16(i), len(segments)))
        t_start += len(segment)

    signal = np.reshape(segments, -1)
    noised_signal = np.reshape(noised_segments, -1)
    regime_mask = np.reshape(regime_masks, -1)
    return signal, noised_signal, regime_mask

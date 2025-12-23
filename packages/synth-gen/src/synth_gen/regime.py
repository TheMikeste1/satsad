from collections.abc import Sequence
from typing import Protocol

import numpy as np

from synth_gen import Mode
from synth_gen.modifier import Modifier
from synth_gen.noise import Noise


class DurationDeteminator(Protocol):
    def __call__(self, rng: np.random.Generator) -> int: ...


class Regime:
    def __init__(
        self, modes: list[Mode], modifiers: list[Modifier], noise: list[Noise], transition_probablity: float, duration_deteminator: DurationDeteminator
    ):
        if len(modes) == 0:
            raise ValueError("A regime must have at least one mode")

        self._modes = modes
        self._modifiers = modifiers
        self._noise = noise
        self.transition_probability = transition_probablity

        self._duration_deteminator = duration_deteminator
        self._min_time_remaining = 0

    @property
    def modes(self):
        return self._modes

    @property
    def modifiers(self):
        return self._modifiers

    @property
    def noise(self):
        return self._noise

    def enter(self, rng: np.random.Generator):
        self._min_time_remaining = self._duration_deteminator(rng)

    def step(self, t: float, rng: np.random.Generator) -> tuple[np.float64, np.float64]:
        signal = np.float64(sum((m.generate(t) for m in self._modes)))
        for mod in self._modifiers:
            signal = mod(t, signal)

        noised_signal = signal + sum(n(rng) for n in self._noise)
        self._min_time_remaining -= 1
        return signal, noised_signal

    def should_transition(self, rng: np.random.Generator):
        return self._min_time_remaining <= 0 and rng.random() < self.transition_probability


class RegimeController:
    def __init__(self, regimes: Sequence[Regime]):
        if len(regimes) == 0:
            raise ValueError("There must be at least one regime")

        self._regimes = regimes
        self._current = self._regimes[0]
        self._current_id = 0

    def start(self, rng: np.random.Generator):
        self._select_new_regime(rng)

    def step(self, t: float, rng: np.random.Generator) -> tuple[tuple[np.float64, np.float64], int]:
        if self._current.should_transition(rng):
            self._select_new_regime(rng)

        return self._current.step(t, rng), self._current_id

    def _select_new_regime(self, rng: np.random.Generator):
        if len(self._regimes) == 1:
            self._current_id = 0
        else:
            self._current_id = rng.choice([i for i in range(len(self._regimes)) if i != self._current_id])
        self._current = self._regimes[self._current_id]
        self._current.enter(rng)

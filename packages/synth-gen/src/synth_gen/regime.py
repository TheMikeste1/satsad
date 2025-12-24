from collections.abc import Iterator, Sequence
from itertools import chain
from typing import Protocol

import numpy as np

from synth_gen import Mode, Sample
from synth_gen.anomaly import Anomaly, ObservationAnomaly, SystemAnomaly
from synth_gen.modifier import Modifier
from synth_gen.noise import Noise


class DurationDeteminant(Protocol):
    def __call__(self, rng: np.random.Generator) -> int: ...


class Regime:
    def __init__(
        self,
        modes: list[Mode],
        modifiers: list[Modifier],
        system_anomalies: Sequence[SystemAnomaly],
        observation_anomalies: Sequence[ObservationAnomaly],
        noise: list[Noise],
        transition_probability: float,
        duration_determinant: DurationDeteminant,
    ):
        if len(modes) == 0:
            raise ValueError("A regime must have at least one mode")

        self._modes = modes
        self._modifiers = modifiers
        self._noise = noise
        self._system_anomalies = system_anomalies
        self._observation_anomalies = observation_anomalies
        self.transition_probability = transition_probability

        self._duration_determinant = duration_determinant
        self._min_time_remaining = 0

    @property
    def system_anomalies(self):
        return self._system_anomalies

    @property
    def observation_anomalies(self):
        return self._observation_anomalies

    def enter(self, rng: np.random.Generator):
        for anomaly in chain(self._system_anomalies, self._observation_anomalies):
            anomaly.reset()

        self._min_time_remaining = self._duration_determinant(rng)

    def step(self, t: float) -> np.float64:
        self._min_time_remaining -= 1

        signal = np.float64(sum((m.sample(t) for m in self._modes)))
        for mod in self._modifiers:
            signal = mod(t, signal)
        return signal

    def apply_noise(self, signal: np.float64, rng: np.random.Generator) -> np.float64:
        return signal + sum(n.sample(rng) for n in self._noise)

    def should_transition(self, rng: np.random.Generator):
        return self._min_time_remaining <= 0 and rng.random() < self.transition_probability


class RegimeController:
    def __init__(self, regimes: Sequence[Regime], system_anomalies: Sequence[SystemAnomaly], observation_anomalies: Sequence[ObservationAnomaly]):
        if len(regimes) == 0:
            raise ValueError("There must be at least one regime")

        self._regimes = regimes
        self._system_anomalies = system_anomalies
        self._observation_anomalies = observation_anomalies
        self._current = self._regimes[0]
        self._current_id = 0

    def start(self, rng: np.random.Generator):
        self._select_new_regime(rng)

    def step(self, t: float, rng: np.random.Generator) -> Sample:
        if self._current.should_transition(rng):
            self._select_new_regime(rng)

        signal = self._current.step(t)
        signal, system_anomalies = _apply_anomalies(t, signal, chain(self._current.system_anomalies, self._system_anomalies), rng)

        observed_signal = self._current.apply_noise(signal, rng)
        observed_signal, observation_anomalies = _apply_anomalies(
            t, observed_signal, chain(self._current.observation_anomalies, self._observation_anomalies), rng
        )

        return Sample(signal, observed_signal, self._current_id, tuple(system_anomalies), tuple(observation_anomalies))

    def _select_new_regime(self, rng: np.random.Generator):
        if len(self._regimes) == 1:
            self._current_id = 0
        else:
            self._current_id = rng.choice([i for i in range(len(self._regimes)) if i != self._current_id])
        self._current = self._regimes[self._current_id]
        self._current.enter(rng)


def _apply_anomalies(t: float, signal: np.float64, anomalies: Iterator[Anomaly], rng: np.random.Generator):
    system_anomalies = []
    activated_exclusive = False
    for anomaly in anomalies:
        if not activated_exclusive and anomaly.should_apply(t, rng):
            signal = anomaly.apply(signal, rng)
            if anomaly.exclusive():
                activated_exclusive = True
                system_anomalies = [anomaly.id()]
            else:
                system_anomalies.append(anomaly.id())
        anomaly.step(rng)
    return signal, system_anomalies

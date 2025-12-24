import os
from pathlib import Path

import numpy as np
import pandas as pd

from synth_gen import Mode, generate_dataset
from synth_gen.anomaly import Dropout, LevelShift, LinearDrift, NoiseInflation, Spike
from synth_gen.modifier import Amplify, ChangingAmplify
from synth_gen.noise import Gaussian
from synth_gen.regime import Regime
from synth_gen.sample import Sample

# ---------------------------
# Duration helpers
# ---------------------------


def fixed_duration(n: int):
    return lambda rng: n


def uniform_duration(low: int, high: int):
    return lambda rng: rng.integers(low, high)


# ---------------------------
# Base regimes
# ---------------------------

REGIMES = [
    # Regime 0: low-frequency stable
    Regime(
        modes=[Mode(1.0, period=100, phase=0)],
        modifiers=[],
        system_anomalies=[],
        observation_anomalies=[],
        noise=[Gaussian(0.0, 0.05)],
        transition_probability=0.05,
        duration_determinant=uniform_duration(200, 400),
    ),
    # Regime 1: higher frequency, amplified
    Regime(
        modes=[Mode(1.0, period=40, phase=0)],
        modifiers=[Amplify(1.5)],
        system_anomalies=[],
        observation_anomalies=[],
        noise=[Gaussian(0.0, 0.08)],
        transition_probability=0.05,
        duration_determinant=uniform_duration(150, 300),
    ),
    # Regime 2: slowly changing gain
    Regime(
        modes=[Mode(0.8, period=60, phase=0)],
        modifiers=[ChangingAmplify(rate_of_change=1e-4, by=1.0)],
        system_anomalies=[],
        observation_anomalies=[],
        noise=[Gaussian(0.0, 0.06)],
        transition_probability=0.05,
        duration_determinant=uniform_duration(250, 500),
    ),
]


# ---------------------------
# Global anomalies
# ---------------------------

GLOBAL_SYSTEM_ANOMALIES = [
    LevelShift(magnitude=1.0, p=1e-4),
    LinearDrift(slope=0.002, p_start=5e-5, p_stop=0.01),
]

GLOBAL_OBSERVATION_ANOMALIES = [
    Spike(magnitude=3.0, p=5e-4),
    Dropout(p=2e-4, fill_value=0.0),
    NoiseInflation(sigma=1.0, p=3e-4),
]


# ---------------------------
# Dataset suite
# ---------------------------


def generate_all_datasets(seed: int = 0):
    rng = np.random.default_rng(seed)

    datasets = {}

    # 1. Baseline (no anomalies)
    datasets["baseline"] = generate_dataset(
        regimes=REGIMES,
        global_system_anomalies=[],
        global_observation_anomalies=[],
        t_start=0,
        count=50_000,
        rng=rng,
    )

    # 2. Regime segmentation only
    datasets["regime_only"] = generate_dataset(
        regimes=REGIMES,
        global_system_anomalies=[],
        global_observation_anomalies=[],
        t_start=0,
        count=100_000,
        rng=rng,
    )

    # 3. System anomalies only
    datasets["system_anomalies"] = generate_dataset(
        regimes=REGIMES,
        global_system_anomalies=GLOBAL_SYSTEM_ANOMALIES,
        global_observation_anomalies=[],
        t_start=0,
        count=100_000,
        rng=rng,
    )

    # 4. Observation anomalies only
    datasets["observation_anomalies"] = generate_dataset(
        regimes=REGIMES,
        global_system_anomalies=[],
        global_observation_anomalies=GLOBAL_OBSERVATION_ANOMALIES,
        t_start=0,
        count=100_000,
        rng=rng,
    )

    # 5. Mixed (easy)
    datasets["mixed_easy"] = generate_dataset(
        regimes=REGIMES,
        global_system_anomalies=[
            LevelShift(magnitude=0.7, p=2e-4),
        ],
        global_observation_anomalies=[
            Spike(magnitude=2.0, p=3e-4),
        ],
        t_start=0,
        count=150_000,
        rng=rng,
    )

    # 6. Mixed (hard / realistic)
    datasets["mixed_hard"] = generate_dataset(
        regimes=REGIMES,
        global_system_anomalies=GLOBAL_SYSTEM_ANOMALIES,
        global_observation_anomalies=GLOBAL_OBSERVATION_ANOMALIES,
        t_start=0,
        count=200_000,
        rng=rng,
    )

    return datasets


def save_dataset(name: str, data: list[Sample]):
    df = pd.DataFrame(data)
    output_dir = Path(__file__).resolve().parent.parent.parent.parent / "data"
    os.makedirs(output_dir, exist_ok=True)
    out_file = output_dir / f"{name}.csv"
    df.to_csv(out_file)
    print(f"\t{out_file}")


def main():
    print("Generating. . .")
    datasets = generate_all_datasets()

    print("Saving. . .")
    for name, data in datasets.items():
        save_dataset(name, data)

    print("Done!")


if __name__ == "__main__":
    main()

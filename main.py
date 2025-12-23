import numpy as np
import plotly.express as px
import synth_gen.modifier as sgm
import synth_gen.noise as sgn
from synth_gen import Mode, generate_signal
from synth_gen.modifier import Modifier
from synth_gen.noise import Noise


def main():
    rng = np.random.default_rng()

    t = np.arange(1_000, dtype=np.float64)
    t.setflags(write=False)
    mode = Mode(2, 100, 1)
    modifiers: list[Modifier] = [
        sgm.Clip(-1, 1),
    ]
    noise: list[Noise] = [sgn.Impulsive(0.05, sgn.Uniform(-1, 1))]
    signal = generate_signal(t, mode, modifiers, noise, rng=rng)

    fig = px.line(x=t, y=list(signal))
    fig.show()


if __name__ == "__main__":
    main()

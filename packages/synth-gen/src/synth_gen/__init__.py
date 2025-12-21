from collections.abc import Callable, Iterable
from functools import reduce

import numpy as np

type Vector = np.ndarray[tuple[int], np.dtype[np.float64]]
type VectorModifier = Callable[[Vector], Vector]


def generate_signal(num_datapoints: int, components: VectorModifier | Iterable[VectorModifier]) -> Vector:
    if not isinstance(components, Iterable):
        components = (components,)

    seed = np.arange(0, num_datapoints, dtype=np.float64)
    return reduce(lambda data, func: func(data), components, seed)

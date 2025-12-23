import numpy as np

type Vec[T: np.generic] = np.ndarray[tuple[int], np.dtype[T]]

type VecF64 = Vec[np.float64]
type VecU16 = Vec[np.uint16]

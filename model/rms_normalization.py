import numpy as np
from typing import List


class Solution:

  def rms_norm(self, x: List[float], gamma: List[float], eps: float) -> List[float]:
    x_arr = np.array(x, dtype=np.float64)
    gamma_arr = np.array(gamma, dtype=np.float64)

    # Compute root mean square (RMS)
    rms = np.sqrt(np.mean(x_arr**2) + eps)

    # Normalize by RMS and scale by gamma
    out = (x_arr / rms) * gamma_arr

    # Round to 4 decimal places and return as a list
    return np.round(out, decimals=4).tolist()
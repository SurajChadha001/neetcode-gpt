import numpy as np
from numpy.typing import NDArray


class Solution:

  def forward(
      self,
      x: NDArray[np.float64],
      gamma: NDArray[np.float64],
      beta: NDArray[np.float64],
  ) -> NDArray[np.float64]:
    eps = 1e-5

    # Compute mean and variance of the feature vector x
    mean = np.mean(x)
    var = np.var(x)  # Uses biased variance (ddof=0) matching PyTorch default

    # Normalize x
    x_hat = (x - mean) / np.sqrt(var + eps)

    # Scale and shift using gamma and beta
    out = gamma * x_hat + beta

    return np.round(out, 5)
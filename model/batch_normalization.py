import numpy as np
from typing import List, Tuple


class Solution:

  def batch_norm(
      self,
      x: List[List[float]],
      gamma: List[float],
      beta: List[float],
      running_mean: List[float],
      running_var: List[float],
      momentum: float,
      eps: float,
      training: bool,
  ) -> Tuple[List[List[float]], List[float], List[float]]:
    x_arr = np.array(x, dtype=np.float64)
    gamma_arr = np.array(gamma, dtype=np.float64)
    beta_arr = np.array(beta, dtype=np.float64)
    running_mean_arr = np.array(running_mean, dtype=np.float64)
    running_var_arr = np.array(running_var, dtype=np.float64)

    if training:
      # Compute batch statistics along the batch dimension (axis 0)
      batch_mean = np.mean(x_arr, axis=0)
      batch_var = np.var(x_arr, axis=0, ddof=0)  # Biased variance

      # Normalize using batch statistics
      x_hat = (x_arr - batch_mean) / np.sqrt(batch_var + eps)

      # Update running statistics using biased variance
      running_mean_arr = (
          1 - momentum
      ) * running_mean_arr + momentum * batch_mean
      running_var_arr = (1 - momentum) * running_var_arr + momentum * batch_var
    else:
      # Normalize using running statistics during inference
      x_hat = (x_arr - running_mean_arr) / np.sqrt(running_var_arr + eps)

    # Apply affine transform: scale and shift
    y = gamma_arr * x_hat + beta_arr

    # Round all results to 4 decimal places and convert to lists
    y_rounded = np.round(y, decimals=4).tolist()
    running_mean_rounded = np.round(running_mean_arr, decimals=4).tolist()
    running_var_rounded = np.round(running_var_arr, decimals=4).tolist()

    return y_rounded, running_mean_rounded, running_var_rounded
import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:

  def train(
      self, X: NDArray[np.float64], y: NDArray[np.float64], epochs: int, lr: float
  ) -> Tuple[NDArray[np.float64], float]:
    n_samples, n_features = X.shape
    
    # Initialize weights to zeros and bias to 0.0
    w = np.zeros(n_features, dtype=np.float64)
    b = 0.0

    for _ in range(epochs):
      # Forward pass: y_hat = X @ w + b
      y_hat = X @ w + b

      # Compute gradients for MSE loss: L = (1/n) * sum((y_hat - y)^2)
      error = y_hat - y
      dw = (2.0 / n_samples) * (X.T @ error)
      db = (2.0 / n_samples) * np.sum(error)

      # Gradient descent update
      w -= lr * dw
      b -= lr * db

    return (np.round(w, 5), round(b, 5))
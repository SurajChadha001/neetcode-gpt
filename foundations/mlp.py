
import numpy as np
from numpy.typing import NDArray
from typing import List


class Solution:
    def forward(
        self,
        x: NDArray[np.float64],
        weights: List[NDArray[np.float64]],
        biases: List[NDArray[np.float64]]
    ) -> NDArray[np.float64]:

        for i, (w, b) in enumerate(zip(weights, biases)):
            x = np.dot(x, w) + b

            if i < len(weights) - 1:
                x = np.maximum(0, x)

        return x

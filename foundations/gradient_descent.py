class Solution:

    def get_minimizer(self, iterations: int, learning_rate: float, init: int) -> float:

        # Objective function: f(x) = x^2
        # Derivative: f'(x) = 2x
        # Update rule: x = x - learning_rate * f'(x)

        x = init

        for _ in range(iterations):
            gradient = 2 * x
            x = x - learning_rate * gradient

        # Round final answer to 5 decimal places
        return round(x, 5)
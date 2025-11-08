import numpy as np


def main():
    """Run code."""
    X = np.array(
        [
            [-1, 2, 3, 13, 17],
            [4, -5, 6, 14, 18],
            [7, 8, -9, 15, 19],
            [-10, 11, 12, 16, -20],
        ],
        dtype=np.float64,
    )
    y = np.array([100, 200, 300, 400], dtype=np.float64)

    my_neuron = Neuron((X.shape))
    print("my_neuron.input_size:", my_neuron.input_size)
    print("my_neuron.w:", my_neuron.w)
    print("linear transformation:", my_neuron._get_linear_transformation(X))
    print("my_neuron.forward(X):", my_neuron.forward(X))
    print(
        "my_neuron.gradient_of_J(y_true=y, x=X):",
        my_neuron.gradient_of_J(y_true=y, x=X),
    )


class Neuron:
    """Implement single neuron class."""

    def __init__(self, input_size: tuple) -> None:
        """Initialize neuron entity."""
        self.input_size = input_size
        self.w = np.random.default_rng().random(self.input_size[-1])
        self.b = np.random.default_rng().random()
        # activation function hyperparameter
        self.alpha = 0.01

    def leaky_relu(self, vector: np.array) -> np.array:
        """
        Break linearity.

        Parameters
        ----------
        vector : np.array
            y_pred before activation function applied.

        """
        return np.maximum(self.alpha * vector, vector)

    def derivative_of_leaky_relu(self, vector: np.array) -> np.array:
        """
        Calculate the derivative of the activation function.

        Parameters
        ----------
        vector : np.array
            y_pred before activation function applied.

        """
        derivative = np.asarray(vector, copy=True)
        return np.where(derivative < 0, self.alpha, 1)

    def _get_linear_transformation(self, x: np.array) -> np.array:
        """
        Make the matrix multiplication of x and weights.

        The result is y_pred before activation function.
        """
        return np.dot(x, self.w) + self.b

    def forward(self, x: np.array) -> np.array:
        """Calculate forward pass with activation function."""
        vector = self._get_linear_transformation(x)
        return self.leaky_relu(vector)

    def gradient_of_J(self, y_true: np.array, x: np.array) -> np.array:
        """Compute the gradient after forward pass."""
        y_pred = self.forward(x)
        error = y_pred - y_true

        leaky_relu_derivative = self.derivative_of_leaky_relu(
            self._get_linear_transformation(x)
        )

        return np.dot((error * leaky_relu_derivative), x)


if __name__ == "__main__":
    main()

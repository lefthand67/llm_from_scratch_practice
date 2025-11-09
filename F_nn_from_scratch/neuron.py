import numpy as np


class Neuron:
    """Implement single neuron class."""

    def __init__(self, alpha: float = 0.01):
        """Initialize neuron entity."""
        self.w = None
        self.b = 0.0
        # linear transformation
        self.z: np.array = None
        # activation transformation
        self.a: np.array = None
        # activation function hyperparameter
        self.alpha = alpha

    def leaky_relu(self) -> np.ndarray:
        """
        Break linearity.

        Parameters
        ----------
        alpha : float64
            y_pred before activation function applied.

        """
        return np.maximum(self.alpha * self.z, self.z)

    def derivative_of_leaky_relu(self) -> np.array:
        """Calculate the derivative of the activation function."""
        return np.where(self.z < 0, self.alpha, 1)

    def forward(
        self, X: np.array, activation_function: str = "leaky_relu"
    ) -> np.ndarray:
        """Calculate forward pass with activation function."""
        # initialize weights
        if self.w is None:
            input_size = X.shape[-1]
            self.w = np.random.default_rng().random(input_size) * 0.1

        # compute linear transformation
        self.z = np.dot(X, self.w) + self.b

        # compute activation transformation
        if activation_function == "leaky_relu":
            self.a = self.leaky_relu()
        else:
            raise ValueError(
                f"Activation function {activation_function} not implemented (yet)"
            )

        return self.a

    def gradient(
        self, X: np.array, y_true: np.array, activation_function: str = "leaky_relu"
    ) -> np.array:
        """Gradients for squared error loss: J = (y_pred - y_true)^2."""
        self.forward(X, activation_function=activation_function)
        error = self.a - y_true

        leaky_relu_derivative = self.derivative_of_leaky_relu()

        dJ_dw = np.dot((error * leaky_relu_derivative), X)
        dJ_db = np.sum(error * leaky_relu_derivative)

        return dJ_dw, dJ_db

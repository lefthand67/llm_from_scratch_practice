import numpy as np


class Neuron:
    """Implement single neuron class."""

    def __init__(self: "Neuron") -> None:
        """Initialize neuron entity."""
        self.w: np.ndarray = None
        self.b: np.float64 = np.float64(0.0)
        # linear transformation
        self.z: np.ndarray = None
        # activation transformation
        self.a: np.ndarray = None
        # activation function hyperparameter
        self.alpha: np.float32 = np.float32(0.01)
        # learning rate for weights update
        self.lr: np.float32 = np.float32(0.01)

    def forward(
        self,
        X: np.ndarray,
        alpha: np.float32 = None,
        activation_f: str = "leaky_relu",
    ) -> np.ndarray:
        """Calculate forward pass with activation function."""
        # initialize weights
        if self.w is None:
            # size of one example
            input_size = X.shape[-1]
            self.w = np.random.default_rng().random(input_size) + 0.001

        # compute linear transformation
        self.z = np.dot(X, self.w) + self.b

        # compute activation transformation
        if alpha is not None:
            self.alpha = alpha
        if activation_f == "leaky_relu":
            self.a = np.maximum(self.alpha * self.z, self.z)
        else:
            raise ValueError(
                f"Activation function {activation_f} not implemented (yet)"
            )

        return self.a

    def gradient(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
        alpha: np.float32 = None,
        activation_f: str = "leaky_relu",
    ) -> np.ndarray:
        """Gradients for squared error loss: J = (y_pred - y_true)^2 / 2."""
        if alpha is not None:
            self.alpha = alpha

        # make forward pass
        self.forward(
            X,
            activation_f=activation_f,
        )

        # compute J'
        dJ_da = self.a - y_true

        # compute a'
        if activation_f == "leaky_relu":
            da_dz = np.where(self.z < 0, self.alpha, 1)
        else:
            raise ValueError(
                f"Activation function {activation_f} not implemented (yet)"
            )

        dJ_dw = np.dot((dJ_da * da_dz), X)
        dJ_db = np.sum(dJ_da * da_dz)

        return dJ_dw, dJ_db


class TestNeuron:
    """Test Neuron class."""

    def __init__(self: "TestNeuron") -> None:
        """Initialize test class."""
        self.test_cases: list = list()

    def create_neuron_test_case(
        self,
        name: str = "negative_gradients_increase_weights",
        X: np.ndarray = np.array([[1.0, 2.0]]),
        y_true: np.ndarray = np.array([1.5]),
        w: np.ndarray = np.array([0.3, 0.4]),
        b: np.float64 = np.float64(0.1),
        lr: np.float32 = np.float32(0.01),
        activation_f: str = "leaky_relu",
        cost_f: str = "mse",
    ):
        """
        Create a complete single neuron test case.

        Base case is negative gradient which inscreases weights.
        """
        # final result
        result = {
            "name": name,
            "initial_X": X,
            "y_true": y_true,
            "initial_w": w,
            "initial_b": b,
            "learning_rate": lr,
        }

        # linear transformation
        z = np.dot(X, w) + b
        result["z"] = z

        # activation function
        alpha = 0.01
        if activation_f == "leaky_relu":
            a = np.maximum(z * alpha, z)
        result["a"] = a

        # cost function
        if cost_f == "mse":
            u = a - y_true
            J = u**2 / 2
        else:
            raise ValueError(f'Cost function "{cost_f}" is not implemented yet.')

        result["J"] = J

        # backpropagation

        # J'
        dJ_da = u
        result["dJ_da"] = dJ_da

        # a'
        da_dz = np.where(z < 0, alpha, 1)
        result["da_dz"] = da_dz

        # z'
        dz_dw = X
        result["dz_dw"] = dz_dw

        # b'
        dz_b = 1
        result["dz_b"] = dz_b

        dJ_dw = np.dot(dJ_da * da_dz, dz_dw)  # where dz_dw = X
        result["dJ_dw"] = dJ_dw

        dJ_db = np.sum(dJ_da * da_dz) * dz_b  # where dz_b = 1
        result["dJ_db"] = dJ_db

        # w - lr * (-gradient) = w + lr*|gradient|
        result["expected_w"] = np.subtract(w, (lr * dJ_dw))
        result["expected_b"] = np.subtract(b, (lr * dJ_db))

        self.test_cases.append(result)

        return self.test_cases

import numpy as np

from neuron import Neuron


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

    my_neuron = Neuron()
    my_neuron.gradient(X, y)

    # print("my_neuron.input_size:", my_neuron.input_size)
    print("my_neuron.w:", my_neuron.w)
    print("my_neuron.b:", my_neuron.b)
    # print("linear transformation:", my_neuron._get_linear_transformation(X))
    print("linear transformation:", my_neuron.z)
    print("Activation transformation:", my_neuron.a)
    print(
        "my_neuron.gradient(y_true=y, X=X):",
        my_neuron.gradient(X=X, y_true=y),
    )


if __name__ == "__main__":
    main()

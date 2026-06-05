# Sigmoid activation function
# Formula: s(x) = 1 / (1 + e^(-x))
# Squashes any real number into (0, 1), which makes it natural for binary output layers
# where you want to interpret the output as a probability. The problem is that for very
# large or very small inputs the output saturates near 0 or 1, and the gradient goes
# almost to zero, which slows down learning in deep hidden layers.
# Derivative: s(x) * (1 - s(x))

import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)


if __name__ == "__main__":
    x = np.array([1, 2, 3])
    print("sigmoid:   ", sigmoid(x))
    print("derivative:", sigmoid_derivative(x))

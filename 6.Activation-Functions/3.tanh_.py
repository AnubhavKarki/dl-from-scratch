# Tanh activation function
# Formula: tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
# Maps any real number to (-1, 1). Because the output is zero-centered, gradients during
# backprop tend to be more balanced than with sigmoid, which makes tanh generally better
# for hidden layers. It still saturates at the extremes so deep networks can still suffer
# from vanishing gradients, but less severely than sigmoid.
# Derivative: 1 - tanh^2(x), which you can compute cheaply from the forward pass output.

import numpy as np


def tanh(x):
    return np.tanh(x)


def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2 # OR 1 - np.square(np.tanh(x))


if __name__ == "__main__":
    x = np.array([-2, -1, 0, 1, 2], dtype=float)
    print("tanh:      ", tanh(x))
    print("derivative:", tanh_derivative(x))

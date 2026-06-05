# ReLU (Rectified Linear Unit) activation function
# Formula: relu(x) = max(0, x)
# The simplest fix to the vanishing gradient problem. For any positive input the gradient
# is exactly 1, so gradients flow through without shrinking. Neurons with negative inputs
# output zero and pass no gradient back, which keeps the network sparse and fast.
# The downside is the dying ReLU problem: a neuron that always gets negative input stops
# learning entirely because its gradient is permanently zero.
# Derivative: 1 if x > 0, else 0 (a step function).

import numpy as np


def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return (x > 0).astype(float)


if __name__ == "__main__":
    x = np.array([-3, -1, 0, 1, 3], dtype=float)
    print("relu:      ", relu(x))
    print("derivative:", relu_derivative(x))

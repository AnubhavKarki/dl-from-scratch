# Softmax activation function
# Formula: softmax(x_i) = e^x_i / sum(e^x_j) for all j in the vector
# Turns a vector of raw scores (logits) into a probability distribution that sums to 1.
# Each output is in (0, 1) and all outputs together sum to exactly 1, so you can read
# them directly as class probabilities. Subtracting the row max before exp is a standard
# numerical stability trick that prevents overflow without changing the result.
# Derivative: the full form is a Jacobian matrix (diag(s) - s * s^T), but in practice
# when paired with cross-entropy loss the two cancel to just y_hat - y.

import numpy as np

def softmax(x):
    # subtract row max before exp for numerical stability
    x_exp = np.exp(x - np.max(x, axis=1, keepdims=True))
    return x_exp / np.sum(x_exp, axis=1, keepdims=True)


def softmax_jacobian(s):
    # s is a 1D softmax output vector of shape (n,)
    # returns the (n, n) Jacobian: diag(s) - s * s^T
    return np.diag(s) - np.outer(s, s)


def softmax_cross_entropy_gradient(y_hat, y):
    # collapsed gradient of softmax + cross-entropy w.r.t. pre-softmax logits
    # the Jacobian and cross-entropy gradient cancel to just y_hat - y
    return y_hat - y


if __name__ == "__main__":
    x = np.array([[9, 2, 5, 0, 0], [7, 5, 0, 0, 0]])
    s = softmax(x)
    y = np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]])
    print("softmax:\n", s)
    print("jacobian (row 0):\n", softmax_jacobian(s[0]))
    print("collapsed gradient:\n", softmax_cross_entropy_gradient(s, y))

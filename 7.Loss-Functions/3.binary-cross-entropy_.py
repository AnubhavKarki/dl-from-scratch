# Binary Cross-Entropy Loss (BCE)
# Formula: L = -(1/m) * sum(y * log(y_hat) + (1 - y) * log(1 - y_hat))
# Derived from maximum likelihood estimation under a Bernoulli distribution. When the
# true label is 1 it scores how close y_hat is to 1 via -log(y_hat), and when the label
# is 0 it scores via -log(1 - y_hat). A confident correct prediction gets near-zero loss,
# a confident wrong prediction gets a very large loss. This asymmetry is what makes it
# much better than MSE for binary classification.
# Gradient w.r.t. y_hat: -(y / y_hat) + (1 - y) / (1 - y_hat)
# When combined with a sigmoid output the gradient simplifies further to just y_hat - y.

import numpy as np


def bce_loss(y_hat, y):
    m = y.shape[1]
    return -(1 / m) * np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))


def bce_gradient(y_hat, y):
    # gradient of BCE w.r.t. y_hat (pre-sigmoid activation if combined)
    return -(y / y_hat) + (1 - y) / (1 - y_hat)


if __name__ == "__main__":
    y_hat = np.array([[0.9, 0.2, 0.1, 0.7]])
    y = np.array([[1.0, 0.0, 0.0, 1.0]])
    print("BCE loss:    ", bce_loss(y_hat, y))
    print("BCE gradient:", bce_gradient(y_hat, y))

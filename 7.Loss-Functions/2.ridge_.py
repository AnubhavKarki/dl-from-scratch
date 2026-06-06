# L2 Loss (Ridge)
# Formula: L2 = sum((y - y_hat)^2)
# Measures error as the sum of squared differences. Squaring amplifies large errors more
# than small ones, so the model is penalized harder for big mistakes and pushed to keep
# all predictions reasonably close to the targets. As a regularizer it shrinks weights
# toward zero smoothly rather than all the way to zero like L1.
# Gradient: 2 * (y_hat - y), which scales with the size of the error.

import numpy as np


def l2_loss(y_hat, y):
    return np.sum((y - y_hat) ** 2)


def l2_gradient(y_hat, y):
    return 2 * (y_hat - y)


if __name__ == "__main__":
    y_hat = np.array([0.9, 0.2, 0.1, 0.4, 0.9])
    y = np.array([1.0, 0.0, 0.0, 1.0, 1.0])
    print("L2 loss:    ", l2_loss(y_hat, y))
    print("L2 gradient:", l2_gradient(y_hat, y))

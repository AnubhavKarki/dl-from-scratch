# L1 Loss (Lasso)
# Formula: L1 = sum(|y - y_hat|)
# Measures error as the sum of absolute differences. Because the penalty grows linearly
# with error size, L1 is less sensitive to large outliers than L2. It also tends to
# produce sparse solutions when used as a regularizer, pushing less important weights
# all the way to zero rather than just making them small.
# Gradient: sign(y_hat - y), which is either +1 or -1 depending on the direction of error.
# Note: the gradient is undefined at exactly zero but in practice just treated as 0 there.

import numpy as np


def l1_loss(y_hat, y):
    return np.sum(np.abs(y - y_hat))


def l1_gradient(y_hat, y):
    return np.sign(y_hat - y)


if __name__ == "__main__":
    y_hat = np.array([0.9, 0.2, 0.1, 0.4, 0.9])
    y     = np.array([1.0, 0.0, 0.0, 1.0, 1.0])
    print("L1 loss:    ", l1_loss(y_hat, y))
    print("L1 gradient:", l1_gradient(y_hat, y))

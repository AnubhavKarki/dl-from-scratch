# Vanilla SGD: W^[l] = W^[l] - lr * dW^[l]  for each layer l
#              b^[l] = b^[l] - lr * db^[l]
#
# The same update rule covers batch GD, stochastic GD, and mini-batch GD.
# The only difference between the three is how many examples were used to
# compute the gradient before calling this update:
#   Batch GD:     all m examples — low variance, slow per iteration on large data
#   SGD:          1 example — high variance, fast per step, noisy convergence path
#   Mini-Batch:   b examples — the practical default in deep learning
#
# Params and grads are lists of (W, b) and (dW, db) tuples respectively.

import numpy as np


def sgd(params, grads, lr):
    return [(W - lr * dW, b - lr * db) for (W, b), (dW, db) in zip(params, grads)]


if __name__ == "__main__":
    np.random.seed(1)
    params = [(np.random.randn(3, 4), np.zeros((3, 1))),
              (np.random.randn(1, 3), np.zeros((1, 1)))]
    grads  = [(np.random.randn(3, 4), np.random.randn(3, 1)),
              (np.random.randn(1, 3), np.random.randn(1, 1))]

    updated = sgd(params, grads, lr=0.01)
    print(f"W[0] mean before: {params[0][0].mean():.4f}  after: {updated[0][0].mean():.4f}")
    print(f"W[1] mean before: {params[1][0].mean():.4f}  after: {updated[1][0].mean():.4f}")

# Inverted Dropout
# Forward: D = (rand(shape) < keep_prob),  A = A * D / keep_prob
# Backward: dA = dA * D / keep_prob  (same mask, same rescaling)
#
# Dropout randomly zeros out a fraction of neurons each training step, forcing the
# network not to rely on any single path. Dividing by keep_prob (the inverted part)
# keeps the expected value of each activation unchanged, so no adjustment is needed
# at test time — you just use the full network as-is.
# The mask D must be saved during the forward pass and reused in the backward pass
# so that gradients flow through exactly the same connections that activated.

import numpy as np


def dropout_forward(A, keep_prob, seed=None):
    if seed is not None:
        np.random.seed(seed)
    D = (np.random.rand(*A.shape) < keep_prob)
    A_dropped = (A * D) / keep_prob
    return A_dropped, D


def dropout_backward(dA, D, keep_prob):
    return (dA * D) / keep_prob


if __name__ == "__main__":
    np.random.seed(1)
    A = np.random.randn(3, 5)
    A_dropped, D = dropout_forward(A, keep_prob=0.8, seed=1)
    print("original A mean:   ", A.mean().round(4))
    print("dropped  A mean:   ", A_dropped.mean().round(4))
    print("fraction zeroed:   ", (D == 0).mean().round(2))

    dA = np.random.randn(*A.shape)
    dA_back = dropout_backward(dA, D, keep_prob=0.8)
    print("dA nonzero match:  ", np.all((dA_back != 0) == (D == 1)))

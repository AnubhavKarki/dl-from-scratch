# Weight Initialization Strategies
# Zeros: W = 0 — breaks symmetry, all neurons learn identically, never use for hidden layers
# Random large: W = randn * 10 — exploding gradients, saturates activations
# He: W = randn * sqrt(2 / n_in) — designed for ReLU, keeps variance stable across layers
# Xavier: W = randn * sqrt(1 / n_in) — designed for sigmoid/tanh, same idea with factor 1
#
# The core problem initialization solves: if weights start too large, activations saturate
# and gradients vanish. Too small and signals decay to zero through depth. The sqrt(2/n)
# or sqrt(1/n) scaling keeps the variance of activations roughly constant layer to layer.

import numpy as np


def initialize_zeros(layer_dims):
    return [(np.zeros((n_out, n_in)), np.zeros((n_out, 1)))
            for n_in, n_out in zip(layer_dims[:-1], layer_dims[1:])]


def initialize_random(layer_dims, scale=10):
    np.random.seed(3)
    return [(np.random.randn(n_out, n_in) * scale, np.zeros((n_out, 1)))
            for n_in, n_out in zip(layer_dims[:-1], layer_dims[1:])]


def initialize_he(layer_dims):
    np.random.seed(3)
    return [(np.random.randn(n_out, n_in) * np.sqrt(2.0 / n_in), np.zeros((n_out, 1)))
            for n_in, n_out in zip(layer_dims[:-1], layer_dims[1:])]


def initialize_xavier(layer_dims):
    np.random.seed(3)
    return [(np.random.randn(n_out, n_in) * np.sqrt(1.0 / n_in), np.zeros((n_out, 1)))
            for n_in, n_out in zip(layer_dims[:-1], layer_dims[1:])]


if __name__ == "__main__":
    dims = [5, 4, 3, 1]
    for name, layers in [
        ("zeros",  initialize_zeros(dims)),
        ("random", initialize_random(dims)),
        ("he",     initialize_he(dims)),
        ("xavier", initialize_xavier(dims)),
    ]:
        W, _ = layers[0]
        print(f"{name}  W[0] mean={W.mean():.4f}  std={W.std():.4f}")

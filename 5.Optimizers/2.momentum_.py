# SGD with Momentum (Polyak, 1964)
# Maintains an exponentially weighted moving average of past gradients (velocity v):
#   v_dW = beta * v_dW + (1 - beta) * dW
#   W    = W - lr * v_dW
#
# The velocity accumulates gradient direction across steps. In dimensions where the
# gradient consistently points the same way, velocity builds up and the step grows.
# In dimensions where the gradient oscillates in sign, velocity cancels out and the
# step shrinks — this is why momentum damps oscillations in ravine-shaped loss surfaces.
#
# beta = 0 recovers vanilla SGD. beta = 0.9 is the standard default.
#
# State: velocity is a list of (vW, vb) tuples initialised to zeros.

import numpy as np


def init_velocity(params):
    return [(np.zeros_like(W), np.zeros_like(b)) for W, b in params]


def momentum(params, grads, v, beta, lr):
    new_v = [(beta * vW + (1 - beta) * dW, beta * vb + (1 - beta) * db)
             for (vW, vb), (dW, db) in zip(v, grads)]
    new_params = [(W - lr * vW, b - lr * vb)
                  for (W, b), (vW, vb) in zip(params, new_v)]
    return new_params, new_v


if __name__ == "__main__":
    np.random.seed(1)
    params = [(np.random.randn(3, 4), np.zeros((3, 1))),
              (np.random.randn(1, 3), np.zeros((1, 1)))]
    grads  = [(np.random.randn(3, 4), np.random.randn(3, 1)),
              (np.random.randn(1, 3), np.random.randn(1, 1))]
    v = init_velocity(params)

    for _ in range(3):
        params, v = momentum(params, grads, v, beta=0.9, lr=0.01)

    print(f"after 3 steps  W[0] mean: {params[0][0].mean():.4f}")
    print(f"velocity  v[0] W mean:    {v[0][0].mean():.4f}")

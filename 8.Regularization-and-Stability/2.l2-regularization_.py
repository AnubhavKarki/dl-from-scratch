# L2 Regularization (Weight Decay)
# Cost with penalty: J_reg = J + (λ / 2m) * sum_l( ||W[l]||_F^2 )
# Gradient update: dW[l] += (λ / m) * W[l]
#
# Adding the Frobenius norm of each weight matrix to the cost penalizes large weights.
# During backprop this adds a decay term to every dW, which pulls weights toward zero
# each step. The result is a smoother decision boundary that generalizes better because
# the model cannot rely on any single large weight.

import numpy as np


def compute_cost_with_l2(AL, Y, layers, lambd):
    m = Y.shape[1]
    cross_entropy = -(1/m) * np.sum(Y * np.log(AL) + (1 - Y) * np.log(1 - AL))
    l2_penalty = (lambd / (2 * m)) * sum(np.sum(W ** 2) for W, _ in layers)
    return float(np.squeeze(cross_entropy + l2_penalty))


def backward_with_l2(grads, layers, lambd, m):
    return [(dW + (lambd / m) * W, db) for (dW, db), (W, _) in zip(grads, layers)]


if __name__ == "__main__":
    np.random.seed(1)
    AL = np.array([[0.8, 0.3, 0.6]])
    Y  = np.array([[1.0, 0.0, 1.0]])
    layers = [
        (np.random.randn(4, 3), np.zeros((4, 1))),
        (np.random.randn(1, 4), np.zeros((1, 1))),
    ]
    cost = compute_cost_with_l2(AL, Y, layers, lambd=0.7)
    print(f"regularized cost: {cost:.6f}")

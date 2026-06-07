# Deep L-layer Neural Network: modular building blocks for arbitrary depth
# Architecture: [LINEAR -> RELU] * (L-1) -> LINEAR -> SIGMOID
#
# Forward (one layer): Z[l] = W[l] * A[l-1] + b[l],  A[l] = g(Z[l])
# Cost: J = -(1/m) * sum(Y * log(AL) + (1 - Y) * log(1 - AL))
# Backward (one layer):
#   dZ[l]   = dA[l] * g'(Z[l])
#   dW[l]   = (1/m) * dZ[l] * A[l-1]^T
#   db[l]   = (1/m) * sum(dZ[l])
#   dA[l-1] = W[l]^T * dZ[l]
#
# The key idea is that the backward pass is just the forward pass run in reverse.
# Each layer stores its inputs and pre-activation values in a cache during the forward
# pass, and the backward pass uses that cache to compute local gradients and chain them
# back to the layer before. Making each layer modular means you can stack as many as
# you want without rewriting the math for each depth.

import numpy as np


# --- activations ---


def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))


def relu(Z):
    return np.maximum(0, Z)


# --- parameter initialization ---


def init_2layer(n_x, n_h, n_y):
    np.random.seed(1)
    return [
        (np.random.randn(n_h, n_x) * 0.01, np.zeros((n_h, 1))),
        (np.random.randn(n_y, n_h) * 0.01, np.zeros((n_y, 1))),
    ]


def init_deep(layer_dims):
    np.random.seed(3)
    return [
        (np.random.randn(n_out, n_in) * 0.01, np.zeros((n_out, 1)))
        for n_in, n_out in zip(layer_dims[:-1], layer_dims[1:])
    ]


# --- forward propagation ---


def layer_forward(A_prev, W, b, activation):
    Z = np.dot(W, A_prev) + b
    A = relu(Z) if activation == "relu" else sigmoid(Z)
    return A, (A_prev, W, Z)


def forward(X, params):
    caches = []
    A = X
    for W, b in params[:-1]:
        A, cache = layer_forward(A, W, b, "relu")
        caches.append(cache)
    W, b = params[-1]
    AL, cache = layer_forward(A, W, b, "sigmoid")
    caches.append(cache)
    return AL, caches


# --- cost ---


def compute_cost(AL, Y):
    m = Y.shape[1]
    return float(
        np.squeeze(-(1 / m) * np.sum(Y * np.log(AL) + (1 - Y) * np.log(1 - AL)))
    )


# --- backward propagation ---


def layer_backward(dA, cache, activation):
    A_prev, W, Z = cache
    if activation == "relu":
        dZ = dA * (Z > 0)
    else:
        s = sigmoid(Z)
        dZ = dA * s * (1 - s)
    m = A_prev.shape[1]
    dW = (1 / m) * np.dot(dZ, A_prev.T)
    db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)
    return dA_prev, dW, db


def backward(AL, Y, caches):
    grads = []
    Y = Y.reshape(AL.shape)
    dA = -(Y / AL - (1 - Y) / (1 - AL))

    dA, dW, db = layer_backward(dA, caches[-1], "sigmoid")
    grads.append((dW, db))

    for cache in reversed(caches[:-1]):
        dA, dW, db = layer_backward(dA, cache, "relu")
        grads.append((dW, db))

    grads.reverse()
    return grads


# --- parameter update ---


def update_params(params, grads, lr):
    return [(W - lr * dW, b - lr * db) for (W, b), (dW, db) in zip(params, grads)]


# --- full training loops ---


def two_layer_model(X, Y, layer_dims, lr=0.0075, epochs=3000, print_cost=False):
    np.random.seed(1)
    (W1, b1), (W2, b2) = init_2layer(*layer_dims)
    costs = []

    for i in range(epochs):
        A1, c1 = layer_forward(X, W1, b1, "relu")
        A2, c2 = layer_forward(A1, W2, b2, "sigmoid")
        cost = compute_cost(A2, Y)

        dA2 = -(Y / A2 - (1 - Y) / (1 - A2))
        dA1, dW2, db2 = layer_backward(dA2, c2, "sigmoid")
        _, dW1, db1 = layer_backward(dA1, c1, "relu")

        W1, b1 = W1 - lr * dW1, b1 - lr * db1
        W2, b2 = W2 - lr * dW2, b2 - lr * db2

        if i % 100 == 0:
            costs.append(cost)
            if print_cost:
                print(f"epoch {i:4d}  cost {cost:.6f}")

    return [(W1, b1), (W2, b2)], costs


def L_layer_model(X, Y, layer_dims, lr=0.0075, epochs=3000, print_cost=False):
    np.random.seed(1)
    params = init_deep(layer_dims)
    costs = []

    for i in range(epochs):
        AL, caches = forward(X, params)
        cost = compute_cost(AL, Y)
        grads = backward(AL, Y, caches)
        params = update_params(params, grads, lr)

        if i % 100 == 0:
            costs.append(cost)
            if print_cost:
                print(f"epoch {i:4d}  cost {cost:.6f}")

    return params, costs


def predict(params, X, Y):
    AL, _ = forward(X, params)
    preds = (AL > 0.5).astype(int)
    acc = np.mean(preds == Y)
    print(f"accuracy: {acc:.4f}")
    return preds


if __name__ == "__main__":
    np.random.seed(1)
    X = np.random.randn(5, 200)
    Y = (np.random.rand(1, 200) > 0.5).astype(float)

    params, costs = L_layer_model(
        X, Y, [5, 4, 3, 1], lr=0.0075, epochs=500, print_cost=True
    )
    predict(params, X, Y)

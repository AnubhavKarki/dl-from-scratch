# Shallow Neural Network: one hidden layer, tanh hidden, sigmoid output
# Forward:
#   Z1 = W1 * X + b1,  A1 = tanh(Z1)
#   Z2 = W2 * A1 + b2, A2 = sigmoid(Z2)
#   J  = -(1/m) * sum(Y * log(A2) + (1 - Y) * log(1 - A2))
# Backward:
#   dZ2 = A2 - Y
#   dW2 = (1/m) * dZ2 * A1^T,  db2 = (1/m) * sum(dZ2)
#   dZ1 = W2^T * dZ2 * (1 - A1^2)   <- tanh derivative applied here
#   dW1 = (1/m) * dZ1 * X^T,   db1 = (1/m) * sum(dZ1)
#
# Adding a hidden layer with a nonlinear activation (tanh here) lets the network learn
# decision boundaries that are not just straight lines. The backward pass uses the chain
# rule layer by layer: the output gradient flows back through the sigmoid, then through
# the tanh in the hidden layer, accumulating weight gradients at each step.

import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def layer_sizes(X, Y):
    return X.shape[0], 4, Y.shape[0]


def initialize_parameters(n_x, n_h, n_y):
    np.random.seed(2)
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}


def forward(X, params):
    W1, b1 = params["W1"], params["b1"]
    W2, b2 = params["W2"], params["b2"]

    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)

    cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
    return A2, cache


def compute_cost(A2, Y):
    m = Y.shape[1]
    return float(-np.squeeze((1 / m) * np.sum(Y * np.log(A2) + (1 - Y) * np.log(1 - A2))))


def backward(params, cache, X, Y):
    m  = X.shape[1]
    W2 = params["W2"]
    A1, A2 = cache["A1"], cache["A2"]

    dZ2 = A2 - Y
    dW2 = (1 / m) * np.dot(dZ2, A1.T)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)

    # tanh derivative: 1 - A1^2
    dZ1 = np.dot(W2.T, dZ2) * (1 - A1 ** 2)
    dW1 = (1 / m) * np.dot(dZ1, X.T)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

    return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}


def update_parameters(params, grads, lr=1.2):
    params["W1"] -= lr * grads["dW1"]
    params["b1"] -= lr * grads["db1"]
    params["W2"] -= lr * grads["dW2"]
    params["b2"] -= lr * grads["db2"]
    return params


def train(X, Y, n_h=4, epochs=10000, lr=1.2, print_cost=False):
    np.random.seed(3)
    n_x, _, n_y = layer_sizes(X, Y)
    params = initialize_parameters(n_x, n_h, n_y)

    for i in range(epochs):
        A2, cache = forward(X, params)
        cost      = compute_cost(A2, Y)
        grads     = backward(params, cache, X, Y)
        params    = update_parameters(params, grads, lr)

        if print_cost and i % 1000 == 0:
            print(f"epoch {i:5d}  cost {cost:.6f}")

    return params


def predict(params, X):
    A2, _ = forward(X, params)
    return (A2 > 0.5).astype(int)


if __name__ == "__main__":
    np.random.seed(1)
    X = np.random.randn(2, 400)
    Y = (np.sum(X ** 2, axis=0, keepdims=True) < 1).astype(float)

    params = train(X, Y, n_h=4, epochs=10000, lr=1.2, print_cost=True)
    preds  = predict(params, X)
    acc    = float(np.dot(Y, preds.T) + np.dot(1 - Y, 1 - preds.T)) / Y.size * 100
    print(f"accuracy: {acc:.1f}%")

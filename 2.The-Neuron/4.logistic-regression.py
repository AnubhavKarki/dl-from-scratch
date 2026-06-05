# Logistic Regression as a Neural Network
# Forward: A = sigmoid(W^T * X + b)
# Loss: J = -(1/m) * sum(Y * log(A) + (1 - Y) * log(1 - A))
# Backward: dW = (1/m) * X * (A - Y)^T,  db = (1/m) * sum(A - Y)
# Update: W = W - lr * dW,  b = b - lr * db
#
# Logistic regression is a single neuron with a sigmoid output trained with binary
# cross-entropy. There is no hidden layer, just a linear combination of inputs passed
# through sigmoid to get a probability. The gradient descent loop here is the same
# machinery used in every neural network, just applied to one set of weights and a bias.

import numpy as np
import copy


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def initialize(dim):
    w = np.zeros((dim, 1))
    b = 0.0
    return w, b


def propagate(w, b, X, Y):
    m = X.shape[1]
    A    = sigmoid(np.dot(w.T, X) + b)
    cost = -(1 / m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))
    dw   = (1 / m) * np.dot(X, (A - Y).T)
    db   = (1 / m) * np.sum(A - Y)
    return {"dw": dw, "db": db}, float(np.squeeze(cost))


def optimize(w, b, X, Y, epochs=100, lr=0.009, print_cost=False):
    w, b  = copy.deepcopy(w), copy.deepcopy(b)
    costs = []

    for i in range(epochs):
        grads, cost = propagate(w, b, X, Y)
        w -= lr * grads["dw"]
        b -= lr * grads["db"]

        if i % 100 == 0:
            costs.append(cost)
            if print_cost:
                print(f"epoch {i:4d}  cost {cost:.6f}")

    return {"w": w, "b": b}, grads, costs


def predict(w, b, X):
    A = sigmoid(np.dot(w.T, X) + b)
    return (A > 0.5).astype(int)


def model(X_train, Y_train, X_test, Y_test, epochs=2000, lr=0.5, print_cost=False):
    w, b = initialize(X_train.shape[0])
    params, grads, costs = optimize(w, b, X_train, Y_train, epochs, lr, print_cost)
    w, b = params["w"], params["b"]

    Y_pred_train = predict(w, b, X_train)
    Y_pred_test  = predict(w, b, X_test)

    train_acc = 100 - np.mean(np.abs(Y_pred_train - Y_train)) * 100
    test_acc  = 100 - np.mean(np.abs(Y_pred_test  - Y_test))  * 100
    print(f"train accuracy: {train_acc:.2f}%")
    print(f"test  accuracy: {test_acc:.2f}%")

    return {"w": w, "b": b, "costs": costs,
            "Y_pred_train": Y_pred_train, "Y_pred_test": Y_pred_test}


if __name__ == "__main__":
    np.random.seed(1)
    X_train = np.random.randn(3, 100)
    Y_train = (np.random.rand(1, 100) > 0.5).astype(float)
    X_test  = np.random.randn(3, 20)
    Y_test  = (np.random.rand(1, 20) > 0.5).astype(float)
    model(X_train, Y_train, X_test, Y_test, epochs=1000, lr=0.01, print_cost=True)

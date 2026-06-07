import numpy as np

# softmax over columns, one sample per column
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


# single forward step through one RNN cell
# returns next hidden state, prediction, and cache for backprop
def rnn_cell_forward(xt, a_prev, parameters):
    Wax = parameters["Wax"]
    Waa = parameters["Waa"]
    Wya = parameters["Wya"]
    ba  = parameters["ba"]
    by  = parameters["by"]

    # hidden state: tanh(Waa·a_prev + Wax·xt + ba)
    a_next  = np.tanh(np.dot(Waa, a_prev) + np.dot(Wax, xt) + ba)

    # output: softmax(Wya·a_next + by)
    yt_pred = softmax(np.dot(Wya, a_next) + by)

    cache = (a_next, a_prev, xt, parameters)

    return a_next, yt_pred, cache


# runs rnn_cell_forward across all T_x timesteps
# returns hidden states, predictions, and all caches
def rnn_forward(x, a0, parameters):
    caches = []

    n_x, m, T_x = x.shape
    n_y, n_a    = parameters["Wya"].shape

    a      = np.zeros((n_a, m, T_x))
    y_pred = np.zeros((n_y, m, T_x))
    a_next = a0

    for t in range(T_x):
        a_next, yt_pred, cache = rnn_cell_forward(x[:, :, t], a_next, parameters)
        a[:, :, t]      = a_next
        y_pred[:, :, t] = yt_pred
        caches.append(cache)

    caches = (caches, x)

    return a, y_pred, caches


# backward pass through a single RNN cell
# returns gradients for dx, da_prev, dWax, dWaa, dba
def rnn_cell_backward(da_next, cache):
    (a_next, a_prev, xt, parameters) = cache

    Wax = parameters["Wax"]
    Waa = parameters["Waa"]
    ba  = parameters["ba"]

    # gradient through tanh: d/dx tanh(x) = 1 - tanh²(x)
    dtanh = da_next * (1 - np.square(np.tanh(np.dot(Wax, xt) + np.dot(Waa, a_prev) + ba)))

    # gradients w.r.t. input weights and input
    dxt  = np.dot(Wax.T, dtanh)
    dWax = np.dot(dtanh, xt.T)

    # gradients w.r.t. recurrent weights and previous hidden state
    da_prev = np.dot(Waa.T, dtanh)
    dWaa    = np.dot(dtanh, a_prev.T)

    # gradient w.r.t. bias
    dba = np.sum(dtanh, axis=1, keepdims=True)

    gradients = {"dxt": dxt, "da_prev": da_prev, "dWax": dWax, "dWaa": dWaa, "dba": dba}

    return gradients


# full BPTT over the sequence, accumulates gradients at each timestep
# returns dx, da0, dWax, dWaa, dba
def rnn_backward(da, caches):
    (caches, x)               = caches
    (a1, a0, x1, parameters)  = caches[0]

    n_a, m, T_x = da.shape
    n_x, m      = x1.shape

    dx       = np.zeros((n_x, m, T_x))
    dWax     = np.zeros((n_a, n_x))
    dWaa     = np.zeros((n_a, n_a))
    dba      = np.zeros((n_a, 1))
    da0      = np.zeros((n_a, m))
    da_prevt = np.zeros((n_a, m))

    # iterate backwards through time, accumulating gradients
    for t in reversed(range(T_x)):
        gradients = rnn_cell_backward(da[:, :, t] + da_prevt, caches[t])
        dxt, da_prevt, dWaxt, dWaat, dbat = (
            gradients["dxt"], gradients["da_prev"],
            gradients["dWax"], gradients["dWaa"], gradients["dba"]
        )
        dx[:, :, t] = dxt
        dWax += dWaxt
        dWaa += dWaat
        dba  += dbat

    # gradient at t=0 is the gradient w.r.t. the initial hidden state
    da0 = da_prevt

    gradients = {"dx": dx, "da0": da0, "dWax": dWax, "dWaa": dWaa, "dba": dba}

    return gradients


if __name__ == "__main__":
    np.random.seed(1)

    n_x, m, T_x, n_a, n_y = 3, 10, 4, 5, 2

    x  = np.random.randn(n_x, m, T_x)
    a0 = np.random.randn(n_a, m)

    parameters = {
        "Wax": np.random.randn(n_a, n_x),
        "Waa": np.random.randn(n_a, n_a),
        "Wya": np.random.randn(n_y, n_a),
        "ba":  np.random.randn(n_a, 1),
        "by":  np.random.randn(n_y, 1),
    }

    print("RNN Forward Pass")
    a, y_pred, caches = rnn_forward(x, a0, parameters)
    print(f"a shape:         {a.shape}")
    print(f"y_pred shape:    {y_pred.shape}")
    print(f"a[last timestep] mean: {a[:, :, -1].mean():.4f}")

    print("\nRNN Backward Pass")
    da = np.random.randn(n_a, m, T_x)
    gradients = rnn_backward(da, caches)
    for k, v in gradients.items():
        print(f"{k} shape: {v.shape}")

import numpy as np


# softmax over columns, one sample per column
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# single forward step through one LSTM cell
# ft=forget gate, it=input gate, cct=candidate cell, ot=output gate
# returns next hidden state, next cell state, prediction, and cache for backprop
def lstm_cell_forward(xt, a_prev, c_prev, parameters):
    Wf = parameters["Wf"]
    bf = parameters["bf"]
    Wi = parameters["Wi"]
    bi = parameters["bi"]
    Wc = parameters["Wc"]
    bc = parameters["bc"]
    Wo = parameters["Wo"]
    bo = parameters["bo"]
    Wy = parameters["Wy"]
    by = parameters["by"]

    n_x, m  = xt.shape
    n_y, n_a = Wy.shape

    # concatenate previous hidden state and current input along the feature axis
    concat = np.concatenate((a_prev, xt), axis=0)

    # forget gate: how much of c_prev to keep
    ft = sigmoid(np.dot(Wf, concat) + bf)

    # input gate: how much of the candidate to write
    it = sigmoid(np.dot(Wi, concat) + bi)

    # candidate cell value
    cct = np.tanh(np.dot(Wc, concat) + bc)

    # new cell state: forget old + write new candidate
    c_next = (ft * c_prev) + (it * cct)

    # output gate: how much of the cell to expose
    ot = sigmoid(np.dot(Wo, concat) + bo)

    # next hidden state
    a_next = ot * np.tanh(c_next)

    # prediction
    yt_pred = softmax(np.dot(Wy, a_next) + by)

    cache = (a_next, c_next, a_prev, c_prev, ft, it, cct, ot, xt, parameters)

    return a_next, c_next, yt_pred, cache


# runs lstm_cell_forward across all T_x timesteps
# returns hidden states, predictions, cell states, and all caches
def lstm_forward(x, a0, parameters):
    caches = []

    Wy = parameters['Wy']

    n_x, m, T_x = x.shape
    n_y, n_a    = parameters['Wy'].shape

    a = np.zeros((n_a, m, T_x))
    c = np.zeros((n_a, m, T_x))
    y = np.zeros((n_y, m, T_x))

    a_next = a0
    c_next = np.zeros((n_a, m))

    for t in range(T_x):
        xt = x[:, :, t]
        a_next, c_next, yt, cache = lstm_cell_forward(xt, a_next, c_next, parameters)
        a[:, :, t] = a_next
        c[:, :, t] = c_next
        y[:, :, t] = yt
        caches.append(cache)

    caches = (caches, x)

    return a, y, c, caches


# backward pass through a single LSTM cell
# returns gradients for dxt, da_prev, dc_prev, and all gate weights and biases
def lstm_cell_backward(da_next, dc_next, cache):
    (a_next, c_next, a_prev, c_prev, ft, it, cct, ot, xt, parameters) = cache

    n_x, m = xt.shape
    n_a, m = a_next.shape

    # gate gradients — chain rule through each gate activation
    dot  = da_next * np.tanh(c_next) * ot * (1 - ot)
    dcct = (dc_next * it  + ot * (1 - np.square(np.tanh(c_next))) * it  * da_next) * (1 - np.square(cct))
    dit  = (dc_next * cct + ot * (1 - np.square(np.tanh(c_next))) * cct * da_next) * it  * (1 - it)
    dft  = (dc_next * c_prev + ot * (1 - np.square(np.tanh(c_next))) * c_prev * da_next) * ft * (1 - ft)

    # weight gradients: d_gate · concat^T
    concat = np.concatenate((a_prev, xt), axis=0)
    dWf = np.dot(dft,  concat.T)
    dWi = np.dot(dit,  concat.T)
    dWc = np.dot(dcct, concat.T)
    dWo = np.dot(dot,  concat.T)

    # bias gradients: sum over batch dimension
    dbf = np.sum(dft,  axis=1, keepdims=True)
    dbi = np.sum(dit,  axis=1, keepdims=True)
    dbc = np.sum(dcct, axis=1, keepdims=True)
    dbo = np.sum(dot,  axis=1, keepdims=True)

    # W[:, :n_a] is the recurrent block, W[:, n_a:] is the input block
    da_prev = (np.dot(parameters['Wf'][:, :n_a].T, dft)  +
               np.dot(parameters['Wi'][:, :n_a].T, dit)  +
               np.dot(parameters['Wc'][:, :n_a].T, dcct) +
               np.dot(parameters['Wo'][:, :n_a].T, dot))

    dc_prev = dc_next * ft + ot * (1 - np.square(np.tanh(c_next))) * ft * da_next

    dxt     = (np.dot(parameters['Wf'][:, n_a:].T, dft)  +
               np.dot(parameters['Wi'][:, n_a:].T, dit)  +
               np.dot(parameters['Wc'][:, n_a:].T, dcct) +
               np.dot(parameters['Wo'][:, n_a:].T, dot))

    gradients = {
        "dxt": dxt, "da_prev": da_prev, "dc_prev": dc_prev,
        "dWf": dWf, "dbf": dbf,
        "dWi": dWi, "dbi": dbi,
        "dWc": dWc, "dbc": dbc,
        "dWo": dWo, "dbo": dbo,
    }

    return gradients


# full BPTT over the sequence, accumulates gradients at each timestep
# returns dx, da0, and all gate weight/bias gradients
def lstm_backward(da, caches):
    (caches, x) = caches
    (a1, c1, a0, c0, f1, i1, cc1, o1, x1, parameters) = caches[0]

    n_a, m, T_x = da.shape
    n_x, m      = x1.shape

    dx       = np.zeros((n_x, m, T_x))
    da0      = np.zeros((n_a, m))
    da_prevt = np.zeros((n_a, m))
    dc_prevt = np.zeros((n_a, m))
    dWf      = np.zeros((n_a, n_a + n_x))
    dWi      = np.zeros((n_a, n_a + n_x))
    dWc      = np.zeros((n_a, n_a + n_x))
    dWo      = np.zeros((n_a, n_a + n_x))
    dbf      = np.zeros((n_a, 1))
    dbi      = np.zeros((n_a, 1))
    dbc      = np.zeros((n_a, 1))
    dbo      = np.zeros((n_a, 1))

    # iterate backwards through time, accumulating gradients
    for t in reversed(range(T_x)):
        gradients = lstm_cell_backward(da[:, :, t] + da_prevt, dc_prevt, caches[t])
        da_prevt     = gradients['da_prev']
        dc_prevt     = gradients['dc_prev']
        dx[:, :, t]  = gradients['dxt']
        dWf += gradients['dWf']
        dWi += gradients['dWi']
        dWc += gradients['dWc']
        dWo += gradients['dWo']
        dbf += gradients['dbf']
        dbi += gradients['dbi']
        dbc += gradients['dbc']
        dbo += gradients['dbo']

    # gradient at t=0 is the gradient w.r.t. the initial hidden state
    da0 = gradients['da_prev']

    gradients = {
        "dx": dx, "da0": da0,
        "dWf": dWf, "dbf": dbf,
        "dWi": dWi, "dbi": dbi,
        "dWc": dWc, "dbc": dbc,
        "dWo": dWo, "dbo": dbo,
    }

    return gradients


if __name__ == "__main__":
    np.random.seed(1)

    n_x, m, T_x, n_a, n_y = 3, 10, 7, 5, 2

    x  = np.random.randn(n_x, m, T_x)
    a0 = np.random.randn(n_a, m)

    parameters = {
        "Wf": np.random.randn(n_a, n_a + n_x),
        "bf": np.random.randn(n_a, 1),
        "Wi": np.random.randn(n_a, n_a + n_x),
        "bi": np.random.randn(n_a, 1),
        "Wc": np.random.randn(n_a, n_a + n_x),
        "bc": np.random.randn(n_a, 1),
        "Wo": np.random.randn(n_a, n_a + n_x),
        "bo": np.random.randn(n_a, 1),
        "Wy": np.random.randn(n_y, n_a),
        "by": np.random.randn(n_y, 1),
    }

    print("LSTM Forward Pass")
    a, y, c, caches = lstm_forward(x, a0, parameters)
    print(f"a shape:         {a.shape}")
    print(f"y shape:         {y.shape}")
    print(f"c shape:         {c.shape}")
    print(f"a[last timestep] mean: {a[:, :, -1].mean():.4f}")

    print("\nLSTM Backward Pass")
    da = np.random.randn(n_a, m, T_x)
    gradients = lstm_backward(da, caches)
    for k, v in gradients.items():
        print(f"{k} shape: {v.shape}")

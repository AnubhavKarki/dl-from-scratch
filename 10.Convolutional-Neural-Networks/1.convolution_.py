# 2D Convolution and Pooling: forward and backward passes
# Conv output size: n_out = floor((n_in - f + 2*pad) / stride) + 1
# Conv forward: Z[i,h,w,c] = sum(A_prev[i, h*s:h*s+f, w*s:w*s+f, :] * W[:,:,:,c]) + b[c]
# Pool output size: n_out = floor((n_in - f) / stride) + 1
#
# A filter (f x f x n_C_prev) slides across the input volume and computes a dot product
# at every position. Each filter detects a different local pattern; stacking n_C filters
# gives an output volume of depth n_C. Padding keeps spatial size intact across layers.
#
# Conv backward: dA accumulates W * dZ at each slice position; dW accumulates a_slice * dZ;
# db sums dZ over spatial positions.
# Max pool backward: gradient flows only to whichever element was the max (mask of 1s/0s).
# Avg pool backward: gradient is spread equally across every element in the window (dZ / f^2).

import numpy as np


def zero_pad(X, pad):
    return np.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)))


def conv_forward(A_prev, W, b, stride=1, pad=0):
    m, n_H_prev, n_W_prev, _ = A_prev.shape
    f, _, _, n_C = W.shape

    n_H = int((n_H_prev - f + 2 * pad) / stride) + 1
    n_W = int((n_W_prev - f + 2 * pad) / stride) + 1

    Z = np.zeros((m, n_H, n_W, n_C))
    A_prev_pad = zero_pad(A_prev, pad)

    for i in range(m):
        a_pad = A_prev_pad[i]
        for h in range(n_H):
            vs = h * stride
            for w in range(n_W):
                hs = w * stride
                for c in range(n_C):
                    a_slice = a_pad[vs:vs+f, hs:hs+f, :]
                    Z[i, h, w, c] = np.sum(a_slice * W[:, :, :, c]) + b[0, 0, 0, c]

    cache = (A_prev, W, b, stride, pad)
    return Z, cache


def conv_backward(dZ, cache):
    A_prev, W, b, stride, pad = cache
    m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape
    f, _, _, n_C = W.shape
    _, n_H, n_W, _ = dZ.shape

    dA_prev = np.zeros(A_prev.shape)
    dW = np.zeros(W.shape)
    db = np.zeros(b.shape)

    A_prev_pad = zero_pad(A_prev, pad)
    dA_prev_pad = zero_pad(dA_prev, pad)

    for i in range(m):
        a_pad = A_prev_pad[i]
        da_pad = dA_prev_pad[i]
        for h in range(n_H):
            vs = h * stride
            for w in range(n_W):
                hs = w * stride
                for c in range(n_C):
                    a_slice = a_pad[vs:vs+f, hs:hs+f, :]
                    da_pad[vs:vs+f, hs:hs+f, :] += W[:, :, :, c] * dZ[i, h, w, c]
                    dW[:, :, :, c] += a_slice * dZ[i, h, w, c]
                    db[:, :, :, c] += dZ[i, h, w, c]
        dA_prev[i] = da_pad[pad:-pad, pad:-pad, :] if pad > 0 else da_pad

    return dA_prev, dW, db


def pool_forward(A_prev, f, stride, mode="max"):
    m, n_H_prev, n_W_prev, n_C = A_prev.shape

    n_H = int((n_H_prev - f) / stride) + 1
    n_W = int((n_W_prev - f) / stride) + 1

    A = np.zeros((m, n_H, n_W, n_C))

    for i in range(m):
        for h in range(n_H):
            vs = h * stride
            for w in range(n_W):
                hs = w * stride
                for c in range(n_C):
                    window = A_prev[i, vs:vs+f, hs:hs+f, c]
                    A[i, h, w, c] = np.max(window) if mode == "max" else np.mean(window)

    cache = (A_prev, f, stride)
    return A, cache


def pool_backward(dA, cache, mode="max"):
    A_prev, f, stride = cache
    m, _, _, _ = A_prev.shape
    _, n_H, n_W, n_C = dA.shape

    dA_prev = np.zeros(A_prev.shape)

    for i in range(m):
        a_prev = A_prev[i]
        for h in range(n_H):
            vs = h * stride
            for w in range(n_W):
                hs = w * stride
                for c in range(n_C):
                    if mode == "max":
                        window = a_prev[vs:vs+f, hs:hs+f, c]
                        mask = (window == np.max(window))
                        dA_prev[i, vs:vs+f, hs:hs+f, c] += mask * dA[i, h, w, c]
                    else:
                        dA_prev[i, vs:vs+f, hs:hs+f, c] += np.full((f, f), dA[i, h, w, c] / (f * f))

    return dA_prev


if __name__ == "__main__":
    np.random.seed(1)
    A_prev = np.random.randn(2, 5, 7, 4)
    W = np.random.randn(3, 3, 4, 8)
    b = np.random.randn(1, 1, 1, 8)

    Z, cache = conv_forward(A_prev, W, b, stride=2, pad=1)
    print(f"conv fwd  Z mean: {Z.mean():.4f}  shape: {Z.shape}")

    dZ = np.random.randn(*Z.shape)
    dA, dW, db = conv_backward(dZ, cache)
    print(f"conv bwd  dA mean: {dA.mean():.4f}  dW mean: {dW.mean():.4f}")

    A_pool, cache_pool = pool_forward(A_prev, f=3, stride=2, mode="max")
    print(f"pool fwd  A mean: {A_pool.mean():.4f}  shape: {A_pool.shape}")

    dA_pool = np.random.randn(*A_pool.shape)
    dA_prev = pool_backward(dA_pool, cache_pool, mode="max")
    print(f"pool bwd  dA_prev mean: {dA_prev.mean():.4f}")

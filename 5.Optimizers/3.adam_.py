# Adam: Adaptive Moment Estimation (Kingma and Ba, 2014)
# Combines momentum (first moment) with RMSProp (second moment) and adds bias correction:
#
#   v  = beta1 * v + (1 - beta1) * g          first moment  — exponential mean of gradients
#   s  = beta2 * s + (1 - beta2) * g^2        second moment — exponential mean of squared gradients
#   v^ = v / (1 - beta1^t)                    bias-corrected first moment
#   s^ = s / (1 - beta2^t)                    bias-corrected second moment
#   W  = W - lr * v^ / (sqrt(s^) + eps)       parameter update
#
# Bias correction compensates for v and s being initialised to zero, which would otherwise
# bias early estimates toward zero. The correction factor shrinks toward 1 as t grows.
#
# The per-parameter adaptive rate from s^ allows fast directions to slow down automatically
# and slow directions to speed up, reducing the need to hand-tune the learning rate.
#
# Default hyperparameters from the paper: beta1=0.9, beta2=0.999, eps=1e-8.
# State: v and s are lists of (vW, vb) and (sW, sb) tuples initialised to zeros.

import numpy as np


def init_adam(params):
    v = [(np.zeros_like(W), np.zeros_like(b)) for W, b in params]
    s = [(np.zeros_like(W), np.zeros_like(b)) for W, b in params]
    return v, s


def adam(params, grads, v, s, t, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
    new_params, new_v, new_s = [], [], []
    for (W, b), (dW, db), (vW, vb), (sW, sb) in zip(params, grads, v, s):
        vW = beta1 * vW + (1 - beta1) * dW
        vb = beta1 * vb + (1 - beta1) * db
        sW = beta2 * sW + (1 - beta2) * dW ** 2
        sb = beta2 * sb + (1 - beta2) * db ** 2

        vW_c = vW / (1 - beta1 ** t)
        vb_c = vb / (1 - beta1 ** t)
        sW_c = sW / (1 - beta2 ** t)
        sb_c = sb / (1 - beta2 ** t)

        new_params.append((W - lr * vW_c / (np.sqrt(sW_c) + eps),
                           b - lr * vb_c / (np.sqrt(sb_c) + eps)))
        new_v.append((vW, vb))
        new_s.append((sW, sb))

    return new_params, new_v, new_s


if __name__ == "__main__":
    np.random.seed(1)
    params = [(np.random.randn(3, 4), np.zeros((3, 1))),
              (np.random.randn(1, 3), np.zeros((1, 1)))]
    grads  = [(np.random.randn(3, 4), np.random.randn(3, 1)),
              (np.random.randn(1, 3), np.random.randn(1, 1))]
    v, s = init_adam(params)

    for t in range(1, 4):
        params, v, s = adam(params, grads, v, s, t)

    print(f"after 3 steps  W[0] mean: {params[0][0].mean():.4f}")
    print(f"v[0] W mean: {v[0][0].mean():.4f}  s[0] W mean: {s[0][0].mean():.6f}")

# Gradient Checking via Finite Differences
# Approximation: grad_approx[i] = (J(θ + ε*e_i) - J(θ - ε*e_i)) / (2ε)
# Difference:    ||grad - grad_approx|| / (||grad|| + ||grad_approx||)
#
# The numerical approximation of a derivative comes from the definition of the limit:
# perturbing each parameter by a tiny ε and measuring how much the cost changes gives
# you a ground truth gradient that doesn't depend on any backprop math. Comparing it
# to the analytic gradient produced by backprop is the standard sanity check — if the
# relative difference is below ~1e-7 the implementation is correct. Much larger means
# there is a bug in the backward pass.

import numpy as np


def gradient_check_1d(f, grad, theta, eps=1e-7):
    theta_plus  = theta + eps
    theta_minus = theta - eps
    approx = (f(theta_plus) - f(theta_minus)) / (2 * eps)
    diff = abs(grad - approx) / (abs(grad) + abs(approx) + 1e-12)
    return diff, approx


def dict_to_vector(params):
    keys = sorted(params.keys())
    flat = np.concatenate([params[k].ravel() for k in keys])
    shapes = {k: params[k].shape for k in keys}
    return flat, keys, shapes


def vector_to_dict(flat, keys, shapes):
    params = {}
    idx = 0
    for k in keys:
        size = int(np.prod(shapes[k]))
        params[k] = flat[idx:idx+size].reshape(shapes[k])
        idx += size
    return params


def gradient_check_nd(forward_fn, params, grads, eps=1e-7):
    flat_params, keys, shapes = dict_to_vector(params)
    flat_grads, _, _ = dict_to_vector(grads)
    n = len(flat_params)
    approx = np.zeros(n)

    for i in range(n):
        p_plus = flat_params.copy(); p_plus[i] += eps
        p_minus = flat_params.copy(); p_minus[i] -= eps
        J_plus  = forward_fn(vector_to_dict(p_plus,  keys, shapes))
        J_minus = forward_fn(vector_to_dict(p_minus, keys, shapes))
        approx[i] = (J_plus - J_minus) / (2 * eps)

    diff = np.linalg.norm(flat_grads - approx) / (np.linalg.norm(flat_grads) + np.linalg.norm(approx) + 1e-12)
    return diff, approx


if __name__ == "__main__":
    # 1D example: f(θ) = θ^3, f'(θ) = 3θ^2
    theta = 2.0
    diff, approx = gradient_check_1d(lambda t: t**3, grad=3*theta**2, theta=theta)
    print(f"1D: analytic={3*theta**2:.6f}  approx={approx:.6f}  diff={diff:.2e}")

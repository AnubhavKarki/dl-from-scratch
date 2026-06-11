# Mini-Batch Gradient Descent: shuffling and partitioning the training set
#
# Three gradient descent variants differ only in how much data each update sees:
#   Batch GD:     all m examples per update — low-variance gradient, expensive iteration
#   SGD:          1 example per update — high-variance gradient, cheap but noisy
#   Mini-Batch GD: b examples per update — practical balance used in virtually all deep learning
#
# Shuffle: random permutation applied synchronously to X and Y so labels stay aligned.
# Partition: slice every b columns; the last batch may be smaller if m % b != 0.
#
# Powers of 2 are the conventional choice (32, 64, 128, 256) to align with GPU memory layout.
# Reshuffling every epoch ensures different mini-batches across training iterations.

import numpy as np


def random_mini_batches(X, Y, batch_size=64, seed=0):
    np.random.seed(seed)
    m = X.shape[1]

    perm = np.random.permutation(m)
    X_s = X[:, perm]
    Y_s = Y[:, perm].reshape(1, m)

    batches = []
    n_full = m // batch_size
    for k in range(n_full):
        batches.append((X_s[:, k * batch_size:(k + 1) * batch_size],
                        Y_s[:, k * batch_size:(k + 1) * batch_size]))
    if m % batch_size != 0:
        batches.append((X_s[:, n_full * batch_size:],
                        Y_s[:, n_full * batch_size:]))
    return batches


if __name__ == "__main__":
    np.random.seed(1)
    X = np.random.randn(12288, 148)
    Y = (np.random.randn(1, 148) > 0).astype(float)

    batches = random_mini_batches(X, Y, batch_size=64, seed=0)
    print(f"total batches: {len(batches)}")
    print(f"batch 0   X: {batches[0][0].shape}  Y: {batches[0][1].shape}")
    print(f"last batch X: {batches[-1][0].shape}  Y: {batches[-1][1].shape}")

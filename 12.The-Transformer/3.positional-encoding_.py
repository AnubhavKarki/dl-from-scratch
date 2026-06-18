# Sinusoidal Positional Encoding (Vaswani et al., 2017 — Attention Is All You Need)
#
# PE(pos, 2i)   = sin( pos / 10000^(2i/d) )
# PE(pos, 2i+1) = cos( pos / 10000^(2i/d) )
#
# pos: position of the token in the sequence
# i:   index into the encoding dimension (i = k // 2 for dimension k)
# d:   total encoding dimension (must match the embedding size)
#
# Alternating sin/cos across paired dimensions gives each position a unique signature.
# Values stay in [-1, 1] so they do not distort word embeddings when added.
# Row norms are constant (~sqrt(d/2)) regardless of position, so no position dominates by scale.
# Linear relationships exist between encodings at different positions, letting the model
# attend by relative offset: PE(pos+k) can be expressed as a linear function of PE(pos).
#
# get_angles: shared angle computation reused by both sin (even k) and cos (odd k) columns.
# positional_encoding: returns (1, positions, d) so it broadcasts directly onto embeddings.

import numpy as np


def get_angles(pos, k, d):
    i = k // 2
    return pos / np.power(10000, (2 * i) / np.float32(d))


def positional_encoding(positions, d):
    angle_rads = get_angles(
        np.arange(positions)[:, np.newaxis],
        np.arange(d)[np.newaxis, :],
        d,
    )
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    return angle_rads[np.newaxis, ...]


if __name__ == "__main__":
    positions, d = 50, 64
    pe = positional_encoding(positions, d)
    print(f"shape: {pe.shape}")
    print(f"PE[pos=0, :4]: {pe[0, 0, :4].round(4)}")
    print(f"PE[pos=1, :4]: {pe[0, 1, :4].round(4)}")

    norms = np.linalg.norm(pe[0], axis=-1)
    print(f"row norms (should be constant): {norms[:5].round(4)}")

    corr = (pe @ pe.swapaxes(-1, -2))[0]
    diagonal_max = all(corr[i, i] == corr[i].max() for i in range(positions))
    print(f"diagonal dominates correlation matrix: {diagonal_max}")

# Padding mask and look-ahead mask for Transformer attention (Vaswani et al., 2017)
#
# Padding mask: real tokens get 1, zero-padded positions get 0.
#   Applied before softmax as: scores += (1 - mask) * -1e9
#   Pushes padded positions to -inf so they contribute ~0 after softmax.
#
# Look-ahead (causal) mask: lower triangular matrix of 1s.
#   Position i can attend only to positions 0..i — prevents peeking at future tokens.
#   Required in the decoder's self-attention to preserve the autoregressive property.
#
# Padding mask shape:  (batch, 1, seq_len) — broadcasts over query positions
# Look-ahead mask shape: (1, seq_len, seq_len) — one mask shared across the batch
#
# attention(Q, K, V, mask):
#   scores   = Q @ K^T / sqrt(d_k)         scaled similarity (batch, n_q, n_k)
#   scores  += (1 - mask) * -1e9           zero out masked positions
#   weights  = softmax(scores, axis=-1)     (batch, n_q, n_k)
#   output   = weights @ V                  (batch, n_q, d_v)

import importlib.util
import pathlib
import numpy as np

_spec = importlib.util.spec_from_file_location(
    "scaled_attention", pathlib.Path(__file__).parent / "1.scaled-dot-product-attention_.py"
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
softmax = _mod.softmax


def create_padding_mask(token_ids):
    return (token_ids != 0).astype(float)[:, np.newaxis, :]


def create_look_ahead_mask(n):
    return np.tril(np.ones((1, n, n)))


def attention(Q, K, V, mask=None, scale=True):
    scores = Q @ K.swapaxes(-1, -2)
    if scale:
        scores /= np.sqrt(K.shape[-1])
    if mask is not None:
        scores += (1.0 - mask) * -1e9
    weights = softmax(scores, axis=-1)
    return weights @ V, weights


def causal_attention(Q, K, V, scale=True):
    mask = create_look_ahead_mask(Q.shape[-2])
    return attention(Q, K, V, mask, scale)


if __name__ == "__main__":
    np.random.seed(42)
    seq_len, d_k = 4, 8

    Q = np.random.randn(1, seq_len, d_k)
    K = np.random.randn(1, seq_len, d_k)
    V = np.random.randn(1, seq_len, d_k)

    out, weights = causal_attention(Q, K, V)
    print(f"output shape: {out.shape}")
    print(f"weights (upper triangle should be ~0):\n{weights[0].round(3)}")

    tokens = np.array([[7.0, 6.0, 0.0, 0.0, 0.0]])
    print(f"\npadding mask for [7, 6, 0, 0, 0]: {create_padding_mask(tokens)}")
    print(f"look-ahead mask (4x4):\n{create_look_ahead_mask(4)[0]}")

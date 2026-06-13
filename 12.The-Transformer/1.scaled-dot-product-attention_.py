# Scaled Dot-Product Attention (Vaswani et al., 2017 — Attention Is All You Need)
#
# Attention(Q, K, V) = softmax( Q K^T / sqrt(d_k) ) V
#
# Q: queries  (n_q x d_k)
# K: keys     (n_k x d_k)
# V: values   (n_k x d_v)
#
# Step 1 — similarity scores: Q @ K.T gives an (n_q x n_k) matrix where entry [i,j]
#          measures how much query i attends to key j.
# Step 2 — scaling by sqrt(d_k): dot products grow in magnitude with dimension, pushing
#          softmax into saturation. Dividing by sqrt(d_k) keeps gradients well-behaved.
# Step 3 — softmax row-wise: each row of scores becomes a probability distribution over
#          the n_k key positions — the attention weights.
# Step 4 — weighted sum: weights @ V produces one output vector per query, blending
#          the value vectors according to how much each key was attended to.
#
# This is faster than Bahdanau additive attention because it uses only matrix multiplications
# with no learned alignment network — the similarity measure is the scaled dot product itself.

import numpy as np


def softmax(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True) 


def attention_qkv(Q, K, V):
    d_k = K.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)
    weights = softmax(scores, axis=-1)
    return weights @ V, weights


if __name__ == "__main__":
    np.random.seed(42)
    n_q, n_k, d_k, d_v = 5, 7, 64, 64

    Q = np.random.randn(n_q, d_k)
    K = np.random.randn(n_k, d_k)
    V = np.random.randn(n_k, d_v)

    out, weights = attention_qkv(Q, K, V)
    print(f"Q: {Q.shape}  K: {K.shape}  V: {V.shape}")
    print(f"output:  {out.shape}  mean: {out.mean():.4f}")
    print(f"weights: {weights.shape}  each row sums to 1: {np.allclose(weights.sum(axis=1), 1)}")
    print(f"weights[0] (query 0 over {n_k} keys): {weights[0].round(3)}")

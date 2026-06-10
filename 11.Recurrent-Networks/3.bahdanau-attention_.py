# Bahdanau Attention: soft alignment over encoder states
# Paper: Bahdanau et al., 2014 — Neural Machine Translation by Jointly Learning to Align and Translate
#
# Alignment score: e_ij = v_a^T * tanh(W_a * s_{i-1} + U_a * h_j)
# Implemented as a 2-layer feedforward network over the concatenated [h_j ; s_{i-1}]:
#   activations = tanh(concat(h_j, s_{i-1}) @ W1)    shape: (K, attention_size)
#   scores      = activations @ W2                    shape: (K, 1)
#
# Attention weights: alpha_ij = softmax(e_ij) over all K encoder positions
# Context vector:    c_i = sum_j( alpha_ij * h_j )   shape: (hidden_size,)
#
# The decoder receives this context vector at each step instead of a single fixed
# encoder summary, letting the model learn which input positions matter for each output.

import numpy as np


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def alignment_scores(encoder_states, decoder_state, W1, W2):
    K = encoder_states.shape[0]
    combined = np.concatenate([encoder_states, np.repeat(decoder_state, K, axis=0)], axis=1)
    activations = np.tanh(combined @ W1)
    return activations @ W2


def attention(encoder_states, decoder_state, W1, W2):
    scores = alignment_scores(encoder_states, decoder_state, W1, W2)
    weights = softmax(scores)
    return (weights * encoder_states).sum(axis=0)


if __name__ == "__main__":
    np.random.seed(42)
    hidden_size = 16
    attention_size = 10
    K = 5

    encoder_states = np.random.randn(K, hidden_size)
    decoder_state = np.random.randn(1, hidden_size)
    W1 = np.random.randn(2 * hidden_size, attention_size)
    W2 = np.random.randn(attention_size, 1)

    scores = alignment_scores(encoder_states, decoder_state, W1, W2)
    print(f"alignment scores: {scores.ravel()}")

    ctx = attention(encoder_states, decoder_state, W1, W2)
    print(f"context vector:   {ctx}")

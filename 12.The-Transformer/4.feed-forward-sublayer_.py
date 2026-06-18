# Feed-Forward Sub-layer and Layer Normalization (Vaswani et al., 2017)
#
# FFN(x) = relu(x @ W1 + b1) @ W2 + b2
#
# Applied position-wise (independently to each token vector, same weights across positions).
# d_ff is typically 4 * d_model — expands then contracts to let the network learn richer features.
# Originally uses ReLU; later architectures (BERT, GPT) switch to GELU or SiLU.
#
# Layer Normalization: normalises over the feature dimension (not the batch dimension).
#   LayerNorm(x) = gamma * (x - mean(x)) / sqrt(var(x) + eps) + beta
#   gamma and beta are learned scale and shift; eps prevents division by zero.
#   Applied after the residual connection: output = LayerNorm(x + sublayer(x))
#
# Every sub-layer in the Transformer (attention and FFN alike) wraps in:
#   output = LayerNorm(x + sublayer(x))
# This keeps training stable in deep stacks and makes each sub-layer learn a residual.

import importlib.util
import pathlib
import numpy as np

_spec = importlib.util.spec_from_file_location(
    "relu_mod",
    pathlib.Path(__file__).parents[1] / "6.Activation-Functions" / "4.relu_.py",
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
relu = _mod.relu


def layer_norm(x, gamma, beta, eps=1e-6):
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return gamma * (x - mean) / np.sqrt(var + eps) + beta


def feed_forward(x, W1, b1, W2, b2):
    return relu(x @ W1 + b1) @ W2 + b2


def init_feed_forward(d_model, d_ff, seed=0):
    rng = np.random.default_rng(seed)
    W1 = rng.standard_normal((d_model, d_ff)) * np.sqrt(2 / d_model)
    b1 = np.zeros(d_ff)
    W2 = rng.standard_normal((d_ff, d_model)) * np.sqrt(2 / d_ff)
    b2 = np.zeros(d_model)
    gamma = np.ones(d_model)
    beta = np.zeros(d_model)
    return W1, b1, W2, b2, gamma, beta


if __name__ == "__main__":
    batch, seq_len, d_model, d_ff = 2, 5, 64, 256
    W1, b1, W2, b2, gamma, beta = init_feed_forward(d_model, d_ff, seed=42)

    x = np.random.default_rng(42).standard_normal((batch, seq_len, d_model))
    ffn_out = feed_forward(x, W1, b1, W2, b2)
    print(f"input:  {x.shape}")
    print(f"output: {ffn_out.shape}")

    x_normed = layer_norm(x + ffn_out, gamma, beta)
    print(f"after residual + LayerNorm: {x_normed.shape}")
    print(f"mean per token (should be ~0): {x_normed.mean(axis=-1).round(4)}")
    print(f"std  per token (should be ~1): {x_normed.std(axis=-1).round(4)}")

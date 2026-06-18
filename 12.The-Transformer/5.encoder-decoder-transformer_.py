# Full Transformer: Encoder + Decoder (Vaswani et al., 2017 — Attention Is All You Need)
# Implemented with TensorFlow Keras — uses Keras MultiHeadAttention.
# The scratch NumPy multi-head attention (Phase 12 item 2) will replace this once built.
#
# Encoder block (repeated N times):
#   1. Multi-head self-attention (all queries, keys, values from the same sequence)
#   2. Residual + LayerNorm
#   3. Feed-forward (two dense layers, ReLU, same weights at each position)
#   4. Residual + LayerNorm
#
# Decoder block (repeated N times):
#   1. Masked multi-head self-attention (look-ahead mask prevents attending to future tokens)
#   2. Residual + LayerNorm
#   3. Cross-attention (Q from decoder, K/V from encoder output)
#   4. Residual + LayerNorm
#   5. Feed-forward
#   6. Residual + LayerNorm
#
# Full model: Encoder processes the source sequence, Decoder generates the target sequence
# token by token using the encoder output as context via cross-attention.
# Final dense layer + softmax produces a probability over the target vocabulary.

import importlib.util
import pathlib
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Embedding, LayerNormalization, MultiHeadAttention

_here = pathlib.Path(__file__).parent

_pe_spec = importlib.util.spec_from_file_location("pe_mod", _here / "3.positional-encoding_.py")
_pe_mod = importlib.util.module_from_spec(_pe_spec)
_pe_spec.loader.exec_module(_pe_mod)
positional_encoding = _pe_mod.positional_encoding

_mask_spec = importlib.util.spec_from_file_location("mask_mod", _here / "2.causal-masking_.py")
_mask_mod = importlib.util.module_from_spec(_mask_spec)
_mask_spec.loader.exec_module(_mask_mod)
create_padding_mask = _mask_mod.create_padding_mask
create_look_ahead_mask = _mask_mod.create_look_ahead_mask


def _pe_tf(positions, d):
    return tf.cast(positional_encoding(positions, d), tf.float32)


def FullyConnected(embedding_dim, fully_connected_dim):
    return tf.keras.Sequential([
        Dense(fully_connected_dim, activation="relu"),
        Dense(embedding_dim),
    ])


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, embedding_dim, num_heads, fully_connected_dim,
                 dropout_rate=0.1, layernorm_eps=1e-6):
        super().__init__()
        self.mha = MultiHeadAttention(
            num_heads=num_heads, key_dim=embedding_dim, dropout=dropout_rate
        )
        self.ffn = FullyConnected(embedding_dim, fully_connected_dim)
        self.layernorm1 = LayerNormalization(epsilon=layernorm_eps)
        self.layernorm2 = LayerNormalization(epsilon=layernorm_eps)
        self.dropout_ffn = Dropout(dropout_rate)

    def call(self, x, training, mask):
        attn_out = self.mha(query=x, key=x, value=x, attention_mask=mask, training=training)
        x = self.layernorm1(x + attn_out)
        ffn_out = self.dropout_ffn(self.ffn(x), training=training)
        return self.layernorm2(x + ffn_out)


class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, embedding_dim, num_heads, fully_connected_dim,
                 input_vocab_size, maximum_position_encoding,
                 dropout_rate=0.1, layernorm_eps=1e-6):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.embedding = Embedding(input_vocab_size, embedding_dim)
        self.pos_encoding = _pe_tf(maximum_position_encoding, embedding_dim)
        self.enc_layers = [
            EncoderLayer(embedding_dim, num_heads, fully_connected_dim,
                         dropout_rate, layernorm_eps)
            for _ in range(num_layers)
        ]
        self.dropout = Dropout(dropout_rate)

    def call(self, x, training, mask):
        seq_len = tf.shape(x)[1]
        x = self.embedding(x) * tf.math.sqrt(tf.cast(self.embedding_dim, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)
        for layer in self.enc_layers:
            x = layer(x, training, mask)
        return x


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, embedding_dim, num_heads, fully_connected_dim,
                 dropout_rate=0.1, layernorm_eps=1e-6):
        super().__init__()
        self.mha1 = MultiHeadAttention(
            num_heads=num_heads, key_dim=embedding_dim, dropout=dropout_rate
        )
        self.mha2 = MultiHeadAttention(
            num_heads=num_heads, key_dim=embedding_dim, dropout=dropout_rate
        )
        self.ffn = FullyConnected(embedding_dim, fully_connected_dim)
        self.layernorm1 = LayerNormalization(epsilon=layernorm_eps)
        self.layernorm2 = LayerNormalization(epsilon=layernorm_eps)
        self.layernorm3 = LayerNormalization(epsilon=layernorm_eps)
        self.dropout_ffn = Dropout(dropout_rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        attn1, w1 = self.mha1(
            query=x, key=x, value=x,
            attention_mask=look_ahead_mask, return_attention_scores=True,
        )
        q1 = self.layernorm1(x + attn1)
        attn2, w2 = self.mha2(
            query=q1, key=enc_output, value=enc_output,
            attention_mask=padding_mask, return_attention_scores=True,
        )
        q2 = self.layernorm2(q1 + attn2)
        ffn_out = self.dropout_ffn(self.ffn(q2), training=training)
        return self.layernorm3(q2 + ffn_out), w1, w2


class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, embedding_dim, num_heads, fully_connected_dim,
                 target_vocab_size, maximum_position_encoding,
                 dropout_rate=0.1, layernorm_eps=1e-6):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.embedding = Embedding(target_vocab_size, embedding_dim)
        self.pos_encoding = _pe_tf(maximum_position_encoding, embedding_dim)
        self.dec_layers = [
            DecoderLayer(embedding_dim, num_heads, fully_connected_dim,
                         dropout_rate, layernorm_eps)
            for _ in range(num_layers)
        ]
        self.dropout = Dropout(dropout_rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        seq_len = tf.shape(x)[1]
        attention_weights = {}
        x = self.embedding(x) * tf.math.sqrt(tf.cast(self.embedding_dim, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)
        for i, layer in enumerate(self.dec_layers):
            x, b1, b2 = layer(x, enc_output, training, look_ahead_mask, padding_mask)
            attention_weights[f"decoder_layer{i+1}_block1_self_att"] = b1
            attention_weights[f"decoder_layer{i+1}_block2_decenc_att"] = b2
        return x, attention_weights


class Transformer(tf.keras.Model):
    def __init__(self, num_layers, embedding_dim, num_heads, fully_connected_dim,
                 input_vocab_size, target_vocab_size,
                 max_positional_encoding_input, max_positional_encoding_target,
                 dropout_rate=0.1, layernorm_eps=1e-6):
        super().__init__()
        self.encoder = Encoder(
            num_layers, embedding_dim, num_heads, fully_connected_dim,
            input_vocab_size, max_positional_encoding_input, dropout_rate, layernorm_eps,
        )
        self.decoder = Decoder(
            num_layers, embedding_dim, num_heads, fully_connected_dim,
            target_vocab_size, max_positional_encoding_target, dropout_rate, layernorm_eps,
        )
        self.final_layer = Dense(target_vocab_size, activation="softmax")

    def call(self, input_sentence, output_sentence, training,
             enc_padding_mask, look_ahead_mask, dec_padding_mask):
        enc_output = self.encoder(input_sentence, training, enc_padding_mask)
        dec_output, attention_weights = self.decoder(
            output_sentence, enc_output, training, look_ahead_mask, dec_padding_mask,
        )
        return self.final_layer(dec_output), attention_weights


if __name__ == "__main__":
    model = Transformer(
        num_layers=2, embedding_dim=32, num_heads=2, fully_connected_dim=64,
        input_vocab_size=100, target_vocab_size=80,
        max_positional_encoding_input=50, max_positional_encoding_target=50,
    )

    inp = tf.constant(np.random.randint(1, 100, (2, 10)))
    tar = tf.constant(np.random.randint(1, 80, (2, 8)))

    enc_mask = tf.cast(create_padding_mask(inp.numpy()), tf.float32)
    look_mask = tf.cast(create_look_ahead_mask(tar.shape[1]), tf.float32)
    dec_mask = tf.cast(create_padding_mask(inp.numpy()), tf.float32)

    output, _ = model(inp, tar, training=False,
                      enc_padding_mask=enc_mask,
                      look_ahead_mask=look_mask,
                      dec_padding_mask=dec_mask)
    print(f"output shape: {output.shape}")
    print(f"each position sums to 1: {np.allclose(output.numpy().sum(-1), 1.0, atol=1e-5)}")

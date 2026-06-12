# Deep Learning from Scratch

Every forward pass. Every gradient. Every architecture. Written by hand.

---

This repository is my personal deep learning curriculum, implemented entirely from scratch in NumPy after reading the original papers. No PyTorch, no TensorFlow, no abstracted magic. Every matrix multiply, every chain-rule gradient, and every architectural decision is written and understood before it is committed.

A handful of entries are marked **Library** to demonstrate how the concept maps onto production APIs (PyTorch, Hugging Face) after having built it manually first. The implementation always comes before the library version.

> This repo is strictly Deep Learning: networks, gradients, and architectures, from a single neuron to Transformers and GANs.
> For classical ML (SVMs, trees, boosting) see [ml-from-scratch](#).
> For Reinforcement Learning (Q-learning, PPO, DDPG) see [rl-from-scratch](#).

---

## Navigation

The curriculum is strictly sequential. Each phase builds on the last. Activation functions, optimizers, and loss functions are introduced exactly where they are first needed, not front-loaded as abstract theory. Follow top to bottom. Jump into any phase if you already have the prerequisites.

---

## Legend

| Label | Meaning |
|-------|---------|
| Done | Implemented |
| Pending | Not yet started |
| Scratch | NumPy only, no frameworks |
| Library | Library-assisted: PyTorch or Hugging Face, for API fluency after the scratch version |

---

## The Curriculum

---

### Phase 1: Mathematical Foundations

The minimum math needed to understand everything that follows. Covered as notation and intuition, not formal proof. Only what gets used downstream.

| # | Topic | Type | Status |
|---|-------|------|--------|
| 1 | Scalars, vectors, matrices: shapes, indexing, broadcasting | Scratch | Pending |
| 2 | Dot product, matrix multiplication as linear maps | Scratch | Pending |
| 3 | Transpose, inverse, identity | Scratch | Pending |
| 4 | Norms: L1, L2, Frobenius | Scratch | Pending |
| 5 | Chain rule and partial derivatives | Scratch | Pending |
| 6 | Jacobians and the gradient as a vector | Scratch | Pending |
| 7 | Probability basics: distributions, expectation, MLE | Scratch | Pending |
| 8 | Information theory: entropy, cross-entropy, KL divergence | Scratch | Pending |

---

### Phase 2: The Neuron

Where everything begins. A single computation unit traced scalar by scalar.

| # | Concept | Type | Status |
|---|---------|------|--------|
| 1 | Single linear neuron: forward pass, weights, bias | Scratch | Done |
| 2 | Mean Squared Error loss: derivation from first principles | Scratch | Done |
| 3 | Manual gradient computation, scalar, step-by-step | Scratch | Done |
| 4 | Vanilla SGD weight update | Scratch | Done |
| 5 | Activation: Sigmoid, formula, derivative, saturation behaviour | Scratch | Done |
| 6 | Two-layer network with sigmoid and chain-rule backprop | Scratch | Done |
| 7 | Activation: Tanh, formula, derivative, comparison to sigmoid in hidden layers | Scratch | Done |
| 8 | Bias terms: role in shifting decision boundaries | Scratch | Pending |

---

### Phase 3: Backpropagation

The algorithm that makes training possible, derived from first principles then lifted from scalars to full matrix form.

| # | Concept | Type | Status |
|---|---------|------|--------|
| 1 | Manual chain rule: scalar, step-by-step trace | Scratch | Done |
| 2 | Computational graph: nodes, edges, local gradients | Scratch | Pending |
| 3 | Vectorized backprop: full matrix form, no loops | Scratch | Done |
| 4 | Shape sanity checks at every layer | Scratch | Pending |
| 5 | Loss decreasing sanity test | Scratch | Pending |

---

### Phase 4: Training Mechanics

How gradients move through data in practice. Batching strategies, epoch management, and learning rate behaviour before adaptive optimizers are introduced.

| # | Concept | Type | Status |
|---|---------|------|--------|
| 1 | Batch Gradient Descent | Scratch | Done |
| 2 | Stochastic Gradient Descent: online, one sample at a time | Scratch | Done |
| 3 | Mini-Batch Gradient Descent | Scratch | Done |
| 4 | Dataset shuffling per epoch and why it matters | Scratch | Done |
| 5 | Gradient averaging vs summing: the math difference | Scratch | Pending |
| 6 | Learning rate scheduling: step decay, exponential decay | Scratch | Done |
| 7 | Warmup and cosine annealing | Scratch | Pending |

---

### Phase 5: Optimizers

The full optimizer lineage, from raw SGD to modern adaptive methods. Each implemented from scratch against the original paper, then benchmarked on the same task.

| # | Optimizer | Paper | Type | Status |
|---|-----------|-------|------|--------|
| 1 | Vanilla SGD | | Scratch | Done |
| 2 | SGD with Momentum | Polyak, 1964 | Scratch | Done |
| 3 | Nesterov Accelerated Gradient | Nesterov, 1983 | Scratch | Pending |
| 4 | Adagrad: adaptive per-parameter learning rate | Duchi et al., 2011 | Scratch | Pending |
| 5 | RMSProp: leaky Adagrad | Hinton, unpublished 2012 | Scratch | Pending |
| 6 | Adam: adaptive moments | Kingma and Ba, 2014 | Scratch | Done |
| 7 | AdamW: decoupled weight decay | Loshchilov and Hutter, 2017 | Scratch | Pending |
| 8 | Nadam: Adam with Nesterov momentum | Dozat, 2016 | Scratch | Pending |
| 9 | Optimizer comparison on identical task, convergence curves | | Scratch | Pending |

---

### Phase 6: Activation Functions Reference

Introduced throughout the curriculum exactly where each is first needed. Collected here as a reference. Each entry covers the forward pass, exact derivative, numerical considerations, and the phase it first appears in.

| # | Activation | First Used | Derivative | Vanishing Gradient Risk | Status |
|---|-----------|-----------|-----------|------------------------|--------|
| 1 | Sigmoid | Phase 2 | s(x)(1-s(x)) | High | Done |
| 2 | Tanh | Phase 2 | 1-tanh^2(x) | Moderate | Done |
| 3 | ReLU | Phase 9 | 1 if x > 0, else 0 | Low | Done |
| 4 | Leaky ReLU | Phase 9 | alpha if x <= 0, else 1 | Very Low | Pending |
| 5 | ELU | Phase 9 | alpha*e^x if x < 0 | Very Low | Pending |
| 6 | Softmax (numerically stable) | Phase 8 | Jacobian form | None | Done |
| 7 | GELU | Phase 11 | Approximated | Very Low | Pending |
| 8 | Swish / SiLU | Phase 11 | x*s(x) + s(x)(1 - x*s(x)) | Very Low | Pending |
| 9 | Activation comparison: vanishing gradient demo across all | | | | Pending |

---

### Phase 7: Loss Functions

Each loss function derived from probability theory where applicable. Gradients worked out by hand before implementation. Combined gradients such as Softmax and Cross-Entropy derived as a single efficient expression.

| # | Loss Function | Use Case | Type | Status |
|---|--------------|----------|------|--------|
| 1 | Mean Squared Error | Regression | Scratch | Done |
| 2 | Mean Absolute Error | Robust regression | Scratch | Pending |
| 3 | Huber Loss | Regression with outliers | Scratch | Pending |
| 4 | Binary Cross-Entropy | Binary classification | Scratch | Done |
| 5 | Categorical Cross-Entropy | Multi-class classification | Scratch | Pending |
| 6 | Softmax and Cross-Entropy combined gradient, single efficient backward pass | Multi-class | Scratch | Done |

---

### Phase 8: Regularization and Training Stability

Techniques that make deep networks trainable. Applied from dense networks onward and carried through every subsequent architecture.

| # | Technique | Type | Status |
|---|-----------|------|--------|
| 1 | L1 Regularization: sparse weights | Scratch | Done |
| 2 | L2 Weight Decay: penalize large weights | Scratch | Done |
| 3 | Dropout: inverted dropout, training vs inference behaviour | Scratch | Done |
| 4 | Gradient Clipping: norm clipping and value clipping | Scratch | Pending |
| 5 | Xavier and Glorot Initialization: for sigmoid and tanh | Scratch | Done |
| 6 | He Initialization: for ReLU and variants | Scratch | Done |
| 7 | Batch Normalization: forward pass and full backward derivation | Scratch | Pending |
| 8 | Layer Normalization: used in Transformers and sequence models | Scratch | Pending |

---

### Phase 9: Fully Connected Networks and Classical Tasks

Dense layer abstraction built. First end-to-end training runs on real tasks. Modular structure introduced here and carried forward through all subsequent phases.

Activations active in this phase: Sigmoid, Tanh, ReLU, Softmax

| # | Concept | Type | Status |
|---|---------|------|--------|
| 1 | Dense layer class: forward and backward | Scratch | Pending |
| 2 | Parameter dictionaries and state management | Scratch | Pending |
| 3 | Deep MLP with configurable depth and width | Scratch | Done |
| 4 | Regression: synthetic data, linear and nonlinear | Scratch | Pending |
| 5 | Binary classification: circles and moons datasets | Scratch | Done |
| 6 | Multi-class classification: Softmax output layer | Scratch | Pending |
| 7 | Train, validation, and test split | Scratch | Pending |
| 8 | Overfitting a tiny dataset: sanity check | Scratch | Pending |
| 9 | Metrics: accuracy, precision, recall, F1 | Scratch | Pending |
| 10 | Loss and accuracy curve plotting | Scratch | Pending |
| 11 | MNIST classification from scratch: full training run | Scratch | Pending |

---

### Phase 10: Convolutional Neural Networks

The architecture that made deep learning viable for vision. Built operation by operation starting from the raw convolution kernel, through pooling, skip connections, and ending with a ResNet-style block.

Activations introduced in this phase: ReLU (dominant), Leaky ReLU

| # | Concept | Type | Status |
|---|---------|------|--------|
| 1 | 2D convolution: naive loop implementation | Scratch | Done |
| 2 | Convolution: vectorized via im2col | Scratch | Pending |
| 3 | Padding and stride: output size derivation | Scratch | Done |
| 4 | Backprop through convolution: gradient w.r.t. input and kernel | Scratch | Done |
| 5 | Max pooling: forward pass and backward | Scratch | Done |
| 6 | Average pooling | Scratch | Done |
| 7 | Flatten layer | Scratch | Pending |
| 8 | Full CNN: conv, pool, flatten, dense | Scratch | Pending |
| 9 | LeNet-5 style architecture | Scratch | Pending |
| 10 | Batch Normalization inside CNNs | Scratch | Pending |
| 11 | Residual skip connections: ResNet block from scratch | Scratch | Pending |
| 12 | AlexNet and VGG-style depth and width experiment | Scratch | Pending |
| 13 | CNN on MNIST and CIFAR-10: full training run | Scratch | Pending |
| 14 | PyTorch CNN: same architecture for reference | Library | Pending |

---

### Phase 11: Recurrent Networks and Sequence Modeling

Sequential data requires architectures that carry state. Built in chronological order of invention: vanilla RNN through BiLSTM, then attention over sequences before the full Transformer.

Activations dominant in this phase: Tanh (state), Sigmoid (gates)

| # | Architecture / Concept | Type | Status |
|---|----------------------|------|--------|
| 1 | Vanilla RNN: forward pass, hidden state | Scratch | Done |
| 2 | Backpropagation Through Time (BPTT) | Scratch | Done |
| 3 | Vanishing and exploding gradients in RNNs: demonstration | Scratch | Pending |
| 4 | Gradient clipping for RNNs | Scratch | Pending |
| 5 | Long Short-Term Memory (LSTM): forget, input, output, cell gates | Scratch | Done |
| 6 | LSTM BPTT: gradient through all four gates | Scratch | Done |
| 7 | Gated Recurrent Unit (GRU): reset and update gates | Scratch | Pending |
| 8 | Stacked and Deep RNNs | Scratch | Pending |
| 9 | Bidirectional RNN: forward and backward pass over sequence | Scratch | Pending |
| 10 | Bidirectional LSTM | Scratch | Pending |
| 11 | Sequence-to-sequence: encoder-decoder with RNN | Scratch | Pending |
| 12 | Bahdanau Attention: soft alignment over encoder states | Scratch | Done |
| 13 | Attention with LSTM decoder | Scratch | Pending |

---

### Phase 12: The Transformer

The architecture that reshaped the field. Implemented component by component from Attention Is All You Need, then extended to GPT and BERT variants.

Activations introduced in this phase: GELU, Swish / SiLU

| # | Component | Paper | Type | Status |
|---|----------|-------|------|--------|
| 1 | Scaled dot-product attention: Q, K, V from scratch | Vaswani et al., 2017 | Scratch | Done |
| 2 | Multi-head attention: projection, split, concat | Vaswani et al., 2017 | Scratch | Pending |
| 3 | Causal masking for decoder and autoregressive models | Vaswani et al., 2017 | Scratch | Pending |
| 4 | Sinusoidal positional encoding: derivation and implementation | Vaswani et al., 2017 | Scratch | Pending |
| 5 | Learned positional embeddings | | Scratch | Pending |
| 6 | Feed-forward sublayer: two linear layers with GELU | | Scratch | Pending |
| 7 | Encoder block: multi-head attention, FFN, LayerNorm, residual | Vaswani et al., 2017 | Scratch | Pending |
| 8 | Decoder block: masked self-attention, cross-attention, FFN | Vaswani et al., 2017 | Scratch | Pending |
| 9 | Full Transformer: encoder-decoder, translation task | Vaswani et al., 2017 | Scratch | Pending |
| 10 | GPT-style decoder-only Transformer: causal language modelling | Radford et al., 2018 | Scratch | Pending |
| 11 | BERT-style encoder-only Transformer: masked language modelling | Devlin et al., 2018 | Scratch | Pending |
| 12 | Byte-Pair Encoding tokenizer from scratch | Sennrich et al., 2016 | Scratch | Pending |
| 13 | Fine-tuning a pretrained BERT (Hugging Face): map concepts to API | | Library | Pending |

---

### Phase 13: Generative Models

Networks that learn the structure of data well enough to generate new samples. Autoencoders as warm-up, then VAEs, the full GAN family, and an introduction to diffusion.

Activations dominant in this phase: Leaky ReLU (GAN discriminator), Sigmoid (output), Tanh (generator output)

| # | Architecture | Paper | Type | Status |
|---|-------------|-------|------|--------|
| 1 | Autoencoder: encoder, bottleneck, decoder | | Scratch | Pending |
| 2 | Denoising Autoencoder | Vincent et al., 2008 | Scratch | Pending |
| 3 | Variational Autoencoder (VAE): reparameterisation trick, ELBO loss | Kingma and Welling, 2013 | Scratch | Pending |
| 4 | GAN: original formulation, minimax game, JS divergence | Goodfellow et al., 2014 | Scratch | Pending |
| 5 | DCGAN: convolutional generator and discriminator | Radford et al., 2015 | Scratch | Pending |
| 6 | Conditional GAN (cGAN): class-conditioned generation | Mirza and Osindero, 2014 | Scratch | Pending |
| 7 | Training instability: mode collapse, Wasserstein distance, gradient penalty | Arjovsky et al., 2017 | Scratch | Pending |
| 8 | Diffusion Models (DDPM): forward noising and reverse denoising | Ho et al., 2020 | Library | Pending |

---

## Approach

Every Scratch implementation follows the same process:

1. Read the original paper
2. Derive the forward pass by hand on paper
3. Derive all gradients by hand
4. Implement in NumPy

This is a reference built for myself and shared for anyone who wants to see how it is actually done.

---

## Prerequisites

```
Python 3.8+
numpy          pip install numpy
matplotlib     pip install matplotlib
```

Library entries additionally require:

```
torch          pip install torch
transformers   pip install transformers
```

Everything marked Scratch runs on pure NumPy with no other dependencies.

---

*Classical ML (SVMs, ensembles, boosting) -> [ml-from-scratch](#)*
*Reinforcement Learning (Q-learning, policy gradients, PPO) -> [rl-from-scratch](#)*

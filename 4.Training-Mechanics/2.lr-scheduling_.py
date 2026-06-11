# Learning Rate Scheduling: exponential decay and fixed interval step decay
#
# Exponential decay (per epoch):
#   alpha = alpha_0 / (1 + decay_rate * epoch)
#   Shrinks the learning rate on every epoch. Decays too aggressively for long runs.
#
# Step decay (fixed interval):
#   alpha = alpha_0 / (1 + decay_rate * floor(epoch / interval))
#   Holds alpha constant for `interval` epochs then drops it — staircase shape.
#   Prevents the rate from vanishing before the model has converged.
#
# Both implement the same core intuition: take large steps early (explore the loss
# landscape) and small steps late (settle into a minimum). The difference is the
# schedule shape — smooth vs staircase.

import numpy as np


def exponential_decay(lr0, epoch, decay_rate):
    return lr0 / (1 + decay_rate * epoch)


def step_decay(lr0, epoch, decay_rate, interval=1000):
    return lr0 / (1 + decay_rate * np.floor(epoch / interval))


if __name__ == "__main__":
    lr0 = 0.1
    for ep in [0, 100, 500, 1000, 2000, 5000]:
        exp  = exponential_decay(lr0, ep, decay_rate=0.3)
        step = step_decay(lr0, ep, decay_rate=0.3, interval=1000)
        print(f"epoch {ep:5d}  exp: {exp:.6f}  step: {step:.6f}")

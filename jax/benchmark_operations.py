import flax
import flax.linen as nn

import jax
import jax.numpy as jnp

LAYER_OPS = {
    nn.Dense,
    nn.Conv,
    nn.LSTMCell,
    nn.MultiHeadDotProductAttention,
    nn.SelfAttention,
}

NORM_OPS = {nn.BatchNorm, nn.LayerNorm, nn.GroupNorm}

ACTIVATION_OPS = {}

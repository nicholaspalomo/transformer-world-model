"""Flax NNX Causal Self-Attention module for sequence modeling."""

import jax
import jax.numpy as jnp
from flax import nnx


class CausalSelfAttention(nnx.Module):
    """Causal Multi-Head Self-Attention with strict lower-triangular mask."""

    def __init__(
        self, embed_dim: int, num_heads: int, dropout_rate: float = 0.0, *, rngs: nnx.Rngs
    ):
        msg = f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        assert embed_dim % num_heads == 0, msg
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv_proj = nnx.Linear(embed_dim, 3 * embed_dim, rngs=rngs)
        self.out_proj = nnx.Linear(embed_dim, embed_dim, rngs=rngs)

    def __call__(self, x: jax.Array, mask: jax.Array = None) -> jax.Array:
        """Forward pass.

        Input:
            x: [Batch, Time, Embed_Dim]
        Output:
            attn_output: [Batch, Time, Embed_Dim]
        """
        batch_size, seq_len, embed_dim = x.shape

        # Compute Q, K, V
        qkv = self.qkv_proj(x)  # [B, T, 3 * E]
        qkv = jnp.reshape(qkv, (batch_size, seq_len, 3, self.num_heads, self.head_dim))
        q, k, v = jnp.split(qkv, 3, axis=2)
        # Query
        q = jnp.squeeze(q, axis=2).swapaxes(1, 2)  # [B, H, T, Head_Dim]
        # Key
        k = jnp.squeeze(k, axis=2).swapaxes(1, 2)  # [B, H, T, Head_Dim]
        # Value
        v = jnp.squeeze(v, axis=2).swapaxes(1, 2)  # [B, H, T, Head_Dim]

        # Scaled dot-product attention
        scores = jnp.matmul(q, k.swapaxes(-2, -1)) / jnp.sqrt(self.head_dim)  # [B, H, T, T]

        # Apply causal mask (lower triangular)
        causal_mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))
        causal_mask = jnp.reshape(causal_mask, (1, 1, seq_len, seq_len))
        scores = jnp.where(causal_mask, scores, -1e9)

        if mask is not None:
            scores = jnp.where(mask, scores, -1e9)

        weights = jax.nn.softmax(scores, axis=-1)
        output = jnp.matmul(weights, v)  # [B, H, T, Head_Dim]
        output = output.swapaxes(1, 2).reshape(batch_size, seq_len, embed_dim)

        return self.out_proj(output)

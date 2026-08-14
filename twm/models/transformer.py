"""Transformer World Model architecture predicting dynamics auto-regressively."""

from typing import Tuple
import jax
import jax.numpy as jnp
from flax import nnx

from twm.envs.tokenization import VectorTokenizer
from twm.models.attention import CausalSelfAttention
from twm.models.heads import DynamicsHead


class TransformerBlock(nnx.Module):
    """Causal Transformer Encoder block with LayerNorm, Self-Attention, and MLP."""

    def __init__(self, embed_dim: int, num_heads: int, mlp_dim: int, *, rngs: nnx.Rngs):
        self.attn = CausalSelfAttention(embed_dim, num_heads, rngs=rngs)
        self.norm1 = nnx.LayerNorm(embed_dim, rngs=rngs)
        self.norm2 = nnx.LayerNorm(embed_dim, rngs=rngs)
        self.mlp = nnx.Sequential(
            nnx.Linear(embed_dim, mlp_dim, rngs=rngs),
            nnx.gelu,
            nnx.Linear(mlp_dim, embed_dim, rngs=rngs)
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class TransformerWorldModel(nnx.Module):
    """Causal Transformer World Model for vector sequence dynamics prediction."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        mlp_dim: int = 512,
        max_seq_len: int = 128,
        *,
        rngs: nnx.Rngs,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len

        self.tokenizer = VectorTokenizer(state_dim, action_dim, embed_dim, rngs=rngs)

        # Learnable positional embeddings
        self.pos_embed = nnx.Param(jax.random.normal(rngs.params(), (1, max_seq_len, embed_dim)) * 0.02)

        list_factory = getattr(nnx, "List", list)
        self.blocks = list_factory([
            TransformerBlock(embed_dim, num_heads, mlp_dim, rngs=rngs)
            for _ in range(num_layers)
        ])
        self.final_norm = nnx.LayerNorm(embed_dim, rngs=rngs)
        self.heads = DynamicsHead(embed_dim, state_dim, mlp_dim, rngs=rngs)

    def __call__(self, states: jax.Array, actions: jax.Array) -> Tuple[jax.Array, jax.Array, jax.Array]:
        """Forward pass over continuous state and action sequences.

        Input:
            states: [Batch, Time, State_Dim]
            actions: [Batch, Time, Action_Dim]
        Returns:
            pred_next_states: [Batch, Time, State_Dim]
            pred_rewards: [Batch, Time]
            pred_continues: [Batch, Time]
        """
        # Embed and interleave state and action tokens -> [Batch, 2*Time, Embed_Dim]
        tokens = self.tokenizer.combine_tokens(states, actions)
        batch_size, token_seq_len, _ = tokens.shape

        # Add positional embedding (compatible with both Flax 0.10 and 0.12+)
        pos_emb = getattr(self.pos_embed, "value", self.pos_embed)
        tokens = tokens + pos_emb[:, :token_seq_len, :]

        # Pass through causal transformer layers
        h = tokens
        for block in self.blocks:
            h = block(h)
        h = self.final_norm(h)

        # Action tokens are at odd indices 1, 3, 5... representing context after s_t and a_t
        action_token_features = h[:, 1::2, :]

        pred_next_states = states + self.heads.predict_state(action_token_features)
        pred_rewards = self.heads.predict_reward(action_token_features)
        pred_continues = self.heads.predict_continue(action_token_features)

        return pred_next_states, pred_rewards, pred_continues

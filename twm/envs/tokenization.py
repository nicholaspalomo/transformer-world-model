"""Vector tokenization for mapping continuous state/action vectors into token embeddings."""

import jax
import jax.numpy as jnp
from flax import nnx


# LINT.IfChange(token_projection)
class VectorTokenizer(nnx.Module):
    """Embeds continuous state and action vectors into shared latent token space."""

    def __init__(self, state_dim: int, action_dim: int, embed_dim: int, rngs: nnx.Rngs):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.embed_dim = embed_dim

        self.state_embed = nnx.Linear(state_dim, embed_dim, rngs=rngs)
        self.action_embed = nnx.Linear(action_dim, embed_dim, rngs=rngs)

    def embed_states(self, states: jax.Array) -> jax.Array:
        """Embed continuous state vectors [Batch, Time, State_Dim] -> [Batch, Time, Embed_Dim]."""
        return self.state_embed(states)

    def embed_actions(self, actions: jax.Array) -> jax.Array:
        """Embed continuous action vectors [Batch, Time, Action_Dim] -> [Batch, Time, Embed_Dim]."""
        return self.action_embed(actions)

    def combine_tokens(self, states: jax.Array, actions: jax.Array) -> jax.Array:
        """Interleave state and action token embeddings into sequential token stream.

        Input:
            states: [Batch, Time, State_Dim]
            actions: [Batch, Time, Action_Dim]
        Output:
            tokens: [Batch, 2 * Time, Embed_Dim] (s_0, a_0, s_1, a_1, ...)
        """
        s_tokens = self.embed_states(states)
        a_tokens = self.embed_actions(actions)

        batch_size, seq_len, embed_dim = s_tokens.shape
        tokens = jnp.zeros((batch_size, 2 * seq_len, embed_dim), dtype=s_tokens.dtype)
        tokens = tokens.at[:, 0::2, :].set(s_tokens)
        tokens = tokens.at[:, 1::2, :].set(a_tokens)
        return tokens


# LINT.ThenChange(//twm/models/transformer.py:token_embed, //twm/models/heads.py:dynamics_head)

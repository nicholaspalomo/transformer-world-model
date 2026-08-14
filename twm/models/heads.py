"""MLP prediction heads for next state, reward, and terminal prediction."""

import jax
import jax.numpy as jnp
from flax import nnx


class DynamicsHead(nnx.Module):
    """MLP prediction heads mapping latent transformer representations to environmental state & reward transitions."""

    def __init__(self, embed_dim: int, state_dim: int, mlp_dim: int = 256, *, rngs: nnx.Rngs):
        self.state_head = nnx.Sequential(
            nnx.Linear(embed_dim, mlp_dim, rngs=rngs),
            nnx.gelu,
            nnx.Linear(mlp_dim, state_dim, rngs=rngs)
        )
        self.reward_head = nnx.Sequential(
            nnx.Linear(embed_dim, mlp_dim, rngs=rngs),
            nnx.gelu,
            nnx.Linear(mlp_dim, 1, rngs=rngs)
        )
        self.continue_head = nnx.Sequential(
            nnx.Linear(embed_dim, mlp_dim, rngs=rngs),
            nnx.gelu,
            nnx.Linear(mlp_dim, 1, rngs=rngs)
        )

    def predict_state(self, hidden_features: jax.Array) -> jax.Array:
        """Predict next continuous state vector delta from hidden token features."""
        return self.state_head(hidden_features)

    def predict_reward(self, hidden_features: jax.Array) -> jax.Array:
        """Predict scalar reward from hidden token features."""
        return jnp.squeeze(self.reward_head(hidden_features), axis=-1)

    def predict_continue(self, hidden_features: jax.Array) -> jax.Array:
        """Predict discount / continuation probability."""
        logits = jnp.squeeze(self.continue_head(hidden_features), axis=-1)
        return jax.nn.sigmoid(logits)

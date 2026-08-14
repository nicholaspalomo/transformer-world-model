"""Unit test for Causal Transformer World Model forward pass and mask shape."""

import unittest

import jax
import jax.numpy as jnp
from flax import nnx

from twm.models.attention import CausalSelfAttention
from twm.models.transformer import TransformerWorldModel


class TestTransformerModel(unittest.TestCase):
    def test_causal_attention_shapes(self):
        rngs = nnx.Rngs(params=jax.random.PRNGKey(0))
        attn = CausalSelfAttention(embed_dim=64, num_heads=4, rngs=rngs)

        x = jnp.zeros((2, 10, 64))
        out = attn(x)
        self.assertEqual(out.shape, (2, 10, 64))

    def test_transformer_world_model_forward(self):
        rngs = nnx.Rngs(params=jax.random.PRNGKey(0))
        model = TransformerWorldModel(
            state_dim=27,
            action_dim=8,
            embed_dim=64,
            num_heads=4,
            num_layers=2,
            mlp_dim=128,
            rngs=rngs,
        )

        states = jnp.zeros((2, 16, 27))
        actions = jnp.zeros((2, 16, 8))

        pred_next_states, pred_rewards, pred_continues = model(states, actions)
        self.assertEqual(pred_next_states.shape, (2, 16, 27))
        self.assertEqual(pred_rewards.shape, (2, 16))
        self.assertEqual(pred_continues.shape, (2, 16))


if __name__ == "__main__":
    unittest.main()

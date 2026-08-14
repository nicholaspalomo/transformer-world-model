"""Unit test for MPPI Planner vectorized trajectory rollouts."""

import unittest
import jax
import jax.numpy as jnp
from flax import nnx

from twm.models.transformer import TransformerWorldModel
from twm.planner.mppi import MPPIPlanner


class TestMPPIPlanner(unittest.TestCase):

    def test_mppi_plan_execution(self):
        rngs = nnx.Rngs(params=jax.random.PRNGKey(0))
        model = TransformerWorldModel(
            state_dim=10,
            action_dim=2,
            embed_dim=32,
            num_heads=2,
            num_layers=1,
            mlp_dim=64,
            rngs=rngs,
        )

        def dummy_forward(states, actions):
            return model(states, actions)

        planner = MPPIPlanner(
            model_forward_fn=dummy_forward,
            action_dim=2,
            horizon=5,
            num_samples=20,
        )

        rng = jax.random.PRNGKey(42)
        current_state = jnp.zeros((10,))
        mean_actions = jnp.zeros((5, 2))

        opt_action, new_mean, costs = planner.plan(rng, current_state, mean_actions)
        self.assertEqual(opt_action.shape, (2,))
        self.assertEqual(new_mean.shape, (5, 2))
        self.assertEqual(costs.shape, (20,))


if __name__ == "__main__":
    unittest.main()

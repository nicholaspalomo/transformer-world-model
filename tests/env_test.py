"""Unit test for Brax environment wrapping and vector tokenization."""

import unittest
import jax
import jax.numpy as jnp
from flax import nnx

from twm.envs.brax_wrapper import BraxEnvWrapper
from twm.envs.tokenization import VectorTokenizer
from twm.utils.buffer import TrajectoryReplayBuffer


class TestEnvironmentAndTokenizer(unittest.TestCase):

    def test_brax_wrapper_reset_step(self):
        env = BraxEnvWrapper(env_name="ant")
        key = jax.random.PRNGKey(0)
        state, obs = env.reset(key)
        self.assertEqual(obs.shape, (env.observation_size,))

        action = jnp.zeros((env.action_size,))
        next_state, next_obs, reward, done, _ = env.step(state, action, key)
        self.assertEqual(next_obs.shape, (env.observation_size,))

    def test_vector_tokenizer(self):
        rngs = nnx.Rngs(params=jax.random.PRNGKey(0))
        tokenizer = VectorTokenizer(state_dim=27, action_dim=8, embed_dim=64, rngs=rngs)

        states = jnp.zeros((4, 10, 27))
        actions = jnp.zeros((4, 10, 8))

        tokens = tokenizer.combine_tokens(states, actions)
        self.assertEqual(tokens.shape, (4, 20, 64))

    def test_replay_buffer(self):
        buffer = TrajectoryReplayBuffer(max_capacity=100, state_dim=10, action_dim=2)
        s = jnp.ones((10,))
        a = jnp.ones((2,))
        for _ in range(50):
            buffer.add(s, a, 1.0, s, 0.0)

        batch = buffer.sample_sequences(jax.random.PRNGKey(0), batch_size=8, seq_len=10)
        self.assertEqual(batch["states"].shape, (8, 10, 10))
        self.assertEqual(batch["actions"].shape, (8, 10, 2))


if __name__ == "__main__":
    unittest.main()

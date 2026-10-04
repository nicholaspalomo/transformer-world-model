"""Unit tests for model checkpointing and replay buffer serialization."""

import os
import shutil
import tempfile
import unittest

import jax
import jax.numpy as jnp
from flax import nnx

from twm.models.diffusion_policy import DiffusionPolicy
from twm.models.transformer import TransformerWorldModel
from twm.utils.buffer import TrajectoryReplayBuffer
from twm.utils.checkpoint import load_checkpoint, save_checkpoint
from twm.utils.prng import PRNGSequence


class TestCheckpointAndBuffer(unittest.TestCase):
    """Test suite for checkpoint saving/restoration and replay buffer persistence."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.prng = PRNGSequence(seed=42)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_transformer_world_model_checkpoint(self):
        rngs = nnx.Rngs(params=self.prng.next())
        model1 = TransformerWorldModel(
            state_dim=35,
            action_dim=12,
            embed_dim=64,
            num_heads=2,
            num_layers=2,
            mlp_dim=128,
            rngs=rngs,
        )

        ckpt_path = os.path.join(self.test_dir, "twm_test.npz")
        saved_path = save_checkpoint(model1, ckpt_path)
        self.assertTrue(os.path.exists(saved_path))

        # Create distinct model
        rngs2 = nnx.Rngs(params=self.prng.next())
        model2 = TransformerWorldModel(
            state_dim=35,
            action_dim=12,
            embed_dim=64,
            num_heads=2,
            num_layers=2,
            mlp_dim=128,
            rngs=rngs2,
        )

        # Restore weights
        load_checkpoint(model2, ckpt_path)

        # Verify forward passes match exactly
        dummy_s = jax.random.normal(self.prng.next(), (2, 4, 35))
        dummy_a = jax.random.normal(self.prng.next(), (2, 4, 12))

        pred_s1, pred_r1, _ = model1(dummy_s, dummy_a)
        pred_s2, pred_r2, _ = model2(dummy_s, dummy_a)

        self.assertTrue(jnp.allclose(pred_s1, pred_s2, atol=1e-5))
        self.assertTrue(jnp.allclose(pred_r1, pred_r2, atol=1e-5))

    def test_diffusion_policy_checkpoint(self):
        rngs = nnx.Rngs(params=self.prng.next())
        policy1 = DiffusionPolicy(
            state_dim=35,
            action_dim=12,
            embed_dim=64,
            num_layers=2,
            num_timesteps=4,
            rngs=rngs,
        )

        ckpt_path = os.path.join(self.test_dir, "policy_test.npz")
        save_checkpoint(policy1, ckpt_path)
        self.assertTrue(os.path.exists(ckpt_path))

        policy2 = DiffusionPolicy(
            state_dim=35,
            action_dim=12,
            embed_dim=64,
            num_layers=2,
            num_timesteps=4,
            rngs=nnx.Rngs(params=self.prng.next()),
        )
        load_checkpoint(policy2, ckpt_path)

        dummy_obs = jax.random.normal(self.prng.next(), (4, 35))
        eval_key = self.prng.next()
        traj1 = policy1.sample_trajectory(eval_key, dummy_obs, deterministic=True)
        traj2 = policy2.sample_trajectory(eval_key, dummy_obs, deterministic=True)

        self.assertTrue(jnp.allclose(traj1.actions, traj2.actions, atol=1e-5))

    def test_replay_buffer_persistence(self):
        buf1 = TrajectoryReplayBuffer(max_capacity=100, state_dim=35, action_dim=12)
        for i in range(20):
            s = jnp.ones(35, dtype=jnp.float32) * i
            a = jnp.zeros(12, dtype=jnp.float32)
            buf1.add(s, a, 1.0, s + 0.1, 0.0)

        buf_path = os.path.join(self.test_dir, "buffer_test.npz")
        saved_file = buf1.save(buf_path)
        self.assertTrue(os.path.exists(saved_file))

        buf2 = TrajectoryReplayBuffer.load_from_file(buf_path, max_capacity=100)
        self.assertEqual(buf2.size, 20)
        self.assertEqual(buf2.state_dim, 35)
        self.assertEqual(buf2.action_dim, 12)
        self.assertTrue(jnp.allclose(buf1.states[:20], buf2.states[:20]))


if __name__ == "__main__":
    unittest.main()

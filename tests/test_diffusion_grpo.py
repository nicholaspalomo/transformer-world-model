"""Unit tests for Diffusion Policy, Joint PD Controller, and GRPO training on ANYmal."""

import unittest

import jax
import jax.numpy as jnp
from flax import nnx

from twm.algorithms.diffusion_grpo import (
    DiffusionGRPOConfig,
    DiffusionGRPOTrainer,
    GRPORolloutBatch,
)
from twm.envs.anymal_env import ANYmalBEnv
from twm.envs.pd_controller import NOMINAL_JOINT_POS, JointPDController, PDControlOutput
from twm.models.diffusion_policy import DiffusionPolicy, ReverseTrajectory
from twm.utils.prng import PRNGSequence


class TestPDController(unittest.TestCase):
    """Test suite for lower-level joint PD controller."""

    def setUp(self):
        self.controller = JointPDController(
            kp=50.0,
            kd=1.5,
            tau_max=40.0,
            action_scale=0.3,
            nominal_qpos=NOMINAL_JOINT_POS,
        )

    def test_target_positions_and_clipping(self):
        # Zero action should correspond exactly to nominal standing pose
        zero_action = jnp.zeros(12, dtype=jnp.float32)
        q_target = self.controller.compute_target_positions(zero_action)
        self.assertTrue(jnp.allclose(q_target, NOMINAL_JOINT_POS, atol=1e-5))

        # Full positive action should offset by +action_scale
        ones_action = jnp.ones(12, dtype=jnp.float32)
        q_target_pos = self.controller.compute_target_positions(ones_action)
        self.assertTrue(jnp.all(q_target_pos >= NOMINAL_JOINT_POS - 1e-5))

    def test_torque_computation(self):
        current_q = NOMINAL_JOINT_POS
        current_qd = jnp.zeros(12, dtype=jnp.float32)
        q_target = NOMINAL_JOINT_POS + 0.1  # 0.1 rad error

        out: PDControlOutput = self.controller.compute_torques(
            q_target, current_q, current_qd
        )

        # Expected torque = 50 * 0.1 = 5.0 Nm
        expected_torque = 50.0 * 0.1
        self.assertTrue(jnp.allclose(out.torques, expected_torque, atol=1e-4))
        self.assertEqual(out.torques.shape, (12,))

    def test_torque_clipping(self):
        # Huge error to test torque limit clamping
        current_q = NOMINAL_JOINT_POS
        current_qd = jnp.zeros(12, dtype=jnp.float32)
        q_target = NOMINAL_JOINT_POS + 5.0  # 5 rad error -> 250 Nm raw torque

        out: PDControlOutput = self.controller.compute_torques(
            q_target, current_q, current_qd
        )
        self.assertTrue(jnp.all(out.torques <= 40.0))
        self.assertTrue(jnp.all(out.torques >= -40.0))


class TestANYmalEnvironment(unittest.TestCase):
    """Test suite for ANYmal B environment with PD controller integration."""

    def setUp(self):
        self.env = ANYmalBEnv(backend="positional", target_velocity=0.8)
        self.prng = PRNGSequence(seed=42)

    def test_reset_and_dimensions(self):
        state = self.env.reset(self.prng.next())
        self.assertEqual(self.env.observation_size, 35)
        self.assertEqual(self.env.action_size, 12)
        self.assertEqual(state.obs.shape, (35,))

    def test_step_with_pd_controller(self):
        state = self.env.reset(self.prng.next())
        action = jnp.zeros(12, dtype=jnp.float32)
        next_state = self.env.step(state, action)

        self.assertEqual(next_state.obs.shape, (35,))
        self.assertTrue(isinstance(float(next_state.reward), float))
        self.assertIn("forward_vel", next_state.metrics)
        self.assertIn("torso_height", next_state.metrics)
        self.assertIn("tracking_reward", next_state.metrics)


class TestDiffusionPolicy(unittest.TestCase):
    """Test suite for Flax NNX Diffusion Policy architecture."""

    def setUp(self):
        self.prng = PRNGSequence(seed=42)
        rngs = nnx.Rngs(params=self.prng.next())
        self.policy = DiffusionPolicy(
            state_dim=35,
            action_dim=12,
            embed_dim=128,
            num_layers=2,
            num_timesteps=4,
            rngs=rngs,
        )

    def test_sample_trajectory(self):
        batch_size = 4
        obs = jax.random.normal(self.prng.next(), (batch_size, 35))

        traj: ReverseTrajectory = self.policy.sample_trajectory(
            self.prng.next(), obs, deterministic=False
        )

        # Verify shapes
        self.assertEqual(traj.actions.shape, (batch_size, 12))
        self.assertEqual(traj.trajectories.shape, (5, batch_size, 12))  # K+1 = 5
        self.assertEqual(traj.log_probs.shape, (batch_size,))
        self.assertEqual(traj.step_log_probs.shape, (4, batch_size))

        # Check action bounds [-1, 1]
        self.assertTrue(jnp.all(traj.actions >= -1.0))
        self.assertTrue(jnp.all(traj.actions <= 1.0))

    def test_exact_trajectory_log_prob_evaluation(self):
        obs = jax.random.normal(self.prng.next(), (2, 35))
        traj = self.policy.sample_trajectory(self.prng.next(), obs)

        # Re-evaluate log probabilities on the same sampled trajectories
        recalc_lp, step_lps = self.policy.evaluate_trajectory_log_prob(
            traj.trajectories, obs
        )

        # Must match the sampling log probabilities exactly
        self.assertTrue(jnp.allclose(traj.log_probs, recalc_lp, atol=1e-4))
        self.assertTrue(jnp.allclose(traj.step_log_probs, step_lps, atol=1e-4))


class TestDiffusionGRPOTrainer(unittest.TestCase):
    """Test suite for GRPO advantage computation, loss, and training loop."""

    def setUp(self):
        self.prng = PRNGSequence(seed=42)
        rngs = nnx.Rngs(params=self.prng.next())
        self.policy = DiffusionPolicy(
            state_dim=35,
            action_dim=12,
            embed_dim=64,
            num_layers=2,
            num_timesteps=4,
            rngs=rngs,
        )
        self.env = ANYmalBEnv(backend="positional")
        self.config = DiffusionGRPOConfig(
            group_size=4,
            batch_size=2,
            rollout_horizon=4,
            learning_rate=1e-3,
            num_epochs=2,
        )
        self.trainer = DiffusionGRPOTrainer(
            policy=self.policy, env=self.env, config=self.config
        )

    def test_group_advantage_normalization(self):
        # Returns for B=2, G=4
        returns = jnp.array(
            [
                [1.0, 2.0, 3.0, 4.0],  # mean = 2.5
                [10.0, 10.0, 10.0, 10.0],  # zero variance
            ],
            dtype=jnp.float32,
        )

        advantages = self.trainer.compute_group_advantages(returns)

        # First group should have zero mean advantages
        self.assertTrue(jnp.allclose(jnp.mean(advantages[0]), 0.0, atol=1e-5))
        # Better returns get positive advantage, worse get negative
        self.assertTrue(advantages[0, 3] > 0.0)
        self.assertTrue(advantages[0, 0] < 0.0)

        # Second group with identical returns should have near-zero advantage
        self.assertTrue(jnp.allclose(advantages[1], 0.0, atol=1e-4))

    def test_sample_and_train_step(self):
        env_states = [self.env.reset(self.prng.next()) for _ in range(2)]

        rollout_batch, next_states = self.trainer.sample_group_rollouts(
            self.prng.next(), env_states
        )

        self.assertEqual(rollout_batch.obs.shape, (2, 4, 35))
        self.assertEqual(rollout_batch.actions.shape, (2, 4, 12))
        self.assertEqual(rollout_batch.returns.shape, (2, 4))
        self.assertEqual(rollout_batch.advantages.shape, (2, 4))
        self.assertEqual(len(next_states), 2)

        # Perform GRPO update step
        loss_output = self.trainer.train_step(rollout_batch)

        self.assertTrue(jnp.isfinite(loss_output.total_loss))
        self.assertTrue(jnp.isfinite(loss_output.policy_loss))
        self.assertTrue(jnp.isfinite(loss_output.kl_div))


if __name__ == "__main__":
    unittest.main()

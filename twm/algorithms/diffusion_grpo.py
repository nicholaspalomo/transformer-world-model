"""Group Relative Policy Optimization (GRPO) for Diffusion Policies."""

from dataclasses import dataclass
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from twm.envs.anymal_env import ANYmalBEnv
from twm.models.diffusion_policy import DiffusionPolicy, ReverseTrajectory
from twm.models.transformer import TransformerWorldModel
from twm.utils.buffer import TrajectoryReplayBuffer


class GRPORolloutBatch(NamedTuple):
    """Batch of group rollouts collected for GRPO optimization."""

    obs: jax.Array  # [B, G, State_Dim]
    actions: jax.Array  # [B, G, Action_Dim]
    trajectories: jax.Array  # [K+1, B, G, Action_Dim]
    old_log_probs: jax.Array  # [B, G]
    returns: jax.Array  # [B, G]
    advantages: jax.Array  # [B, G]
    metrics: dict[str, jax.Array]


class GRPOLossOutput(NamedTuple):
    """Loss metrics returned from GRPO loss evaluation."""

    total_loss: jax.Array
    policy_loss: jax.Array
    kl_div: jax.Array
    clip_fraction: jax.Array
    approx_kl: jax.Array


@dataclass
class DiffusionGRPOConfig:
    """Hyperparameters for Diffusion GRPO training."""

    group_size: int = 8  # Number of candidate rollouts sampled per state G
    batch_size: int = 8  # Number of parallel environment prompts/states B
    rollout_horizon: int = 16  # Multi-step rollout horizon per action decision
    gamma: float = 0.99  # Discount factor
    clip_eps: float = 0.2  # PPO/GRPO clipping epsilon
    beta_kl: float = 0.04  # KL divergence penalty weight
    learning_rate: float = 3e-4  # Optimizer learning rate
    weight_decay: float = 1e-4  # AdamW weight decay
    max_grad_norm: float = 1.0  # Gradient clipping max norm
    num_epochs: int = 4  # Optimization epochs per rollout batch
    adv_eps: float = 1e-6  # Epsilon for advantage standardization
    use_world_model: bool = True  # Approach 2 default: train policy in World Model imagination
    imagination_horizon: int = 16  # Rollout horizon inside the world model


class DiffusionGRPOTrainer:
    """Trainer implementing Group Relative Policy Optimization for Diffusion Policy."""

    def __init__(
        self,
        policy: DiffusionPolicy,
        env: ANYmalBEnv,
        config: DiffusionGRPOConfig | None = None,
        ref_policy: DiffusionPolicy | None = None,
        world_model: TransformerWorldModel | None = None,
        replay_buffer: TrajectoryReplayBuffer | None = None,
    ):
        self.policy = policy
        self.env = env
        self.config = config or DiffusionGRPOConfig()
        self.ref_policy = ref_policy
        self.world_model = world_model
        self.replay_buffer = replay_buffer

        # JIT-compiled environment step and reset functions for ultra-fast physics simulation
        self._step_fn = jax.jit(self.env.step)
        self._reset_fn = jax.jit(self.env.reset)

        # Optax Optimizer with gradient clipping and AdamW
        tx = optax.chain(
            optax.clip_by_global_norm(self.config.max_grad_norm),
            optax.adamw(
                learning_rate=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            ),
        )
        self.optimizer = nnx.Optimizer(self.policy, tx, wrt=nnx.Param)

    def compute_group_advantages(self, returns: jax.Array) -> jax.Array:
        """Compute relative advantages within each group G.

        Args:
            returns: [B, G] episodic returns

        Returns:
            advantages: [B, G] standardized relative advantages
        """
        group_mean = jnp.mean(returns, axis=-1, keepdims=True)  # [B, 1]
        group_std = jnp.std(returns, axis=-1, keepdims=True)  # [B, 1]
        advantages = (returns - group_mean) / (group_std + self.config.adv_eps)
        return advantages

    def sample_group_rollouts(
        self,
        rng: jax.Array,
        env_states: list[Any],
    ) -> tuple[GRPORolloutBatch, list[Any]]:
        """Collect G candidate rollouts per state using World Model imagination (default) or physical sim."""
        if self.world_model is not None and self.config.use_world_model:
            return self.sample_imagined_group_rollouts(rng, env_states)
        return self.sample_physical_group_rollouts(rng, env_states)

    def sample_imagined_group_rollouts(
        self,
        rng: jax.Array,
        env_states: list[Any],
    ) -> tuple[GRPORolloutBatch, list[Any]]:
        """Collect G candidate rollouts imagined through the Causal Transformer World Model."""
        B = len(env_states)
        G = self.config.group_size
        H = self.config.imagination_horizon
        gamma = self.config.gamma

        # Extract current observations [B, State_Dim]
        curr_obs = jnp.stack([s.obs for s in env_states], axis=0)

        # Broadcast observations to group dimension: [B, G, State_Dim]
        group_obs = jnp.repeat(curr_obs[:, None, :], G, axis=1)

        rng, rng_sample = jax.random.split(rng)
        # Sample G reverse diffusion paths per state
        traj: ReverseTrajectory = self.policy.sample_trajectory(
            rng_sample, group_obs, deterministic=False
        )

        # Flatten B and G for vectorized parallel imagination rollouts across B*G candidates
        flat_init_states = jnp.reshape(group_obs, (B * G, 1, self.env.observation_size))
        flat_actions = jnp.reshape(traj.actions, (B * G, 1, self.env.action_size))

        # Roll out H steps purely in JAX through TransformerWorldModel
        def imagine_step(carry_state, _):
            pred_next_states, pred_rewards, _ = self.world_model(carry_state, flat_actions)
            return pred_next_states, (pred_next_states[:, 0, :], pred_rewards[:, 0])

        _, (imagined_states, imagined_rewards) = jax.lax.scan(
            imagine_step, flat_init_states, None, length=H
        )
        # imagined_states: [H, B*G, State_Dim]
        # imagined_rewards: [H, B*G]

        discounts = gamma ** jnp.arange(H)
        total_returns = jnp.sum(imagined_rewards * discounts[:, None], axis=0)  # [B*G]
        returns = jnp.reshape(total_returns, (B, G))

        # Advance real environment along the best imagined candidate action
        best_candidate_indices = jnp.argmax(returns, axis=1)  # [B]
        next_env_states = []
        for b_idx in range(B):
            best_g = int(best_candidate_indices[b_idx])
            best_action = traj.actions[b_idx, best_g]
            sim_state = self._step_fn(env_states[b_idx], best_action)

            # Store grounded real transition into replay buffer if attached
            if self.replay_buffer is not None:
                self.replay_buffer.add(
                    env_states[b_idx].obs,
                    best_action,
                    float(sim_state.reward),
                    sim_state.obs,
                    float(sim_state.done),
                )

            if float(sim_state.done) > 0.5:
                rng, rng_reset = jax.random.split(rng)
                sim_state = self._reset_fn(rng_reset)
            next_env_states.append(sim_state)

        # Compute Group Relative Advantages
        advantages = self.compute_group_advantages(returns)

        # Extract telemetry metrics from imagined states
        # ANYmal state: base linear vel x is index 29, torso height z is index 24
        forward_vels = imagined_states[:, :, 29]
        torso_heights = imagined_states[:, :, 24]
        mean_actions_norm = jnp.mean(jnp.abs(flat_actions))

        metrics = {
            "mean_return": jnp.mean(returns),
            "max_return": jnp.max(returns),
            "min_return": jnp.min(returns),
            "mean_forward_vel": jnp.mean(forward_vels),
            "mean_torso_height": jnp.mean(torso_heights),
            "mean_torque": mean_actions_norm * 40.0,
            "imagined": jnp.array(1.0, dtype=jnp.float32),
        }

        rollout_batch = GRPORolloutBatch(
            obs=group_obs,
            actions=traj.actions,
            trajectories=traj.trajectories,
            old_log_probs=traj.log_probs,
            returns=returns,
            advantages=advantages,
            metrics=metrics,
        )

        return rollout_batch, next_env_states

    def sample_physical_group_rollouts(
        self,
        rng: jax.Array,
        env_states: list[Any],
    ) -> tuple[GRPORolloutBatch, list[Any]]:
        """Collect G candidate rollouts for each of the B environment states using Brax physics."""
        B = len(env_states)
        G = self.config.group_size
        H = self.config.rollout_horizon
        gamma = self.config.gamma

        # Extract current observations [B, State_Dim]
        curr_obs = jnp.stack([s.obs for s in env_states], axis=0)

        # Broadcast observations to group dimension: [B, G, State_Dim]
        group_obs = jnp.repeat(curr_obs[:, None, :], G, axis=1)

        rng, rng_sample = jax.random.split(rng)
        # Sample G reverse diffusion paths per state
        traj: ReverseTrajectory = self.policy.sample_trajectory(
            rng_sample, group_obs, deterministic=False
        )

        # Evaluate candidate actions in environment over horizon H using JIT-compiled step_fn
        returns = jnp.zeros((B, G), dtype=jnp.float32)
        forward_vels = []
        torso_heights = []
        mean_torques = []

        next_env_states = []
        for b_idx in range(B):
            best_candidate_state = env_states[b_idx]
            best_candidate_return = -float("inf")

            for g_idx in range(G):
                sim_state = env_states[b_idx]
                action_g = traj.actions[b_idx, g_idx]

                cand_return = 0.0
                discount = 1.0

                for _h in range(H):
                    sim_state = self._step_fn(sim_state, action_g)
                    cand_return += discount * float(sim_state.reward)
                    discount *= gamma

                    forward_vels.append(float(sim_state.metrics.get("forward_vel", 0.0)))
                    torso_heights.append(float(sim_state.metrics.get("torso_height", 0.55)))
                    mean_torques.append(float(sim_state.metrics.get("mean_torque", 0.0)))

                    if float(sim_state.done) > 0.5:
                        break

                returns = returns.at[b_idx, g_idx].set(cand_return)

                if cand_return > best_candidate_return or g_idx == 0:
                    best_candidate_return = cand_return
                    best_candidate_state = sim_state

            # Advance real environment along the highest-performing group rollout
            if float(best_candidate_state.done) > 0.5:
                rng, rng_reset = jax.random.split(rng)
                best_candidate_state = self._reset_fn(rng_reset)

            next_env_states.append(best_candidate_state)

        # Compute Group Relative Advantages
        advantages = self.compute_group_advantages(returns)

        metrics = {
            "mean_return": jnp.mean(returns),
            "max_return": jnp.max(returns),
            "min_return": jnp.min(returns),
            "mean_forward_vel": jnp.mean(jnp.array(forward_vels)),
            "mean_torso_height": jnp.mean(jnp.array(torso_heights)),
            "mean_torque": jnp.mean(jnp.array(mean_torques)),
            "imagined": jnp.array(0.0, dtype=jnp.float32),
        }

        rollout_batch = GRPORolloutBatch(
            obs=group_obs,
            actions=traj.actions,
            trajectories=traj.trajectories,
            old_log_probs=traj.log_probs,
            returns=returns,
            advantages=advantages,
            metrics=metrics,
        )

        return rollout_batch, next_env_states

    def compute_loss(
        self,
        model: DiffusionPolicy,
        batch: GRPORolloutBatch,
    ) -> tuple[jax.Array, GRPOLossOutput]:
        """Compute GRPO clipped surrogate objective and KL penalty."""
        # 1. Re-evaluate log-probabilities under current policy parameters
        new_log_probs, _ = model.evaluate_trajectory_log_prob(
            batch.trajectories,
            batch.obs,
        )  # [B, G]

        # 2. Probability ratio r_i(theta) = exp(log pi_theta - log pi_old)
        log_ratios = new_log_probs - batch.old_log_probs
        clipped_log_ratios = jnp.clip(log_ratios, -10.0, 10.0)
        ratios = jnp.exp(clipped_log_ratios)

        # 3. Clipped surrogate loss
        surr1 = ratios * batch.advantages
        surr2 = (
            jnp.clip(ratios, 1.0 - self.config.clip_eps, 1.0 + self.config.clip_eps)
            * batch.advantages
        )
        policy_loss = -jnp.mean(jnp.minimum(surr1, surr2))

        # 4. KL divergence penalty (approximate KL = 0.5 * (log r)^2)
        approx_kl = 0.5 * jnp.mean(jnp.square(clipped_log_ratios))

        if self.ref_policy is not None:
            ref_log_probs, _ = self.ref_policy.evaluate_trajectory_log_prob(
                batch.trajectories, batch.obs
            )
            ref_log_ratios = jnp.clip(new_log_probs - ref_log_probs, -10.0, 10.0)
            kl_penalty = jnp.mean(jnp.exp(ref_log_ratios) - 1.0 - ref_log_ratios)
        else:
            kl_penalty = approx_kl

        total_loss = policy_loss + self.config.beta_kl * kl_penalty

        # Metrics
        clip_fraction = jnp.mean((jnp.abs(ratios - 1.0) > self.config.clip_eps).astype(jnp.float32))

        loss_output = GRPOLossOutput(
            total_loss=total_loss,
            policy_loss=policy_loss,
            kl_div=kl_penalty,
            clip_fraction=clip_fraction,
            approx_kl=approx_kl,
        )

        return total_loss, loss_output

    def train_step(self, batch: GRPORolloutBatch) -> GRPOLossOutput:
        """Perform multi-epoch optimization on collected group rollout batch."""
        accum_metrics = None

        for _ in range(self.config.num_epochs):

            def loss_fn(m):
                return self.compute_loss(m, batch)

            grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
            (_, loss_out), grads = grad_fn(self.policy)

            try:
                self.optimizer.update(grads)
            except TypeError:
                self.optimizer.update(self.policy, grads)

            accum_metrics = loss_out

        return accum_metrics

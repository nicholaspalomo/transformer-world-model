"""Model Predictive Path Integral (MPPI) planner compiled with jax.lax.scan."""

from typing import Tuple, Callable
import jax
import jax.numpy as jnp


class MPPIPlanner:
    """Vectorized MPPI Planner using JAX scan for fast parallel dynamics rollouts."""

    def __init__(
        self,
        model_forward_fn: Callable,
        action_dim: int,
        horizon: int = 15,
        num_samples: int = 1000,
        temperature: float = 0.5,
        noise_std: float = 0.5,
        gamma: float = 0.99,
    ):
        self.model_forward_fn = model_forward_fn
        self.action_dim = action_dim
        self.horizon = horizon
        self.num_samples = num_samples
        self.temperature = temperature
        self.noise_std = noise_std
        self.gamma = gamma

    def plan(
        self,
        rng: jax.Array,
        current_state: jax.Array,
        mean_actions: jax.Array,
    ) -> Tuple[jax.Array, jax.Array, jax.Array]:
        """Optimize trajectory actions using sampled perturbations and JAX scan auto-regression.

        Args:
            rng: PRNGKey
            current_state: [State_Dim]
            mean_actions: [Horizon, Action_Dim]
        Returns:
            optimal_action: [Action_Dim]
            new_mean_actions: [Horizon, Action_Dim]
            trajectory_costs: [Num_Samples]
        """
        rng_noise, rng_eval = jax.random.split(rng)

        # 1. Sample N trajectory action perturbations
        noise = jax.random.normal(
            rng_noise, (self.num_samples, self.horizon, self.action_dim)
        ) * self.noise_std
        sampled_actions = mean_actions[None, :, :] + noise  # [N, H, Action_Dim]
        sampled_actions = jnp.clip(sampled_actions, -1.0, 1.0)

        # 2. Roll out N trajectories using jax.lax.scan over Horizon
        def step_fn(carry_state, action_t):
            # carry_state: [N, State_Dim]
            # action_t: [N, Action_Dim]
            states_in = carry_state[:, None, :]
            actions_in = action_t[:, None, :]
            pred_next_states, pred_rewards, _ = self.model_forward_fn(states_in, actions_in)
            next_state = pred_next_states[:, 0, :]
            reward = pred_rewards[:, 0]
            return next_state, (next_state, reward)

        init_states = jnp.tile(current_state[None, :], (self.num_samples, 1))
        actions_time_first = sampled_actions.swapaxes(0, 1)  # [H, N, Action_Dim]

        _, (rolled_states, rolled_rewards) = jax.lax.scan(
            step_fn, init_states, actions_time_first
        )

        # Swap back time and sample dimensions -> [N, H, State_Dim], [N, H]
        rolled_states = rolled_states.swapaxes(0, 1)
        rolled_rewards = rolled_rewards.swapaxes(0, 1)

        # 3. Compute costs and MPPI softmax weights
        discounts = self.gamma ** jnp.arange(self.horizon)
        returns = jnp.sum(rolled_rewards * discounts[None, :], axis=1)
        costs = -returns

        min_cost = jnp.min(costs)
        weights = jax.nn.softmax(-(costs - min_cost) / self.temperature)

        # 4. Weight action perturbations to compute optimal trajectory
        weighted_noise = jnp.sum(weights[:, None, None] * noise, axis=0)  # [H, Action_Dim]
        new_mean_actions = mean_actions + weighted_noise
        new_mean_actions = jnp.clip(new_mean_actions, -1.0, 1.0)

        optimal_action = new_mean_actions[0]
        return optimal_action, new_mean_actions, costs

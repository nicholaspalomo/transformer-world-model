"""Trajectory cost and reward evaluation functions for MPPI rollout scoring."""

import jax
import jax.numpy as jnp


# LINT.IfChange(trajectory_cost)
def trajectory_cost_fn(
    predicted_states: jax.Array, predicted_rewards: jax.Array, gamma: float = 0.99
) -> jax.Array:
    """Compute trajectory cost from predicted rewards over horizon H.

    Input:
        predicted_states: [Num_Samples, Horizon, State_Dim]
        predicted_rewards: [Num_Samples, Horizon]
    Returns:
        costs: [Num_Samples]
    """
    horizon = predicted_rewards.shape[1]
    discounts = gamma ** jnp.arange(horizon)
    returns = jnp.sum(predicted_rewards * discounts, axis=1)

    # Cost is negative return (MPPI minimizes cost)
    return -returns


# LINT.ThenChange(//twm/planner/mppi.py:mppi_planner)

#!/usr/bin/env python3
"""Visualization script for comparing imagined vs ground-truth Brax trajectory rollouts."""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from flax import nnx

from twm.envs.brax_wrapper import BraxEnvWrapper
from twm.models.transformer import TransformerWorldModel
from twm.utils.prng import PRNGSequence


def visualize_rollouts():
    print("=== Visualizing Ground Truth vs Transformer Imagined Trajectories ===")
    env = BraxEnvWrapper(env_name="ant")
    prng = PRNGSequence(seed=42)

    state, obs = env.reset(prng.next())
    real_states = [obs]
    actions = []

    horizon = 20
    for _ in range(horizon):
        a = jax.random.uniform(prng.next(), (env.action_size,), minval=-1.0, maxval=1.0)
        state, obs, r, done, _ = env.step(state, a, prng.next())
        real_states.append(obs)
        actions.append(a)

    real_states = jnp.stack(real_states, axis=0)  # [T+1, State_Dim]
    actions = jnp.stack(actions, axis=0)  # [T, Action_Dim]

    rngs = nnx.Rngs(params=prng.next())
    model = TransformerWorldModel(
        state_dim=env.observation_size,
        action_dim=env.action_size,
        embed_dim=64,
        num_heads=2,
        num_layers=1,
        mlp_dim=128,
        rngs=rngs,
    )

    pred_states, _, _ = model(real_states[:-1][None, :, :], actions[None, :, :])
    imagined_states = pred_states[0]  # [T, State_Dim]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(real_states[1:, 0], label="Ground Truth State[0]", color="blue", linewidth=2)
    ax.plot(
        imagined_states[:, 0],
        label="Imagined World Model State[0]",
        color="orange",
        linestyle="--",
        linewidth=2,
    )
    ax.set_title("Brax Ant Physics vs Transformer World Model Trajectory Rollout")
    ax.set_xlabel("Horizon Step")
    ax.set_ylabel("State Dimension 0")
    ax.legend()
    ax.grid(True)

    plt.savefig("rollout_comparison.png", dpi=150, bbox_inches="tight")
    print("Saved rollout comparison plot to 'rollout_comparison.png'.")


if __name__ == "__main__":
    visualize_rollouts()

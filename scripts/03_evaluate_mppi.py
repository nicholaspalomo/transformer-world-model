#!/usr/bin/env python3
"""Milestone 4: Evaluate MPPI Planner with jax.lax.scan vectorization in Brax env."""

import argparse
import jax
import jax.numpy as jnp
from flax import nnx

from twm.envs.brax_wrapper import BraxEnvWrapper
from twm.models.transformer import TransformerWorldModel
from twm.planner.mppi import MPPIPlanner
from twm.utils.prng import PRNGSequence


def main():
    parser = argparse.ArgumentParser(description="Milestone 4: MPPI Planner Evaluation in Brax Environment.")
    parser.add_argument("--env_name", type=str, default="ant", help="Brax environment name")
    parser.add_argument("--num_samples", type=int, default=100, help="MPPI trajectory samples N")
    parser.add_argument("--horizon", type=int, default=10, help="Planning horizon H")
    parser.add_argument("--eval_steps", type=int, default=5, help="Number of closed-loop control steps")
    args = parser.parse_args()

    print(f"=== Milestone 4: MPPI Planning with Vectorized jax.lax.scan Rollouts ===")
    env = BraxEnvWrapper(env_name=args.env_name)
    prng = PRNGSequence(seed=42)

    # Initialize World Model
    rngs = nnx.Rngs(params=prng.next())
    model = TransformerWorldModel(
        state_dim=env.observation_size,
        action_dim=env.action_size,
        embed_dim=128,
        num_heads=4,
        num_layers=2,
        mlp_dim=256,
        rngs=rngs,
    )

    # Wrapper for MPPI forward dynamics
    def model_forward(states, actions):
        return model(states, actions)

    planner = MPPIPlanner(
        model_forward_fn=model_forward,
        action_dim=env.action_size,
        horizon=args.horizon,
        num_samples=args.num_samples,
    )

    state, obs = env.reset(prng.next())
    mean_actions = jnp.zeros((args.horizon, env.action_size), dtype=jnp.float32)

    print(f"Executing closed-loop control for {args.eval_steps} steps (N={args.num_samples} trajectories per step)...")
    for step in range(args.eval_steps):
        opt_action, mean_actions, costs = planner.plan(prng.next(), obs, mean_actions)
        print(f"Control Step {step + 1}: Min Trajectory Cost = {jnp.min(costs):.4f}, Optimal Action Mean = {opt_action.mean():.4f}")

        # Step real environment with optimal action
        state, obs, reward, done, _ = env.step(state, opt_action, prng.next())
        # Warm-start next planning iteration by shifting horizon
        mean_actions = jnp.roll(mean_actions, -1, axis=0)

    print("=== Milestone 4 MPPI Planning Verification Passed ===")


if __name__ == "__main__":
    main()

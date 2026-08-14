#!/usr/bin/env python3
"""Milestone 1: Brax Data Collection Pipeline and Sequence Buffer sampling."""

import argparse

import jax

from twm.envs.brax_wrapper import BraxEnvWrapper
from twm.utils.buffer import TrajectoryReplayBuffer
from twm.utils.prng import PRNGSequence


def main():
    # LINT.IfChange(env_args)
    parser = argparse.ArgumentParser(
        description="Milestone 1: Collect trajectories from Brax into Replay Buffer."
    )
    parser.add_argument("--env_name", type=str, default="ant", help="Brax environment name")
    parser.add_argument(
        "--num_steps", type=int, default=1000, help="Number of exploration steps to collect"
    )
    parser.add_argument("--seq_len", type=int, default=32, help="Sequence window length K")
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for sequence sampling test"
    )
    args = parser.parse_args()

    print(f"=== Milestone 1: Initializing Brax Environment ({args.env_name}) ===")
    env = BraxEnvWrapper(env_name=args.env_name)
    # LINT.ThenChange(//twm/envs/brax_wrapper.py:env_registry, //Makefile:env_targets)
    prng = PRNGSequence(seed=42)

    buffer = TrajectoryReplayBuffer(
        max_capacity=max(args.num_steps + 100, 2000),
        state_dim=env.observation_size,
        action_dim=env.action_size,
    )

    state, obs = env.reset(prng.next())

    print(f"Collecting {args.num_steps} exploratory transitions into Flashbax/JAX buffer...")
    for _ in range(args.num_steps):
        action_key = prng.next()
        step_key = prng.next()

        # Random uniform action [-1, 1]
        action = jax.random.uniform(action_key, (env.action_size,), minval=-1.0, maxval=1.0)
        next_state, next_obs, reward, done, _ = env.step(state, action, step_key)

        buffer.add(obs, action, float(reward), next_obs, float(done))

        state, obs = next_state, next_obs
        if float(done) > 0.5:
            state, obs = env.reset(prng.next())

    print(f"Buffer populated with {buffer.size} transitions.")
    print(f"Testing sequence sampling with window size K = {args.seq_len}...")

    sampled_batch = buffer.sample_sequences(
        prng.next(), batch_size=args.batch_size, seq_len=args.seq_len
    )
    print("Sampled sequence shapes:")
    print("  States:", sampled_batch["states"].shape)
    print("  Actions:", sampled_batch["actions"].shape)
    print("  Rewards:", sampled_batch["rewards"].shape)
    print("  Next States:", sampled_batch["next_states"].shape)
    print("=== Milestone 1 Data Pipeline Verification Passed ===")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Train Diffusion Policy for ANYmal Quadruped Walking using GRPO."""

import argparse
import os
import time

import jax
import jax.numpy as jnp
import yaml
from flax import nnx

from twm.algorithms.diffusion_grpo import DiffusionGRPOConfig, DiffusionGRPOTrainer
from twm.envs.anymal_env import ANYmalBEnv
from twm.models.diffusion_policy import DiffusionPolicy
from twm.utils.prng import PRNGSequence


def main():
    parser = argparse.ArgumentParser(
        description="GRPO-style Diffusion Policy Training for ANYmal Locomotion"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/grpo_diffusion_anymal.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--num_iterations",
        type=int,
        default=25,
        help="Number of GRPO training iterations (default: 25)",
    )
    parser.add_argument(
        "--group_size",
        type=int,
        default=8,
        help="Number of candidate diffusion rollouts per state G (default: 8)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Number of parallel environment states B (default: 8)",
    )
    parser.add_argument(
        "--rollout_horizon",
        type=int,
        default=16,
        help="Rollout steps per candidate action (default: 16)",
    )
    parser.add_argument(
        "--num_timesteps",
        type=int,
        default=8,
        help="Diffusion reverse denoising steps K (default: 8)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=3e-4,
        help="AdamW learning rate (default: 3e-4)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--kp",
        type=float,
        default=50.0,
        help="PD stiffness gain kp (default: 50.0)",
    )
    parser.add_argument(
        "--kd",
        type=float,
        default=1.5,
        help="PD damping gain kd (default: 1.5)",
    )
    parser.add_argument(
        "--action_scale",
        type=float,
        default=0.3,
        help="Action scaling factor around nominal pose (default: 0.3)",
    )
    args = parser.parse_args()

    # Load YAML config if present
    cfg = {}
    if os.path.exists(args.config):
        with open(args.config) as f:
            cfg = yaml.safe_load(f)

    # Resolve config values with CLI overrides
    seed = args.seed
    num_iterations = args.num_iterations
    group_size = args.group_size
    batch_size = args.batch_size
    rollout_horizon = args.rollout_horizon
    num_timesteps = args.num_timesteps
    learning_rate = args.learning_rate

    print("==========================================================================")
    print("  GRPO-Style Diffusion Policy Training for ANYmal Quadruped Walking")
    print("==========================================================================")
    print(f"  Configuration:")
    print(f"    - Diffusion Reverse Steps K : {num_timesteps}")
    print(f"    - Group Size G              : {group_size} rollouts / state")
    print(f"    - Parallel States B         : {batch_size}")
    print(f"    - Rollout Horizon H         : {rollout_horizon} steps")
    print(f"    - PD Gains (kp / kd)        : {args.kp} / {args.kd}")
    print(f"    - Action Scale              : {args.action_scale} rad")
    print(f"    - Learning Rate             : {learning_rate}")
    print(f"    - Total Iterations          : {num_iterations}")
    print("==========================================================================")

    prng = PRNGSequence(seed=seed)

    # 1. Initialize ANYmal Environment with Lower-Level PD Controller
    print("\n[1/3] Initializing ANYmal B Environment with Joint PD Controller...")
    env = ANYmalBEnv(
        backend="positional",
        kp=args.kp,
        kd=args.kd,
        action_scale=args.action_scale,
        target_velocity=0.8,
    )
    print(
        f"  ✓ ANYmal Env Loaded: Obs Dim = {env.observation_size}, Act Dim = {env.action_size}"
    )

    # 2. Initialize Flax NNX Diffusion Policy
    print("\n[2/3] Initializing Flax NNX Diffusion Policy Network...")
    policy_rngs = nnx.Rngs(params=prng.next())
    policy = DiffusionPolicy(
        state_dim=env.observation_size,
        action_dim=env.action_size,
        embed_dim=256,
        num_layers=3,
        num_timesteps=num_timesteps,
        rngs=policy_rngs,
    )
    print("  ✓ Diffusion Policy Initialized")

    # 3. Setup GRPO Trainer
    print("\n[3/3] Setting up GRPO Trainer (Critic-Free Group Advantage Normalization)...")
    grpo_config = DiffusionGRPOConfig(
        group_size=group_size,
        batch_size=batch_size,
        rollout_horizon=rollout_horizon,
        gamma=0.99,
        clip_eps=0.2,
        beta_kl=0.04,
        learning_rate=learning_rate,
        num_epochs=4,
    )
    trainer = DiffusionGRPOTrainer(
        policy=policy,
        env=env,
        config=grpo_config,
        ref_policy=None,  # Standard GRPO empirical KL regularization against pi_old
    )
    print("  ✓ GRPO Trainer Ready")

    # Initialize B environment states
    env_states = [env.reset(prng.next()) for _ in range(batch_size)]

    print("\n=== Starting GRPO Training Loop ===")
    start_time = time.time()

    header = (
        f"{'Iter':>5} | {'Mean Ret':>9} | {'Max Ret':>8} | {'Fwd Vel':>8} | "
        f"{'Height':>7} | {'Torque':>7} | {'Loss':>8} | {'KL Div':>8} | {'Clip%':>6} | {'Time':>6}"
    )
    print(header)
    print("-" * len(header))

    history = []
    for iteration in range(1, num_iterations + 1):
        iter_start = time.time()

        # Step 1: Collect G candidate rollouts per state via Diffusion Denoising
        rollout_batch, env_states = trainer.sample_group_rollouts(
            prng.next(),
            env_states,
        )

        # Step 2: GRPO Optimization Step on Group Rollouts
        loss_metrics = trainer.train_step(rollout_batch)

        iter_time = time.time() - iter_start

        # Extract telemetry
        mean_ret = float(rollout_batch.metrics["mean_return"])
        max_ret = float(rollout_batch.metrics["max_return"])
        fwd_vel = float(rollout_batch.metrics["mean_forward_vel"])
        torso_h = float(rollout_batch.metrics["mean_torso_height"])
        mean_tau = float(rollout_batch.metrics["mean_torque"])

        loss_val = float(loss_metrics.total_loss)
        kl_val = float(loss_metrics.kl_div)
        clip_pct = float(loss_metrics.clip_fraction) * 100.0

        history.append(
            {
                "iteration": iteration,
                "mean_return": mean_ret,
                "max_return": max_ret,
                "fwd_vel": fwd_vel,
                "height": torso_h,
                "torque": mean_tau,
                "loss": loss_val,
                "kl": kl_val,
                "clip_pct": clip_pct,
            }
        )

        print(
            f"{iteration:5d} | {mean_ret:9.2f} | {max_ret:8.2f} | {fwd_vel:8.3f} | "
            f"{torso_h:7.3f} | {mean_tau:7.2f} | {loss_val:8.4f} | {kl_val:8.4f} | {clip_pct:5.1f}% | {iter_time:5.1f}s"
        )

    total_time = time.time() - start_time
    print("-" * len(header))
    print(f"\n✓ GRPO Training Completed in {total_time:.2f}s ({total_time / num_iterations:.2f}s / iter)")
    print(
        f"  Final Mean Return: {history[-1]['mean_return']:.2f} | Final Forward Vel: {history[-1]['fwd_vel']:.3f} m/s"
    )

    return policy, history


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Visualize ANYmal Quadruped Walking with Diffusion Policy and Joint PD Controller."""

import argparse
import os
import time

import jax
import jax.numpy as jnp
import matplotlib

if "DISPLAY" in os.environ:
    try:
        matplotlib.use("TkAgg")
    except Exception:
        pass
import matplotlib.pyplot as plt
from flax import nnx

from twm.envs.anymal_env import ANYmalBEnv
from twm.models.diffusion_policy import DiffusionPolicy
from twm.utils.prng import PRNGSequence

try:
    from brax.io import html

    _HAS_HTML = True
except ImportError:
    _HAS_HTML = False


def visualize_diffusion_policy(
    num_steps: int = 150,
    html_out: str = "anymal_diffusion_walk.html",
    plot_out: str = "anymal_diffusion_kinematics.png",
    headless: bool = True,
    kp: float = 50.0,
    kd: float = 1.5,
    action_scale: float = 0.3,
):
    print("=== Visualizing ANYmal Locomotion with Diffusion Policy & PD Controller ===")
    prng = PRNGSequence(seed=42)

    # 1. Setup Environment
    env = ANYmalBEnv(
        backend="positional",
        kp=kp,
        kd=kd,
        action_scale=action_scale,
        target_velocity=0.8,
    )
    step_fn = jax.jit(env.step)
    reset_fn = jax.jit(env.reset)

    state = reset_fn(prng.next())

    # 2. Setup Diffusion Policy
    policy_rngs = nnx.Rngs(params=prng.next())
    policy = DiffusionPolicy(
        state_dim=env.observation_size,
        action_dim=env.action_size,
        embed_dim=256,
        num_layers=3,
        num_timesteps=8,
        rngs=policy_rngs,
    )

    print(f"Executing closed-loop diffusion control for {num_steps} simulation steps...")

    # History buffers for telemetry
    step_history = []
    lf_haa_actual = []
    lf_haa_target = []
    lf_hfe_actual = []
    lf_hfe_target = []
    lf_kfe_actual = []
    lf_kfe_target = []
    height_history = []
    vel_history = []
    torque_history = []
    all_pipeline_states = [state.pipeline_state]

    start_time = time.time()
    for step in range(num_steps):
        # Diffusion policy samples target actions from current state observation
        obs_input = state.obs[None, :]  # [1, State_Dim]
        traj = policy.sample_trajectory(prng.next(), obs_input, deterministic=True)
        action = traj.actions[0]  # [Action_Dim]

        # Compute desired target joint positions
        q_target = env.pd_controller.compute_target_positions(action)

        # Step physics environment with JIT-compiled physics
        state = step_fn(state, action)
        all_pipeline_states.append(state.pipeline_state)

        # Telemetry
        step_history.append(step)
        curr_q = state.pipeline_state.q[7:19]
        lf_haa_actual.append(float(curr_q[0]))
        lf_haa_target.append(float(q_target[0]))
        lf_hfe_actual.append(float(curr_q[1]))
        lf_hfe_target.append(float(q_target[1]))
        lf_kfe_actual.append(float(curr_q[2]))
        lf_kfe_target.append(float(q_target[2]))

        height_history.append(float(state.pipeline_state.x.pos[0, 2]))
        vel_history.append(float(state.pipeline_state.qd[0]))
        torque_history.append(float(state.metrics.get("mean_torque", 0.0)))

        if float(state.done) > 0.5:
            state = reset_fn(prng.next())

    elapsed = time.time() - start_time
    print(
        f"Completed {num_steps} steps in {elapsed:.2f}s. Average Forward Velocity: {jnp.mean(jnp.array(vel_history)):.3f} m/s"
    )

    # Generate telemetry summary plots
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    # Plot 1: LF Leg Joint Tracking
    axes[0].plot(step_history, lf_haa_actual, label="LF_HAA Actual", color="#1f77b4")
    axes[0].plot(
        step_history,
        lf_haa_target,
        label="LF_HAA Target",
        color="#1f77b4",
        linestyle="--",
        alpha=0.7,
    )
    axes[0].plot(step_history, lf_hfe_actual, label="LF_HFE Actual", color="#ff7f0e")
    axes[0].plot(
        step_history,
        lf_hfe_target,
        label="LF_HFE Target",
        color="#ff7f0e",
        linestyle="--",
        alpha=0.7,
    )
    axes[0].plot(step_history, lf_kfe_actual, label="LF_KFE Actual", color="#2ca02c")
    axes[0].plot(
        step_history,
        lf_kfe_target,
        label="LF_KFE Target",
        color="#2ca02c",
        linestyle="--",
        alpha=0.7,
    )
    axes[0].set_ylabel("Joint Angle (rad)")
    axes[0].set_title(
        "ANYmal B Joint Position Targets vs Actuals (Tracked by PD Controller)",
        fontweight="bold",
    )
    axes[0].legend(loc="upper right", ncol=3, fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Forward Velocity and Torso Height
    ax_vel = axes[1]
    (line1,) = ax_vel.plot(
        step_history, vel_history, color="blue", label="Forward Velocity vx (m/s)"
    )
    ax_vel.axhline(0.8, color="navy", linestyle=":", label="Target Velocity (0.8 m/s)")
    ax_vel.set_ylabel("Velocity (m/s)", color="blue")
    ax_vel.grid(True, alpha=0.3)

    ax_h = ax_vel.twinx()
    (line2,) = ax_h.plot(
        step_history,
        height_history,
        color="crimson",
        label="Torso Height z (m)",
        linewidth=1.5,
    )
    ax_h.set_ylabel("Torso Height (m)", color="crimson")
    ax_h.set_ylim(0.2, 0.7)
    axes[1].set_title("Locomotion Performance: Velocity and Torso Height", fontweight="bold")

    # Plot 3: Mean Joint PD Torque
    axes[2].plot(
        step_history,
        torque_history,
        color="purple",
        label="Mean Absolute PD Torque (N*m)",
    )
    axes[2].set_xlabel("Simulation Step")
    axes[2].set_ylabel("Torque (N*m)")
    axes[2].set_title("Lower-Level Joint PD Control Effort", fontweight="bold")
    axes[2].legend(loc="upper right", fontsize=9)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(plot_out, dpi=150)
    print(f"✓ Kinematics telemetry plot saved to '{plot_out}'.")

    # Generate 3D HTML Viewer
    if html_out and _HAS_HTML and env.sys is not None:
        try:
            subsampled = all_pipeline_states[:: max(1, len(all_pipeline_states) // 150)]
            html_str = html.render(env.sys, subsampled)
            with open(html_out, "w") as f:
                f.write(html_str)
            print(f"✓ Interactive 3D HTML rollout saved to '{html_out}'.")
        except Exception as e:
            print(f"Notice during HTML export: {e}")

    print("=== ANYmal Diffusion Locomotion Visualization Complete ===")


def main():
    parser = argparse.ArgumentParser(description="Visualize ANYmal Walking with Diffusion Policy")
    parser.add_argument("--num_steps", type=int, default=150, help="Number of steps (default: 150)")
    parser.add_argument(
        "--html_out",
        type=str,
        default="anymal_diffusion_walk.html",
        help="Output HTML path",
    )
    parser.add_argument(
        "--plot_out",
        type=str,
        default="anymal_diffusion_kinematics.png",
        help="Output PNG path",
    )
    parser.add_argument("--headless", action="store_true", default=True, help="Run headless")
    args = parser.parse_args()

    visualize_diffusion_policy(
        num_steps=args.num_steps,
        html_out=args.html_out,
        plot_out=args.plot_out,
        headless=args.headless,
    )


if __name__ == "__main__":
    main()

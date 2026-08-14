#!/usr/bin/env python3
"""Live visualizer for ANYmal B robot in Brax.

Can render to:
1. Live GUI Window on VNC Desktop (:1) using Matplotlib or Pygame/OpenCV
2. Interactive 3D HTML Viewer saved to anymal_rollout.html
"""

import argparse
import os

import jax
import jax.numpy as jnp
import matplotlib

if "DISPLAY" in os.environ:
    try:
        matplotlib.use("TkAgg")
    except Exception:
        pass
import matplotlib.pyplot as plt

from twm.envs.anymal_env import ANYmalBEnv
from twm.utils.prng import PRNGSequence

try:
    from brax.io import html

    _HAS_HTML = True
except ImportError:
    _HAS_HTML = False


# LINT.IfChange(anymal_vis)
def visualize_anymal(
    num_steps: int = 50, html_out: str = "anymal_rollout.html", headless: bool = False
):
    print("=== Launching ANYmal B Robot Simulation in Brax ===")
    env = ANYmalBEnv(backend="positional")
    prng = PRNGSequence(seed=42)

    state = env.reset(prng.next())
    # LINT.ThenChange(//twm/envs/anymal_env.py:env_specs, //Makefile:env_targets)
    step_fn = jax.jit(env.step)
    rollout_pipeline_states = [state.pipeline_state]
    joint_angles = [state.obs[:12]]
    torso_heights = [state.pipeline_state.x.pos[0, 2]]

    print(f"Stepping ANYmal B physics simulation for {num_steps} steps...")
    for step in range(num_steps):
        # Apply sinusoidal nominal walking pattern across legs
        t = step * 0.1
        sin_act = (
            jnp.array(
                [
                    0.0,
                    jnp.sin(t),
                    -jnp.sin(t),  # LF
                    0.0,
                    jnp.sin(t + jnp.pi),
                    -jnp.sin(t + jnp.pi),  # RF
                    0.0,
                    -jnp.sin(t),
                    jnp.sin(t),  # LH
                    0.0,
                    -jnp.sin(t + jnp.pi),
                    jnp.sin(t + jnp.pi),  # RH
                ]
            )
            * 0.4
        )

        state = step_fn(state, sin_act)
        rollout_pipeline_states.append(state.pipeline_state)
        joint_angles.append(state.obs[:12])
        torso_heights.append(state.pipeline_state.x.pos[0, 2])

    # 1. Generate 3D Interactive HTML Visualization
    if _HAS_HTML and env.sys is not None:
        print(f"Generating 3D interactive HTML animation -> {html_out}...")
        try:
            html_content = html.render(env.sys, rollout_pipeline_states)
            with open(html_out, "w") as f:
                f.write(html_content)
            print(f"✓ Saved 3D Interactive HTML viewer to '{html_out}'.")
            print("  (Open in browser or view inside VNC desktop/noVNC: http://localhost:6080)")
        except Exception as e:
            print(f"HTML render notice: {e}")

    # 2. Display live GUI figure window (renders on VNC display :1)
    print("Generating joint kinematics and torso height visualization...")
    joint_angles = jnp.stack(joint_angles, axis=0)  # [T, 12]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax1.plot(joint_angles[:, :3], label=["LF_HAA", "LF_HFE", "LF_KFE"])
    ax1.set_ylabel("Joint Angle (rad)")
    ax1.set_title("ANYmal B Quadruped Kinematics & Locomotion Rollout")
    ax1.legend(loc="upper right")
    ax1.grid(True)

    ax2.plot(torso_heights, color="crimson", linewidth=2, label="Torso Height (z)")
    ax2.set_xlabel("Simulation Step")
    ax2.set_ylabel("Height (m)")
    ax2.legend(loc="upper right")
    ax2.grid(True)

    plt.tight_layout()
    plot_file = "anymal_kinematics.png"
    plt.savefig(plot_file, dpi=150)
    print(f"✓ Saved kinematics plot to '{plot_file}'.")

    # If running with active display inside VNC (DISPLAY=:1), display window
    if "DISPLAY" in os.environ and not headless:
        print(f"Displaying interactive window on {os.environ['DISPLAY']}...")
        try:
            plt.show(block=False)
            plt.pause(2)
        except Exception as e:
            print(f"Window display notice: {e}")

    print("=== ANYmal B Visualization Complete ===")


def main():
    parser = argparse.ArgumentParser(description="Visualize ANYmal B in Brax on VNC display.")
    parser.add_argument("--num_steps", type=int, default=30, help="Simulation steps")
    parser.add_argument(
        "--html_out", type=str, default="anymal_rollout.html", help="HTML 3D output file"
    )
    parser.add_argument("--headless", action="store_true", help="Run without popping GUI window")
    args = parser.parse_args()

    visualize_anymal(num_steps=args.num_steps, html_out=args.html_out, headless=args.headless)


if __name__ == "__main__":
    main()

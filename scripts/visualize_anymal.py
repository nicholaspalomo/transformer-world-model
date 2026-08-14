#!/usr/bin/env python3
"""Live visualizer for ANYmal B robot in Brax.

Can render to:
1. Live GUI Window on VNC Desktop (:1) using Matplotlib or Pygame/OpenCV
2. Interactive 3D HTML Viewer saved to anymal_rollout.html
"""

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

from twm.envs.anymal_env import ANYmalBEnv
from twm.utils.prng import PRNGSequence

try:
    from brax.io import html

    _HAS_HTML = True
except ImportError:
    _HAS_HTML = False


def visualize_anymal(
    num_steps: int = 300,
    html_out: str = "anymal_rollout.html",
    headless: bool = False,
    pause_sec: float = 60.0,
):
    print("=== Launching Continuous Rolling ANYmal B Robot Simulation in Brax ===")
    # LINT.IfChange(anymal_vis)
    env = ANYmalBEnv(backend="positional")
    prng = PRNGSequence(seed=42)

    state = env.reset(prng.next())
    # LINT.ThenChange(//Makefile:env_targets)
    step_fn = jax.jit(env.step)

    try:
        from brax.io import image as brax_image

        has_image = True
    except Exception:
        has_image = False

    # Initialize visualization figures & live rolling lines
    fig = plt.figure(figsize=(14, 7))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.2, 1])

    # Left: Live 3D Simulation Viewport
    ax_3d = fig.add_subplot(gs[:, 0])
    ax_3d.axis("off")
    ax_3d.set_title(
        "ANYmal B Continuous 3D Simulation (Brax Physics)", fontsize=13, fontweight="bold"
    )

    initial_frame = None
    if not headless and has_image and env.sys is not None and "DISPLAY" in os.environ:
        try:
            initial_frames = brax_image.render_array(
                env.sys, [state.pipeline_state], width=540, height=420
            )
            initial_frame = initial_frames[0]
        except Exception:
            initial_frame = None

    if initial_frame is not None:
        im_display = ax_3d.imshow(initial_frame)
    else:
        im_display = None
        ax_3d.text(0.5, 0.5, "Live 3D Simulation Viewport", ha="center", va="center", fontsize=14)

    # Right Top: Rolling Joint Kinematics
    ax1 = fig.add_subplot(gs[0, 1])
    (line_haa,) = ax1.plot([], [], label="LF_HAA", color="#1f77b4")
    (line_hfe,) = ax1.plot([], [], label="LF_HFE", color="#ff7f0e")
    (line_kfe,) = ax1.plot([], [], label="LF_KFE", color="#2ca02c")
    ax1.set_ylabel("Joint Angle (rad)")
    ax1.set_title("Live Quadruped Kinematics (LF Leg)", fontsize=11)
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-1.5, 1.5)

    # Right Bottom: Rolling Torso Height
    ax2 = fig.add_subplot(gs[1, 1], sharex=ax1)
    (line_height,) = ax2.plot([], [], color="crimson", linewidth=2, label="Torso Height (z)")
    ax2.set_xlabel("Simulation Step")
    ax2.set_ylabel("Height (m)")
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0.2, 0.7)

    plt.tight_layout()

    # Track rolling telemetry
    step_history = []
    haa_history = []
    hfe_history = []
    kfe_history = []
    height_history = []
    all_pipeline_states = [state.pipeline_state]

    global_step = 0
    start_time = time.time()
    chunk_size = 15

    print(f"Streaming live continuous physics simulation (headless={headless})...")

    if "DISPLAY" in os.environ and not headless:
        try:
            plt.show(block=False)
        except Exception as e:
            print(f"Window display notice: {e}")

    max_steps = num_steps if headless else 10000000
    while global_step < max_steps:
        if not headless and pause_sec > 0 and (time.time() - start_time >= pause_sec):
            break
        chunk_states = []
        for _ in range(chunk_size):
            t = global_step * 0.1
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
            chunk_states.append(state.pipeline_state)
            all_pipeline_states.append(state.pipeline_state)

            step_history.append(global_step)
            haa_history.append(float(state.obs[0]))
            hfe_history.append(float(state.obs[1]))
            kfe_history.append(float(state.obs[2]))
            height_history.append(float(state.pipeline_state.x.pos[0, 2]))

            global_step += 1

        # Render chunk frames
        chunk_frames = []
        if not headless and has_image and env.sys is not None and "DISPLAY" in os.environ:
            try:
                chunk_frames = brax_image.render_array(env.sys, chunk_states, width=540, height=420)
            except Exception:
                chunk_frames = []

        # Stream chunk frames & update rolling line graphs
        for i, frame in enumerate(chunk_frames):
            if im_display is not None:
                im_display.set_data(frame)

            # Update rolling data window (last 150 steps)
            curr_idx = len(step_history) - chunk_size + i + 1
            start_window = max(0, curr_idx - 150)
            xs = step_history[start_window:curr_idx]

            line_haa.set_data(xs, haa_history[start_window:curr_idx])
            line_hfe.set_data(xs, hfe_history[start_window:curr_idx])
            line_kfe.set_data(xs, kfe_history[start_window:curr_idx])
            line_height.set_data(xs, height_history[start_window:curr_idx])

            if len(xs) > 1:
                ax1.set_xlim(xs[0], xs[-1] + 5)

            if "DISPLAY" in os.environ and not headless:
                fig.canvas.draw_idle()
                plt.pause(0.04)

        if pause_sec > 0 and (time.time() - start_time >= pause_sec):
            break

    # Save final snapshot
    plot_file = "anymal_kinematics.png"
    plt.savefig(plot_file, dpi=150)
    print(f"✓ Saved visual snapshot to '{plot_file}'.")

    # Optionally save full trajectory HTML
    if html_out and _HAS_HTML and env.sys is not None:
        try:
            html_content = html.render(
                env.sys, all_pipeline_states[:: max(1, len(all_pipeline_states) // 200)]
            )
            with open(html_out, "w") as f:
                f.write(html_content)
        except Exception:
            pass

    print("=== ANYmal B Continuous Simulation Complete ===")


def main():
    parser = argparse.ArgumentParser(description="Visualize ANYmal B in Brax on VNC display.")
    parser.add_argument(
        "--num_steps", type=int, default=200, help="Simulation steps (default: 200)"
    )
    parser.add_argument(
        "--html_out", type=str, default="anymal_rollout.html", help="HTML 3D output file"
    )
    parser.add_argument("--headless", action="store_true", help="Run without popping GUI window")
    parser.add_argument(
        "--pause_sec",
        type=float,
        default=60.0,
        help="Seconds to display figure on VNC (default: 60.0)",
    )
    args = parser.parse_args()

    visualize_anymal(
        num_steps=args.num_steps,
        html_out=args.html_out,
        headless=args.headless,
        pause_sec=args.pause_sec,
    )


if __name__ == "__main__":
    main()

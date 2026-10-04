"""Low-level Joint PD Controller for ANYmal B quadruped robot."""

from dataclasses import dataclass, field
from typing import NamedTuple

import jax
import jax.numpy as jnp

# Nominal default standing joint angles for ANYmal B
# Order: [LF_HAA, LF_HFE, LF_KFE, RF_HAA, RF_HFE, RF_KFE, LH_HAA, LH_HFE, LH_KFE, RH_HAA, RH_HFE, RH_KFE]
NOMINAL_JOINT_POS = jnp.array(
    [
        0.0,
        0.4,
        -0.8,  # LF leg
        0.0,
        0.4,
        -0.8,  # RF leg
        0.0,
        -0.4,
        0.8,  # LH leg
        0.0,
        -0.4,
        0.8,  # RH leg
    ],
    dtype=jnp.float32,
)

# Joint position limits (min, max) in radians
JOINT_LOWER_LIMITS = jnp.array(
    [-0.6, -1.5, -2.5, -0.6, -1.5, -2.5, -0.6, -1.5, -2.5, -0.6, -1.5, -2.5],
    dtype=jnp.float32,
)
JOINT_UPPER_LIMITS = jnp.array(
    [0.6, 1.5, 2.5, 0.6, 1.5, 2.5, 0.6, 1.5, 2.5, 0.6, 1.5, 2.5],
    dtype=jnp.float32,
)


class PDControlOutput(NamedTuple):
    """Container for PD control output and telemetry."""

    torques: jax.Array  # Applied joint torques (Nm) [12]
    q_target: jax.Array  # Desired joint angles (rad) [12]
    pos_error: jax.Array  # Position tracking error (rad) [12]
    vel_error: jax.Array  # Velocity tracking error (rad/s) [12]


@dataclass
class JointPDController:
    """Proportional-Derivative joint impedance controller.

    Maps high-level policy actions (residual joint angle targets) and measured joint
    states (q, qd) to actuator torques:
        q_target = q_nominal + action_scale * action
        tau = clip(kp * (q_target - q) - kd * qd, -tau_max, tau_max)
    """

    kp: float = 50.0  # Proportional stiffness gain (N*m/rad)
    kd: float = 1.5  # Derivative damping gain (N*m*s/rad)
    tau_max: float = 40.0  # Peak motor torque limit (N*m)
    action_scale: float = 0.3  # Action multiplier in radians around nominal pose
    nominal_qpos: jax.Array = field(default_factory=lambda: NOMINAL_JOINT_POS)

    def compute_target_positions(self, action: jax.Array) -> jax.Array:
        """Convert normalized action [-1, 1] into target joint angles q_target."""
        # Action is residual displacement around nominal pose
        action = jnp.clip(action, -1.0, 1.0)
        q_target = self.nominal_qpos + action * self.action_scale
        # Enforce physical mechanical joint limits
        return jnp.clip(q_target, JOINT_LOWER_LIMITS, JOINT_UPPER_LIMITS)

    def compute_torques(
        self,
        q_target: jax.Array,
        current_q: jax.Array,
        current_qd: jax.Array,
        target_qd: jax.Array | None = None,
    ) -> PDControlOutput:
        """Calculate joint torques via PD impedance control law.

        Args:
            q_target: Desired joint positions [12] (rad)
            current_q: Measured joint positions [12] (rad)
            current_qd: Measured joint velocities [12] (rad/s)
            target_qd: Optional desired joint velocities [12] (rad/s), default 0

        Returns:
            PDControlOutput containing clipped torques and error signals.
        """
        if target_qd is None:
            target_qd = jnp.zeros_like(current_qd)

        pos_error = q_target - current_q
        vel_error = target_qd - current_qd

        raw_torques = self.kp * pos_error + self.kd * vel_error
        clipped_torques = jnp.clip(raw_torques, -self.tau_max, self.tau_max)

        return PDControlOutput(
            torques=clipped_torques,
            q_target=q_target,
            pos_error=pos_error,
            vel_error=vel_error,
        )

    def step(
        self,
        action: jax.Array,
        current_q: jax.Array,
        current_qd: jax.Array,
    ) -> PDControlOutput:
        """High-level one-step execution: Action -> Target -> Torques."""
        q_target = self.compute_target_positions(action)
        return self.compute_torques(q_target, current_q, current_qd)

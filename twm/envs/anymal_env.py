"""Custom Brax environment for ANYmal B quadruped robot with lower-level PD control."""

import os
from typing import Any

import jax
import jax.numpy as jnp

from twm.envs.pd_controller import NOMINAL_JOINT_POS, JointPDController, PDControlOutput

try:
    import brax  # noqa: F401
    from brax.envs.base import PipelineEnv, State
    from brax.io import mjcf

    _HAS_BRAX = True
except ImportError:
    _HAS_BRAX = False
    PipelineEnv = object
    State = object


class ANYmalBEnv(PipelineEnv if _HAS_BRAX else object):
    """JAX-native Brax environment for ANYmal B quadruped locomotion with lower-level PD control."""

    def __init__(
        self,
        xml_path: str | None = None,
        backend: str = "positional",
        n_frames: int = 4,
        target_velocity: float = 0.8,
        kp: float = 50.0,
        kd: float = 1.5,
        action_scale: float = 0.3,
        **kwargs,
    ):
        if xml_path is None:
            curr_dir = os.path.dirname(os.path.abspath(__file__))
            xml_path = os.path.join(curr_dir, "../../assets/anybotics_anymal_b/scene.xml")

        self._backend_name = backend
        self.target_velocity = target_velocity
        self.pd_controller = JointPDController(
            kp=kp,
            kd=kd,
            tau_max=40.0,
            action_scale=action_scale,
            nominal_qpos=NOMINAL_JOINT_POS,
        )

        if _HAS_BRAX and os.path.exists(xml_path):
            sys = mjcf.load(xml_path)
            super().__init__(sys=sys, backend=backend, n_frames=n_frames, **kwargs)
            self._obs_size = self._get_obs_dim()
            self._act_size = self.sys.act_size()
        else:
            self.sys = None
            # ANYmal B: 12 actuated joints, 35 observation dimensions
            self._obs_size = 35
            self._act_size = 12

    # LINT.IfChange(env_specs)
    @property
    def observation_size(self) -> int:
        return self._obs_size

    @property
    def action_size(self) -> int:
        return self._act_size

    def _get_obs_dim(self) -> int:
        # qpos[7:] (12 joints) + qvel[6:] (12 joint vels) + base_z (1) + base_rot (4) + base_linvel (3) + base_angvel (3)
        # Standard compact ANYmal observation
        return 12 + 12 + 1 + 4 + 3 + 3

    # LINT.ThenChange(//twm/envs/brax_wrapper.py:env_registry, //configs/env_anymal_b.yaml:env_config, //scripts/visualize_anymal.py:anymal_vis)

    def reset(self, rng: jax.Array) -> Any:
        """Reset environment to initial nominal standing pose with small random perturbation."""
        if not _HAS_BRAX or self.sys is None:
            obs = jax.random.normal(rng, (self.observation_size,))
            return None, obs

        rng_init, rng_noise = jax.random.split(rng)

        # Full nominal qpos (7 root coords + 12 joint angles)
        nominal_qpos = jnp.concatenate(
            [
                jnp.array([0.0, 0.0, 0.55, 1.0, 0.0, 0.0, 0.0], dtype=jnp.float32),
                NOMINAL_JOINT_POS,
            ]
        )

        # Small random perturbation on initial joint angles
        noise = jax.random.uniform(rng_noise, (12,), minval=-0.04, maxval=0.04)
        q = nominal_qpos.at[7:19].add(noise)
        qd = jnp.zeros(self.sys.qd_size(), dtype=jnp.float32)

        pipeline_state = self.pipeline_init(q, qd)
        obs = self._get_obs(pipeline_state)
        reward = jnp.zeros((), dtype=jnp.float32)
        done = jnp.zeros((), dtype=jnp.float32)
        metrics = {
            "forward_vel": jnp.zeros((), dtype=jnp.float32),
            "torso_height": jnp.array(0.55, dtype=jnp.float32),
            "tracking_reward": jnp.zeros((), dtype=jnp.float32),
            "torque_penalty": jnp.zeros((), dtype=jnp.float32),
            "upright_bonus": jnp.zeros((), dtype=jnp.float32),
        }

        return State(pipeline_state, obs, reward, done, metrics)

    def step(self, state: Any, action: jax.Array, rng: jax.Array | None = None) -> Any:
        """Step physics simulation forward given 12 joint position targets tracked by PD controller."""
        if not _HAS_BRAX or self.sys is None:
            next_obs = jax.random.normal(rng or jax.random.PRNGKey(0), (self.observation_size,))
            reward = jnp.array(1.0, dtype=jnp.float32)
            done = jnp.array(0.0, dtype=jnp.float32)
            return None, next_obs, reward, done, {}

        # 1. Compute target joint positions q_target from action
        q_target = self.pd_controller.compute_target_positions(action)

        # 2. Extract current joint positions and velocities
        current_q = state.pipeline_state.q[7:19]
        current_qd = state.pipeline_state.qd[6:18]

        # 3. Compute PD control output and torques
        pd_out: PDControlOutput = self.pd_controller.compute_torques(
            q_target, current_q, current_qd
        )

        # 4. Step physics simulation with joint position targets
        pipeline_state = self.pipeline_step(state.pipeline_state, q_target)
        obs = self._get_obs(pipeline_state)

        # 5. Reward computation for ANYmal quadruped walking
        # Kinematic and state quantities
        forward_vel = pipeline_state.qd[0]  # Base x-velocity (m/s)
        lateral_vel = pipeline_state.qd[1]  # Base y-velocity (m/s)
        vertical_vel = pipeline_state.qd[2]  # Base z-velocity (m/s)
        ang_vel = pipeline_state.qd[3:6]  # Base angular velocities (roll, pitch, yaw)

        torso_height = pipeline_state.x.pos[0, 2]  # Base link z-position (m)
        base_quat = pipeline_state.q[3:7]  # [qw, qx, qy, qz]

        # Upright metric: projection of base z-axis onto global vertical
        # qw^2 - qx^2 - qy^2 + qz^2 (dot product of body z-axis with world z-axis)
        qx, qy = base_quat[1], base_quat[2]
        upright_proj = 1.0 - 2.0 * (qx * qx + qy * qy)

        # Reward components
        # (a) Target linear velocity tracking reward (Gaussian kernel)
        vel_error = forward_vel - self.target_velocity
        tracking_reward = jnp.exp(-4.0 * jnp.square(vel_error))

        # (b) Upright orientation reward
        upright_bonus = jnp.clip(upright_proj, 0.0, 1.0)

        # (c) Base height maintenance (target ~0.50m)
        height_reward = jnp.exp(-25.0 * jnp.square(torso_height - 0.50))

        # (d) Penalties
        lateral_penalty = jnp.square(lateral_vel)
        vertical_penalty = jnp.square(vertical_vel)
        ang_vel_penalty = jnp.sum(jnp.square(ang_vel))
        torque_penalty = jnp.mean(jnp.square(pd_out.torques / self.pd_controller.tau_max))
        tracking_err_penalty = jnp.mean(jnp.square(pd_out.pos_error))

        # Combined composite walking reward
        reward = (
            1.5 * tracking_reward
            + 0.5 * upright_bonus
            + 0.5 * height_reward
            - 0.3 * lateral_penalty
            - 0.2 * vertical_penalty
            - 0.1 * ang_vel_penalty
            - 0.05 * torque_penalty
            - 0.1 * tracking_err_penalty
        )

        # Termination conditions (robot fallen, upside down, or height out of bounds)
        is_fallen = (torso_height < 0.25) | (torso_height > 0.85) | (upright_proj < 0.3)
        done = jnp.where(is_fallen, 1.0, 0.0)

        metrics = {
            "forward_vel": forward_vel,
            "torso_height": torso_height,
            "tracking_reward": tracking_reward,
            "torque_penalty": torque_penalty,
            "upright_bonus": upright_bonus,
            "mean_torque": jnp.mean(jnp.abs(pd_out.torques)),
        }

        return state.replace(
            pipeline_state=pipeline_state, obs=obs, reward=reward, done=done, metrics=metrics
        )

    def _get_obs(self, pipeline_state: Any) -> jax.Array:
        """Extract continuous observation vector for ANYmal control policy."""
        # 12 joint positions relative to nominal standing pose
        joint_pos = pipeline_state.q[7:19] - NOMINAL_JOINT_POS
        # 12 joint velocities
        joint_vel = pipeline_state.qd[6:18]
        # Root height (z)
        base_z = pipeline_state.q[2:3]
        # Base orientation quaternion
        base_quat = pipeline_state.q[3:7]
        # Base linear and angular velocity
        base_linvel = pipeline_state.qd[0:3]
        base_angvel = pipeline_state.qd[3:6]

        return jnp.concatenate(
            [
                joint_pos,
                joint_vel,
                base_z,
                base_quat,
                base_linvel,
                base_angvel,
            ]
        )

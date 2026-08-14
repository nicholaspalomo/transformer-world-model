"""Custom Brax environment for ANYmal B quadruped robot."""

import os
from typing import Any

import jax
import jax.numpy as jnp

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
    """JAX-native Brax environment for ANYmal B quadruped locomotion."""

    def __init__(
        self,
        xml_path: str | None = None,
        backend: str = "positional",
        n_frames: int = 4,
        **kwargs,
    ):
        if xml_path is None:
            # Default to bundled repository asset
            curr_dir = os.path.dirname(os.path.abspath(__file__))
            xml_path = os.path.join(curr_dir, "../../assets/anybotics_anymal_b/scene.xml")

        self._backend_name = backend

        if _HAS_BRAX and os.path.exists(xml_path):
            sys = mjcf.load(xml_path)
            super().__init__(sys=sys, backend=backend, n_frames=n_frames, **kwargs)
            self._obs_size = self._get_obs_dim()
            self._act_size = self.sys.act_size()
        else:
            self.sys = None
            # ANYmal B: 12 actuated joint motors, 12 qpos joints + 12 qvel joints + base orientation (37 total obs)
            self._obs_size = 37
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
        """Reset environment to initial nominal standing pose."""
        if not _HAS_BRAX or self.sys is None:
            obs = jax.random.normal(rng, (self.observation_size,))
            return None, obs

        rng_init, rng_noise = jax.random.split(rng)

        # Nominal default standing joint positions for ANYmal B
        # Joint order: LF (HAA, HFE, KFE), RF (HAA, HFE, KFE), LH (HAA, HFE, KFE), RH (HAA, HFE, KFE)
        nominal_qpos = jnp.array(
            [
                0.0,
                0.0,
                0.55,  # Root pos (x, y, z)
                1.0,
                0.0,
                0.0,
                0.0,  # Root quat (w, x, y, z)
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
            ]
        )

        # Small perturbation on initial joint angles
        noise = jax.random.uniform(rng_noise, (12,), minval=-0.05, maxval=0.05)
        q = nominal_qpos.at[7:].add(noise)
        qd = jnp.zeros(self.sys.qd_size())

        pipeline_state = self.pipeline_init(q, qd)
        obs = self._get_obs(pipeline_state)
        reward = jnp.zeros(())
        done = jnp.zeros(())
        metrics = {"forward_vel": jnp.zeros(())}

        return State(pipeline_state, obs, reward, done, metrics)

    def step(self, state: Any, action: jax.Array, rng: jax.Array | None = None) -> Any:
        """Step physics simulation forward given 12 joint torque/target actions."""
        if not _HAS_BRAX or self.sys is None:
            next_obs = jax.random.normal(rng or jax.random.PRNGKey(0), (self.observation_size,))
            reward = jnp.array(1.0, dtype=jnp.float32)
            done = jnp.array(0.0, dtype=jnp.float32)
            return None, next_obs, reward, done, {}

        # Clip action limits [-1, 1]
        action = jnp.clip(action, -1.0, 1.0)
        pipeline_state = self.pipeline_step(state.pipeline_state, action)
        obs = self._get_obs(pipeline_state)

        # Rewards: forward velocity along x-axis + upright bonus - joint torque effort
        forward_vel = pipeline_state.qd[0]
        torso_height = pipeline_state.x.pos[0, 2]  # base link z position
        torque_cost = jnp.sum(jnp.square(action))

        # Reward shaping (WIP for locomotion)
        reward = forward_vel * 1.5 + (torso_height > 0.3) * 0.5 - 0.01 * torque_cost

        # Terminate if robot falls
        done = jnp.where(torso_height < 0.2, 1.0, 0.0)

        metrics = {"forward_vel": forward_vel, "torso_height": torso_height}
        return state.replace(
            pipeline_state=pipeline_state, obs=obs, reward=reward, done=done, metrics=metrics
        )

    def _get_obs(self, pipeline_state: Any) -> jax.Array:
        """Extract continuous observation vector for Transformer World Model."""
        # 12 joint positions
        joint_pos = pipeline_state.q[7:19]
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

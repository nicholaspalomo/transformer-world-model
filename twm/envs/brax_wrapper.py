"""Pure JAX wrapper around Brax environments for state & transition collection."""

from typing import Any

import jax
import jax.numpy as jnp

try:
    import brax.envs as brax_envs

    _HAS_BRAX = True
except ImportError:
    _HAS_BRAX = False


class BraxEnvWrapper:
    """JAX-native wrapper for Brax continuous control environments."""

    def __init__(
        self, env_name: str = "ant", backend: str = "positional", episode_length: int = 1000
    ):
        self.env_name = env_name

        # Simulation algorithm used to integrate the physics
        self.backend = backend

        # Number of simulation steps per environment step
        self.episode_length = episode_length
        self._env = None

        # LINT.IfChange(env_registry)
        if env_name.lower() in ("anymal", "anymal_b", "anybotics_anymal_b"):
            from twm.envs.anymal_env import ANYmalBEnv

            self._env = ANYmalBEnv(backend="positional")
            self.observation_size = self._env.observation_size
            self.action_size = self._env.action_size
            self._reset_fn = jax.jit(self._env.reset)
            self._step_fn = jax.jit(self._env.step)
        elif _HAS_BRAX:
            self._env = brax_envs.create(
                env_name=env_name, backend=backend, episode_length=episode_length
            )
            self.observation_size = self._env.observation_size
            self.action_size = self._env.action_size
            self._reset_fn = jax.jit(self._env.reset)
            self._step_fn = jax.jit(self._env.step)
        else:
            # Fallback mock dimensions for continuous control testing (e.g. Ant)
            self.observation_size = 27
            self.action_size = 8
            self._reset_fn = None
            self._step_fn = None
        # LINT.ThenChange(//twm/envs/anymal_env.py:env_specs, //configs/env_brax_ant.yaml:env_config, //configs/env_anymal_b.yaml:env_config, //Makefile:env_targets, //scripts/01_collect_data.py:env_args, //scripts/03_evaluate_mppi.py:env_args)

    def reset(self, rng: jax.Array) -> tuple[Any, jax.Array]:
        """Reset environment with PRNGKey."""
        if _HAS_BRAX and self._env is not None:
            state = self._reset_fn(rng)
            return state, state.obs
        else:
            # Dummy state and observation
            obs = jax.random.normal(rng, (self.observation_size,))
            return None, obs

    def step(
        self, state: Any, action: jax.Array, rng: jax.Array
    ) -> tuple[Any, jax.Array, jax.Array, jax.Array, dict[str, Any]]:
        """Step environment given state and action."""
        if _HAS_BRAX and self._env is not None:
            next_state = self._step_fn(state, action)
            return (
                next_state,
                next_state.obs,
                next_state.reward,
                next_state.done,
                next_state.metrics,
            )
        else:
            # Dummy transition
            next_obs = jax.random.normal(rng, (self.observation_size,))
            reward = jnp.array(1.0, dtype=jnp.float32)
            done = jnp.array(0.0, dtype=jnp.float32)
            return None, next_obs, reward, done, {}

"""Pure JAX wrapper around Brax environments for state & transition collection."""

from typing import Tuple, Dict, Any, Optional
import jax
import jax.numpy as jnp

try:
    import brax.envs as brax_envs
    _HAS_BRAX = True
except ImportError:
    _HAS_BRAX = False


class BraxEnvWrapper:
    """JAX-native wrapper for Brax continuous control environments."""

    def __init__(self, env_name: str = "ant", backend: str = "positional", episode_length: int = 1000):
        self.env_name = env_name

        # Simulation algorithm used to integrate the physics
        self.backend = backend
        
        # Number of simulation steps per environment step
        self.episode_length = episode_length
        self._env = None

        if env_name.lower() in ("anymal", "anymal_b", "anybotics_anymal_b"):
            from twm.envs.anymal_env import ANYmalBEnv
            self._env = ANYmalBEnv(backend="generalized")
            self.observation_size = self._env.observation_size
            self.action_size = self._env.action_size
        elif _HAS_BRAX:
            self._env = brax_envs.create(env_name=env_name, backend=backend, episode_length=episode_length)
            self.observation_size = self._env.observation_size
            self.action_size = self._env.action_size
        else:
            # Fallback mock dimensions for continuous control testing (e.g. Ant)
            self.observation_size = 27
            self.action_size = 8

    def reset(self, rng: jax.Array) -> Tuple[Any, jax.Array]:
        """Reset environment with PRNGKey."""
        if _HAS_BRAX and self._env is not None:
            state = self._env.reset(rng)
            return state, state.obs
        else:
            # Dummy state and observation
            obs = jax.random.normal(rng, (self.observation_size,))
            return None, obs

    def step(self, state: Any, action: jax.Array, rng: jax.Array) -> Tuple[Any, jax.Array, jax.Array, jax.Array, Dict[str, Any]]:
        """Step environment given state and action."""
        if _HAS_BRAX and self._env is not None:
            next_state = self._env.step(state, action)
            return next_state, next_state.obs, next_state.reward, next_state.done, next_state.metrics
        else:
            # Dummy transition
            next_obs = jax.random.normal(rng, (self.observation_size,))
            reward = jnp.array(1.0, dtype=jnp.float32)
            done = jnp.array(0.0, dtype=jnp.float32)
            return None, next_obs, reward, done, {}

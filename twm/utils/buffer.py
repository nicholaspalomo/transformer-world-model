"""JAX-native trajectory replay buffer integration for sequence sampling."""

from typing import Dict, Tuple
import jax
import jax.numpy as jnp

try:
    import flashbax as fbx
    _HAS_FLASHBAX = True
except ImportError:
    _HAS_FLASHBAX = False


class TrajectoryReplayBuffer:
    """JAX-native replay buffer storing (s, a, r, s') sequences of window size K."""

    def __init__(self, max_capacity: int = 100000, state_dim: int = 27, action_dim: int = 8):
        self.max_capacity = max_capacity
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.states = jnp.zeros((max_capacity, state_dim), dtype=jnp.float32)
        self.actions = jnp.zeros((max_capacity, action_dim), dtype=jnp.float32)
        self.rewards = jnp.zeros((max_capacity,), dtype=jnp.float32)
        self.next_states = jnp.zeros((max_capacity, state_dim), dtype=jnp.float32)
        self.dones = jnp.zeros((max_capacity,), dtype=jnp.float32)

        self.ptr = 0
        self.size = 0

    def add(self, state: jax.Array, action: jax.Array, reward: float, next_state: jax.Array, done: float):
        """Add single step transition."""
        idx = self.ptr
        self.states = self.states.at[idx].set(state)
        self.actions = self.actions.at[idx].set(action)
        self.rewards = self.rewards.at[idx].set(reward)
        self.next_states = self.next_states.at[idx].set(next_state)
        self.dones = self.dones.at[idx].set(done)

        self.ptr = (self.ptr + 1) % self.max_capacity
        self.size = min(self.size + 1, self.max_capacity)

    def sample_sequences(self, rng: jax.Array, batch_size: int = 32, seq_len: int = 32) -> Dict[str, jax.Array]:
        """Sample batch of sequential trajectories of length seq_len.

        Returns dict:
            'states': [Batch, Seq_Len, State_Dim]
            'actions': [Batch, Seq_Len, Action_Dim]
            'rewards': [Batch, Seq_Len]
            'next_states': [Batch, Seq_Len, State_Dim]
            'dones': [Batch, Seq_Len]
        """
        valid_max = self.size - seq_len
        assert valid_max > 0, f"Buffer size ({self.size}) must be greater than seq_len ({seq_len})"

        start_indices = jax.random.randint(rng, (batch_size,), 0, valid_max)

        def get_seq(start_idx):
            idx_range = start_idx + jnp.arange(seq_len)
            return {
                'states': self.states[idx_range],
                'actions': self.actions[idx_range],
                'rewards': self.rewards[idx_range],
                'next_states': self.next_states[idx_range],
                'dones': self.dones[idx_range],
            }

        return jax.vmap(get_seq)(start_indices)

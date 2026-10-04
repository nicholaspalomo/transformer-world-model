"""JAX-native trajectory replay buffer integration for sequence sampling."""

import os

import jax
import jax.numpy as jnp
import numpy as np

try:
    import flashbax as fbx  # noqa: F401

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

    def add(
        self, state: jax.Array, action: jax.Array, reward: float, next_state: jax.Array, done: float
    ):
        """Add single step transition."""
        idx = self.ptr
        self.states = self.states.at[idx].set(state)
        self.actions = self.actions.at[idx].set(action)
        self.rewards = self.rewards.at[idx].set(reward)
        self.next_states = self.next_states.at[idx].set(next_state)
        self.dones = self.dones.at[idx].set(done)

        self.ptr = (self.ptr + 1) % self.max_capacity
        self.size = min(self.size + 1, self.max_capacity)

    def sample_sequences(
        self, rng: jax.Array, batch_size: int = 32, seq_len: int = 32
    ) -> dict[str, jax.Array]:
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
                "states": self.states[idx_range],
                "actions": self.actions[idx_range],
                "rewards": self.rewards[idx_range],
                "next_states": self.next_states[idx_range],
                "dones": self.dones[idx_range],
            }

        return jax.vmap(get_seq)(start_indices)

    def save(self, filepath: str) -> str:
        """Save populated replay buffer transitions to compressed npz file."""
        parent_dir = os.path.dirname(filepath)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

        np.savez_compressed(
            filepath,
            states=np.asarray(self.states[: self.size]),
            actions=np.asarray(self.actions[: self.size]),
            rewards=np.asarray(self.rewards[: self.size]),
            next_states=np.asarray(self.next_states[: self.size]),
            dones=np.asarray(self.dones[: self.size]),
            size=np.array(self.size),
            ptr=np.array(self.ptr),
            state_dim=np.array(self.state_dim),
            action_dim=np.array(self.action_dim),
        )
        return os.path.abspath(filepath)

    def load(self, filepath: str) -> None:
        """Load transitions into buffer in-place from compressed npz file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Replay buffer file does not exist: {filepath}")

        data = np.load(filepath)
        loaded_size = int(data["size"])
        self.size = min(loaded_size, self.max_capacity)
        self.ptr = int(data.get("ptr", self.size % self.max_capacity))
        self.states = self.states.at[: self.size].set(data["states"][: self.size])
        self.actions = self.actions.at[: self.size].set(data["actions"][: self.size])
        self.rewards = self.rewards.at[: self.size].set(data["rewards"][: self.size])
        self.next_states = self.next_states.at[: self.size].set(data["next_states"][: self.size])
        self.dones = self.dones.at[: self.size].set(data["dones"][: self.size])

    @classmethod
    def load_from_file(
        cls, filepath: str, max_capacity: int | None = None
    ) -> "TrajectoryReplayBuffer":
        """Instantiate a TrajectoryReplayBuffer directly from a saved npz file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Replay buffer file does not exist: {filepath}")

        data = np.load(filepath)
        state_dim = int(data["state_dim"])
        action_dim = int(data["action_dim"])
        loaded_size = int(data["size"])
        capacity = max_capacity if max_capacity is not None else max(loaded_size + 1000, 10000)

        buf = cls(max_capacity=capacity, state_dim=state_dim, action_dim=action_dim)
        buf.load(filepath)
        return buf

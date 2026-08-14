"""Pure JAX PRNG key management helpers."""

import jax


class PRNGSequence:
    """PRNG Key generator producing unique PRNG keys sequentially."""

    def __init__(self, seed: int = 42):
        self._key = jax.random.PRNGKey(seed)

    def next(self) -> jax.Array:
        """Generate and return next PRNG key."""
        self._key, subkey = jax.random.split(self._key)
        return subkey

    def current(self) -> jax.Array:
        """Return current PRNG key."""
        return self._key

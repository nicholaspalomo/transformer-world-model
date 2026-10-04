"""Flax NNX model checkpointing utilities using pure compressed NumPy storage."""

import os
from typing import Any

import jax.numpy as jnp
import numpy as np
from flax import nnx
from flax.nnx.statelib import State


def save_checkpoint(model: nnx.Module, filepath: str) -> str:
    """Save Flax NNX model state to a compressed npz file.

    Args:
        model: Flax NNX Module instance to save.
        filepath: Target filepath (e.g. 'checkpoints/model.npz').

    Returns:
        The resolved absolute path of the saved checkpoint.
    """
    parent_dir = os.path.dirname(filepath)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    _, state = nnx.split(model)
    flat_state: dict[str, Any] = {}

    for path, var_state in state.flat_state():
        key = "/".join(str(p) for p in path)
        val = getattr(var_state, "value", var_state)
        flat_state[key] = np.asarray(val)

    np.savez_compressed(filepath, **flat_state)
    return os.path.abspath(filepath)


def load_checkpoint(model: nnx.Module, filepath: str) -> None:
    """Load and update Flax NNX model state in-place from a compressed npz file.

    Args:
        model: Flax NNX Module instance to update with restored weights.
        filepath: Source filepath of the checkpoint.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint file does not exist: {filepath}")

    data = np.load(filepath)
    _, state = nnx.split(model)
    flat_items = []

    for path, var_state in state.flat_state():
        key = "/".join(str(p) for p in path)
        if key not in data:
            raise KeyError(f"Missing parameter key in checkpoint: {key}")
        loaded_arr = jnp.array(data[key])
        current_val = getattr(var_state, "value", var_state)
        if current_val.shape != loaded_arr.shape:
            raise ValueError(
                f"Checkpoint parameter shape mismatch for '{key}': "
                f"expected {current_val.shape}, got {loaded_arr.shape}"
            )
        if hasattr(var_state, "value"):
            var_state.value = loaded_arr
            flat_items.append((path, var_state))
        else:
            flat_items.append((path, loaded_arr))

    new_state = State.from_flat_path(flat_items)
    nnx.update(model, new_state)

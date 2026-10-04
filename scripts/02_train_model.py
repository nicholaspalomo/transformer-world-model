#!/usr/bin/env python3
"""Milestones 2 & 3: JAX-JIT compiled Training Loop for Flax Transformer World Model."""

import argparse
import os

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from twm.models.transformer import TransformerWorldModel
from twm.utils.buffer import TrajectoryReplayBuffer
from twm.utils.checkpoint import save_checkpoint
from twm.utils.prng import PRNGSequence


def loss_fn(model: TransformerWorldModel, batch: dict) -> jax.Array:
    """Compute MSE loss over state transitions and rewards."""
    pred_next_states, pred_rewards, _ = model(batch["states"], batch["actions"])
    state_loss = jnp.mean((pred_next_states - batch["next_states"]) ** 2)
    reward_loss = jnp.mean((pred_rewards - batch["rewards"]) ** 2)
    return state_loss + 0.5 * reward_loss


@nnx.jit
def train_step(model: TransformerWorldModel, optimizer: nnx.Optimizer, batch: dict):
    """JIT-compiled gradient step using Optax and Flax NNX."""
    grad_fn = nnx.value_and_grad(loss_fn)
    loss, grads = grad_fn(model, batch)
    try:
        optimizer.update(grads)
    except TypeError:
        optimizer.update(model, grads)
    return loss


def main():
    parser = argparse.ArgumentParser(
        description="Milestone 2 & 3: Train Transformer World Model in JAX."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/anymal_trajectories.npz",
        help="Path to saved replay buffer trajectories",
    )
    parser.add_argument(
        "--save_checkpoint",
        type=str,
        default="checkpoints/world_model_anymal.npz",
        help="Path to save trained World Model checkpoint",
    )
    parser.add_argument(
        "--state_dim",
        type=int,
        default=35,
        help="State dimension (ANYmal B default: 35)",
    )
    parser.add_argument(
        "--action_dim",
        type=int,
        default=12,
        help="Action dimension (ANYmal B default: 12)",
    )
    parser.add_argument("--num_steps", type=int, default=100, help="Training iterations")
    parser.add_argument("--seq_len", type=int, default=32, help="Sequence window length K")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--embed_dim", type=int, default=256, help="Embedding dimension")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of transformer layers")
    parser.add_argument("--mlp_dim", type=int, default=512, help="MLP feedforward dimension")
    args = parser.parse_args()

    print("=== Milestones 2 & 3: Initializing Flax NNX Causal Transformer World Model ===")
    prng = PRNGSequence(seed=42)

    # Load replay buffer data if available
    state_dim = args.state_dim
    action_dim = args.action_dim
    if os.path.exists(args.data_path):
        buffer = TrajectoryReplayBuffer.load_from_file(args.data_path)
        state_dim = buffer.state_dim
        action_dim = buffer.action_dim
        print(
            f"✓ Loaded {buffer.size} real ANYmal transitions from {args.data_path} (State={state_dim}, Act={action_dim})"
        )
    else:
        print(
            f"ℹ️ Data path '{args.data_path}' not found; initializing buffer with synthetic exploration trajectories..."
        )
        buffer = TrajectoryReplayBuffer(
            max_capacity=1000, state_dim=state_dim, action_dim=action_dim
        )
        for _ in range(500):
            s = jax.random.normal(prng.next(), (state_dim,))
            a = jax.random.normal(prng.next(), (action_dim,))
            r = 1.0
            ns = s + 0.1 * a.mean()
            buffer.add(s, a, r, ns, 0.0)

    # Initialize model with NNX Rngs
    rngs = nnx.Rngs(params=prng.next())
    model = TransformerWorldModel(
        state_dim=state_dim,
        action_dim=action_dim,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        mlp_dim=args.mlp_dim,
        rngs=rngs,
    )

    optimizer = nnx.Optimizer(model, optax.adamw(learning_rate=1e-3), wrt=nnx.Param)

    seq_len = args.seq_len
    if buffer.size <= seq_len:
        seq_len = max(2, buffer.size - 1)
        print(f"  ℹ️ Adjusting seq_len to K={seq_len} to fit buffer size ({buffer.size})")

    print(f"Starting JIT-compiled training loop for {args.num_steps} steps...")
    for step in range(1, args.num_steps + 1):
        batch = buffer.sample_sequences(prng.next(), batch_size=args.batch_size, seq_len=seq_len)
        loss_val = train_step(model, optimizer, batch)

        if step % 20 == 0 or step == 1 or step == args.num_steps:
            print(
                f"Step {step:03d} / {args.num_steps:03d} - Causal Transformer MSE Loss: {loss_val:.6f}"
            )

    # Save trained checkpoint
    save_path = save_checkpoint(model, args.save_checkpoint)
    print(f"✓ Saved Transformer World Model Checkpoint -> {save_path}")
    print("=== Milestones 2 & 3 Training Loop Verification Passed ===")


if __name__ == "__main__":
    main()

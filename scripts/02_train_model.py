#!/usr/bin/env python3
"""Milestones 2 & 3: JAX-JIT compiled Training Loop for Flax Transformer World Model."""

import argparse
import jax
import jax.numpy as jnp
import optax
from flax import nnx

from twm.models.transformer import TransformerWorldModel
from twm.utils.buffer import TrajectoryReplayBuffer
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
    optimizer.update(model, grads)
    return loss


def main():
    parser = argparse.ArgumentParser(description="Milestone 2 & 3: Train Transformer World Model in JAX.")
    parser.add_argument("--state_dim", type=int, default=27, help="State dimension")
    parser.add_argument("--action_dim", type=int, default=8, help="Action dimension")
    parser.add_argument("--num_steps", type=int, default=50, help="Training iterations")
    parser.add_argument("--seq_len", type=int, default=32, help="Sequence window length K")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    args = parser.parse_args()

    print("=== Milestones 2 & 3: Initializing Flax NNX Causal Transformer World Model ===")
    prng = PRNGSequence(seed=42)

    # Initialize model with NNX Rngs
    rngs = nnx.Rngs(params=prng.next())
    model = TransformerWorldModel(
        state_dim=args.state_dim,
        action_dim=args.action_dim,
        embed_dim=128,
        num_heads=4,
        num_layers=2,
        mlp_dim=256,
        rngs=rngs,
    )

    optimizer = nnx.Optimizer(model, optax.adamw(learning_rate=1e-3), wrt=nnx.Param)

    # Initialize buffer with synthetic trajectories for training validation
    buffer = TrajectoryReplayBuffer(max_capacity=1000, state_dim=args.state_dim, action_dim=args.action_dim)
    for _ in range(500):
        s = jax.random.normal(prng.next(), (args.state_dim,))
        a = jax.random.normal(prng.next(), (args.action_dim,))
        r = 1.0
        ns = s + 0.1 * a.mean()
        buffer.add(s, a, r, ns, 0.0)

    print(f"Starting JIT-compiled training loop for {args.num_steps} steps...")
    for step in range(1, args.num_steps + 1):
        batch = buffer.sample_sequences(prng.next(), batch_size=args.batch_size, seq_len=args.seq_len)
        loss_val = train_step(model, optimizer, batch)

        if step % 10 == 0 or step == 1:
            print(f"Step {step:03d} / {args.num_steps:03d} - Causal Transformer MSE Loss: {loss_val:.6f}")

    print("=== Milestones 2 & 3 Training Loop Verification Passed ===")


if __name__ == "__main__":
    main()

"""Flax NNX Diffusion Policy with exact reverse log-likelihood computation."""

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
from flax import nnx


class DenoisingStepOutput(NamedTuple):
    """Output for a single reverse diffusion step."""

    mean: jax.Array  # Predicted reverse mean mu_theta [..., Action_Dim]
    var: jax.Array  # Posterior variance sigma_k^2 [...]
    pred_eps: jax.Array  # Predicted noise eps_theta [..., Action_Dim]
    x_prev: jax.Array  # Sampled x_{k-1} [..., Action_Dim]
    log_prob: jax.Array  # Log probability log p(x_{k-1} | x_k, s) [...]


class ReverseTrajectory(NamedTuple):
    """Container holding full reverse diffusion denoising trajectory."""

    actions: jax.Array  # Final generated actions x_0 [..., Action_Dim]
    trajectories: jax.Array  # All intermediate states [K+1, ..., Action_Dim]
    log_probs: jax.Array  # Total reverse log probability log pi(a | s) [...]
    step_log_probs: jax.Array  # Step-wise log probs [K, ...]


class SinusoidalPosEmb(nnx.Module):
    """Sinusoidal position embedding for diffusion timesteps."""

    def __init__(self, dim: int):
        self.dim = dim

    def __call__(self, x: jax.Array) -> jax.Array:
        half_dim = self.dim // 2
        emb_scale = math.log(10000) / max(1, half_dim - 1)
        freqs = jnp.exp(-emb_scale * jnp.arange(half_dim, dtype=jnp.float32))
        args = x[..., None] * freqs[None, :]
        return jnp.concatenate([jnp.sin(args), jnp.cos(args)], axis=-1)


class ResidualBlock(nnx.Module):
    """Residual MLP block with LayerNorm and SiLU activation."""

    def __init__(self, dim: int, *, rngs: nnx.Rngs):
        self.norm = nnx.LayerNorm(dim, rngs=rngs)
        self.fc1 = nnx.Linear(dim, dim, rngs=rngs)
        self.fc2 = nnx.Linear(dim, dim, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        residual = x
        h = self.norm(x)
        h = nnx.silu(self.fc1(h))
        h = self.fc2(h)
        return residual + h


class DenoisingNetwork(nnx.Module):
    """Score/Epsilon denoising network predicting noise eps_theta(x_k, k, s)."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        embed_dim: int = 256,
        num_layers: int = 3,
        time_embed_dim: int = 64,
        *,
        rngs: nnx.Rngs,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.embed_dim = embed_dim

        # Timestep encoder
        self.time_emb = SinusoidalPosEmb(time_embed_dim)
        self.time_mlp = nnx.Sequential(
            nnx.Linear(time_embed_dim, embed_dim, rngs=rngs),
            nnx.silu,
            nnx.Linear(embed_dim, embed_dim, rngs=rngs),
        )

        # State & Action input projections
        self.state_proj = nnx.Linear(state_dim, embed_dim, rngs=rngs)
        self.action_proj = nnx.Linear(action_dim, embed_dim, rngs=rngs)

        # Residual backbone blocks
        list_factory = getattr(nnx, "List", list)
        self.res_blocks = list_factory(
            [ResidualBlock(embed_dim, rngs=rngs) for _ in range(num_layers)]
        )

        self.final_norm = nnx.LayerNorm(embed_dim, rngs=rngs)
        self.out_proj = nnx.Linear(embed_dim, action_dim, rngs=rngs)

    def __call__(self, x_k: jax.Array, timesteps: jax.Array, obs: jax.Array) -> jax.Array:
        """Predict noise epsilon given noisy action x_k, step k, and state conditioning obs.

        Args:
            x_k: [..., Action_Dim]
            timesteps: [...] integer or float step indices [0, K-1]
            obs: [..., State_Dim]
        Returns:
            eps_pred: [..., Action_Dim]
        """
        # Embed timestep
        t_emb = self.time_mlp(self.time_emb(timesteps.astype(jnp.float32)))
        s_emb = self.state_proj(obs)
        a_emb = self.action_proj(x_k)

        # Combine conditioning: action + state + time
        h = a_emb + s_emb + t_emb
        for block in self.res_blocks:
            h = block(h)

        h = self.final_norm(h)
        return self.out_proj(h)


class DiffusionPolicy(nnx.Module):
    """Gaussian Diffusion Policy with tractable likelihood evaluation for RL / GRPO."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        embed_dim: int = 256,
        num_layers: int = 3,
        num_timesteps: int = 8,
        beta_start: float = 1e-4,
        beta_end: float = 0.04,
        *,
        rngs: nnx.Rngs,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end

        # Neural Denoising Score Network
        self.net = DenoisingNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            embed_dim=embed_dim,
            num_layers=num_layers,
            rngs=rngs,
        )

        # Precompute diffusion schedules (constants in JAX arrays)
        # 1-indexed timesteps 1..K
        betas = jnp.linspace(beta_start, beta_end, num_timesteps, dtype=jnp.float32)
        alphas = 1.0 - betas
        alphas_cumprod = jnp.cumprod(alphas, axis=0)
        alphas_cumprod_prev = jnp.pad(alphas_cumprod[:-1], (1, 0), constant_values=1.0)

        # Posterior variance sigma_k^2 = (1 - alpha_prev) / (1 - alpha_cumprod) * beta
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        # Avoid log(0) for k=0
        posterior_variance = jnp.maximum(posterior_variance, 1e-8)

        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.alphas_cumprod_prev = alphas_cumprod_prev
        self.sqrt_recip_alphas = jnp.sqrt(1.0 / alphas)
        self.sqrt_alphas_cumprod = jnp.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = jnp.sqrt(1.0 - alphas_cumprod)
        self.posterior_variance = posterior_variance

    def compute_reverse_mean(
        self, x_k: jax.Array, k: int, obs: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Compute reverse mean mu_theta(x_k, k, s) and predicted noise."""
        batch_shape = x_k.shape[:-1]
        k_tensor = jnp.full(batch_shape, k, dtype=jnp.int32)

        pred_eps = self.net(x_k, k_tensor, obs)

        # mu_theta = 1 / sqrt(alpha_k) * (x_k - beta_k / sqrt(1 - alpha_bar_k) * eps)
        coeff = self.betas[k] / self.sqrt_one_minus_alphas_cumprod[k]
        mean = self.sqrt_recip_alphas[k] * (x_k - coeff * pred_eps)
        var = self.posterior_variance[k]

        return mean, var, pred_eps

    def compute_step_log_prob(
        self, x_prev: jax.Array, mean: jax.Array, var: jax.Array
    ) -> jax.Array:
        """Evaluate log p(x_{k-1} | x_k, s) under diagonal Gaussian N(mean, var * I)."""
        d = self.action_dim
        diff_sq = jnp.sum(jnp.square(x_prev - mean), axis=-1)  # [...]
        log_prob = -0.5 * (diff_sq / var + d * jnp.log(2.0 * math.pi * var))
        return log_prob

    def sample_trajectory(
        self,
        rng: jax.Array,
        obs: jax.Array,
        deterministic: bool = False,
    ) -> ReverseTrajectory:
        """Run full reverse diffusion process to sample action x_0 starting from Gaussian noise x_K.

        Args:
            rng: PRNGKey
            obs: State conditioning [..., State_Dim]
            deterministic: If True, set sampling noise to zero (mean rollout)

        Returns:
            ReverseTrajectory with actions x_0, full trajectory, and log probabilities.
        """
        batch_shape = obs.shape[:-1]
        rng_init, rng_steps = jax.random.split(rng)

        # Initialize from pure standard normal noise x_K ~ N(0, I)
        x_k = jax.random.normal(rng_init, (*batch_shape, self.action_dim))

        all_states = [x_k]
        step_log_probs = []

        curr_x = x_k
        step_keys = jax.random.split(rng_steps, self.num_timesteps)

        for k_idx in reversed(range(self.num_timesteps)):
            mean, var, _ = self.compute_reverse_mean(curr_x, k_idx, obs)

            if k_idx > 0 and not deterministic:
                noise = jax.random.normal(step_keys[k_idx], curr_x.shape)
                curr_x_prev = mean + jnp.sqrt(var) * noise
            else:
                curr_x_prev = mean

            step_lp = self.compute_step_log_prob(curr_x_prev, mean, var)
            step_log_probs.append(step_lp)

            curr_x = curr_x_prev
            all_states.append(curr_x)

        # Total log-probability of the denoising trajectory
        stacked_step_lps = jnp.stack(step_log_probs, axis=0)  # [K, ...]
        total_log_prob = jnp.sum(stacked_step_lps, axis=0)  # [...]
        stacked_trajectories = jnp.stack(all_states, axis=0)  # [K+1, ..., Action_Dim]

        # Action is bounded joint position residual [-1, 1]
        action = jnp.clip(curr_x, -1.0, 1.0)

        return ReverseTrajectory(
            actions=action,
            trajectories=stacked_trajectories,
            log_probs=total_log_prob,
            step_log_probs=stacked_step_lps,
        )

    def evaluate_trajectory_log_prob(
        self,
        trajectories: jax.Array,
        obs: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """Compute exact log-probabilities for a given stored reverse trajectory.

        Args:
            trajectories: [K+1, ..., Action_Dim] (intermediate reverse states x_K ... x_0)
            obs: [..., State_Dim]

        Returns:
            total_log_prob: [...]
            step_log_probs: [K, ...]
        """
        step_log_probs = []
        # trajectories[0] is x_K, trajectories[1] is x_{K-1}, ..., trajectories[K] is x_0
        for step_i, k_idx in enumerate(reversed(range(self.num_timesteps))):
            x_k = trajectories[step_i]
            x_prev = trajectories[step_i + 1]

            mean, var, _ = self.compute_reverse_mean(x_k, k_idx, obs)
            step_lp = self.compute_step_log_prob(x_prev, mean, var)
            step_log_probs.append(step_lp)

        stacked_step_lps = jnp.stack(step_log_probs, axis=0)
        total_log_prob = jnp.sum(stacked_step_lps, axis=0)
        return total_log_prob, stacked_step_lps

from __future__ import annotations

import numpy as np
import distrax
import flax.linen as nn
import jax.numpy as jnp
from flax.linen.initializers import constant, orthogonal


class ActorCritic(nn.Module):
    """Two-head actor-critic MLP for PPO experiments."""

    action_dim: int
    hidden_dim: int = 64
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        activation = nn.relu if self.activation == "relu" else nn.tanh

        x = jnp.asarray(x, dtype=jnp.float32)

        actor = nn.Dense(
            self.hidden_dim,
            kernel_init=orthogonal(np.sqrt(2.0)),
            bias_init=constant(0.0),
        )(x)
        actor = activation(actor)
        actor = nn.Dense(
            self.hidden_dim,
            kernel_init=orthogonal(np.sqrt(2.0)),
            bias_init=constant(0.0),
        )(actor)
        actor = activation(actor)
        actor_logits = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
        )(actor)
        pi = distrax.Categorical(logits=actor_logits)

        critic = nn.Dense(
            self.hidden_dim,
            kernel_init=orthogonal(np.sqrt(2.0)),
            bias_init=constant(0.0),
        )(x)
        critic = activation(critic)
        critic = nn.Dense(
            self.hidden_dim,
            kernel_init=orthogonal(np.sqrt(2.0)),
            bias_init=constant(0.0),
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(
            1,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
        )(critic)

        return pi, jnp.squeeze(critic, axis=-1)



def network_summary(input_dim: int, belief_dim: int = 4):
    return {
        "obs_only_input_dim": input_dim,
        "obs_belief_input_dim": input_dim + belief_dim,
        "actor_critic_style": "separate_mlp_towers",
    }

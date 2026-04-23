from pathlib import Path
import sys

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.beliefs.dummy_belief import DummyBelief
from core.envs.dummy_env import DummyEnv
from core.models.ppo_netowork import ActorCritic
from core.utils.obs_encoder import ObservationEncoder
from config.basic_config import basic_config
from godot_rl.core.godot_env import GodotEnv


import jax
import jax.numpy as jnp
import optax



NUM_ACTIONS = 13
USE_BELIEF_INPUT = False
PPO_CLIP_EPS = 0.2
VALUE_COEF = 0.5
ENTROPY_COEF = 0.01
LEARNING_RATE = 3e-4


def random_policy(obs_vec, belief):
    del obs_vec, belief

    action = np.random.randint(NUM_ACTIONS)
    logprob = np.log(1.0 / NUM_ACTIONS)
    value = 0.0
    return action, logprob, value


def build_policy_input(obs_vec, belief_vec, use_belief_input=USE_BELIEF_INPUT):
    if use_belief_input:
        return np.concatenate([obs_vec, belief_vec], axis=0)
    return obs_vec


def init_policy_model(obs_dim, belief_dim, action_dim=NUM_ACTIONS, use_belief_input=USE_BELIEF_INPUT):
    input_dim = obs_dim + belief_dim if use_belief_input else obs_dim
    model = ActorCritic(action_dim=action_dim)
    rng = jax.random.PRNGKey(0)
    dummy_x = jnp.zeros((input_dim,), dtype=jnp.float32)
    params = model.init(rng, dummy_x)
    return model, params, rng


def select_action(model, params, rng, obs_vec, belief_vec, use_belief_input=USE_BELIEF_INPUT):
    policy_input = build_policy_input(obs_vec, belief_vec, use_belief_input)
    rng, sample_key = jax.random.split(rng)
    x = jnp.asarray(policy_input, dtype=jnp.float32)
    pi, value = model.apply(params, x)
    action = pi.sample(seed=sample_key)
    logprob = pi.log_prob(action)

    return int(action), float(logprob), float(value), rng


def collect_rollout(env, encoder, belief, model=None, params=None, rng=None, max_steps=50):
    trajectory = []
    obs_list, info = env.reset(basic_config)
    # Extract single observation from list (single-agent case)
    obs = obs_list[0] if isinstance(obs_list, list) else obs_list

    for step in range(max_steps):
        obs_vec = encoder.encode(obs)
        belief_vec = belief.update(obs_vec)
        obs_belief_vec = build_policy_input(obs_vec, belief_vec, use_belief_input=True)

        action, logprob, value, rng = select_action(
            model,
            params,
            rng,
            obs_vec,
            belief_vec,
            use_belief_input=USE_BELIEF_INPUT,
        )

        # Send discrete action (0-12) to Godot controller
        # Format: [action_index] for each environment (single-agent case)
        action_for_env = [np.asarray([action], dtype=np.int32)]
        
        # Step the environment
        obs_list, reward_list, done_list, truncated_list, info_list = env.step(action_for_env)
        print(obs_list[0])
        # Extract single-agent results from lists
        obs_next = obs_list[0] if isinstance(obs_list, list) else obs_list
        reward = reward_list[0] if isinstance(reward_list, list) else reward_list
        done = done_list[0] if isinstance(done_list, list) else done_list
        info = info_list[0] if isinstance(info_list, list) else info_list

        trajectory.append(
            {
                "step": step,
                "obs": obs_vec,
                "belief": belief_vec,
                "obs_belief": obs_belief_vec,
                "action": action,
                "reward": float(reward),
                "done": float(done),
                "logprob": float(logprob),
                "value": float(value),
                "info": info,
            }
        )

        obs = obs_next

        if done:
            break

    return trajectory


def compute_gae(trajectory, last_value=0.0, gamma=0.99, gae_lambda=0.95):
    advantages = np.zeros(len(trajectory), dtype=np.float32)
    returns = np.zeros(len(trajectory), dtype=np.float32)

    next_gae = 0.0
    next_value = float(last_value)

    for t in reversed(range(len(trajectory))):
        reward = float(trajectory[t]["reward"])
        value = float(trajectory[t]["value"])
        done = float(trajectory[t]["done"])

        delta = reward + gamma * next_value * (1.0 - done) - value
        gae = delta + gamma * gae_lambda * (1.0 - done) * next_gae

        advantages[t] = gae
        returns[t] = gae + value

        next_gae = gae
        next_value = value

    return advantages, returns


def attach_returns_and_advantages(
    trajectory,
    last_value=0.0,
    gamma=0.99,
    gae_lambda=0.95,
):
    advantages, returns = compute_gae(
        trajectory,
        last_value=last_value,
        gamma=gamma,
        gae_lambda=gae_lambda,
    )

    for t in range(len(trajectory)):
        trajectory[t]["advantage"] = float(advantages[t])
        trajectory[t]["return"] = float(returns[t])

    return trajectory


def build_ppo_batch(trajectory, use_belief_input=USE_BELIEF_INPUT, normalize_advantage=True): # for 
    obs_key = "obs_belief" if use_belief_input else "obs"

    batch = {
        "inputs": np.asarray([step[obs_key] for step in trajectory], dtype=np.float32),
        "obs": np.asarray([step["obs"] for step in trajectory], dtype=np.float32),
        "belief": np.asarray([step["belief"] for step in trajectory], dtype=np.float32),
        "actions": np.asarray([step["action"] for step in trajectory], dtype=np.int32),
        "old_logprobs": np.asarray([step["logprob"] for step in trajectory], dtype=np.float32),
        "values": np.asarray([step["value"] for step in trajectory], dtype=np.float32),
        "returns": np.asarray([step["return"] for step in trajectory], dtype=np.float32),
        "advantages": np.asarray([step["advantage"] for step in trajectory], dtype=np.float32),
        "rewards": np.asarray([step["reward"] for step in trajectory], dtype=np.float32),
        "dones": np.asarray([step["done"] for step in trajectory], dtype=np.float32),
    }

    if normalize_advantage:
        adv = batch["advantages"]
        batch["advantages"] = (adv - adv.mean()) / (adv.std() + 1e-8)

    return {key: jnp.asarray(value) for key, value in batch.items()}


def ppo_loss(
    model,
    params,
    batch,
    clip_eps=PPO_CLIP_EPS,
    value_coef=VALUE_COEF,
    entropy_coef=ENTROPY_COEF,
):
    def loss_per_sample(x, action, old_logprob, advantage, target_return):
        pi, value = model.apply(params, x)
        new_logprob = pi.log_prob(action)
        ratio = jnp.exp(new_logprob - old_logprob)

        unclipped = ratio * advantage
        clipped = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantage
        actor_loss = -jnp.minimum(unclipped, clipped)

        value_error = target_return - value
        critic_loss = value_error * value_error
        entropy = pi.entropy()

        total = actor_loss + value_coef * critic_loss - entropy_coef * entropy
        return total, (actor_loss, critic_loss, entropy, new_logprob, value)

    total_loss, aux = jax.vmap(
        loss_per_sample,
        in_axes=(0, 0, 0, 0, 0),
    )(
        batch["inputs"],
        batch["actions"],
        batch["old_logprobs"],
        batch["advantages"],
        batch["returns"],
    )

    actor_loss, critic_loss, entropy, new_logprob, value = aux

    metrics = {
        "total_loss": jnp.mean(total_loss),
        "actor_loss": jnp.mean(actor_loss),
        "critic_loss": jnp.mean(critic_loss),
        "entropy": jnp.mean(entropy),
        "mean_ratio": jnp.mean(jnp.exp(new_logprob - batch["old_logprobs"])),
        "mean_value": jnp.mean(value),
    }
    return metrics["total_loss"], metrics


def train_step(model, params, opt_state, optimizer, batch): # update model
    def loss_fn(current_params):
        return ppo_loss(model, current_params, batch)

    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss, metrics


def tree_l2_diff(tree_a, tree_b):
    leaves_a = jax.tree_util.tree_leaves(tree_a)
    leaves_b = jax.tree_util.tree_leaves(tree_b)
    squared_sum = sum(jnp.sum((a - b) ** 2) for a, b in zip(leaves_a, leaves_b))
    return jnp.sqrt(squared_sum)


def repeated_update_sanity_check(model, params, optimizer, opt_state, batch, num_steps=10):
    losses = []
    critic_losses = []
    current_params = params
    current_opt_state = opt_state

    for _ in range(num_steps):
        loss, metrics = ppo_loss(model, current_params, batch)
        losses.append(float(loss))
        critic_losses.append(float(metrics["critic_loss"]))
        current_params, current_opt_state, _, _ = train_step(
            model,
            current_params,
            current_opt_state,
            optimizer,
            batch,
        )

    final_loss, final_metrics = ppo_loss(model, current_params, batch)
    losses.append(float(final_loss))
    critic_losses.append(float(final_metrics["critic_loss"]))

    return {
        "losses": losses,
        "critic_losses": critic_losses,
        "final_params": current_params,
        "final_opt_state": current_opt_state,
    }


def main():
    # Give the editor more time to start and connect when using Play.
    GodotEnv.DEFAULT_TIMEOUT = 180
    # Initialize the environment - If env_path is None, it will wait for you to press PLAY in the Godot editor
    env = GodotEnv(
        env_path=None,  # Set to path of exported Godot binary, or None for in-editor training
        port=11008,
        show_window=False,
    )

    encoder = ObservationEncoder()
    belief = DummyBelief()
    model, params, rng = init_policy_model(
        obs_dim=encoder.obs_dim,
        belief_dim=belief.num_strategies,
        action_dim=NUM_ACTIONS,
        use_belief_input=USE_BELIEF_INPUT,
    )
    optimizer = optax.adam(learning_rate=LEARNING_RATE)
    opt_state = optimizer.init(params)

    trajectory = collect_rollout(
        env,
        encoder,
        belief,
        model=model,
        params=params,
        rng=rng,
        max_steps=1000,
    )

    trajectory = attach_returns_and_advantages(trajectory, last_value=0.0)
    batch = build_ppo_batch(trajectory, use_belief_input=USE_BELIEF_INPUT) 
    params_before = params
    loss_before, metrics_before = ppo_loss(model, params, batch)
    params, opt_state, _, _ = train_step(
        model,
        params,
        opt_state,
        optimizer,
        batch,
    )
    loss_after, metrics_after = ppo_loss(model, params, batch)
    param_delta = tree_l2_diff(params_before, params)
    sanity = repeated_update_sanity_check(
        model,
        params,
        optimizer,
        opt_state,
        batch,
        num_steps=10,
    )

    print(f"Collected {len(trajectory)} transitions.")
    action_dict = {0: "move forward", 1:"move left", 2: "move right", 3:"move back", 4: "attack hero", 5: "attack nearest minions", 6:"retreat", 7:"attack tower", 8: "hold"}
    if trajectory:
        print(f"Policy input dim: {batch['inputs'].shape[-1]}")
        print(f"Advantage shape: {batch['advantages'].shape}")
        print(f"Initial PPO loss: {float(loss_before):.4f}")
        print(f"Post-update PPO loss: {float(loss_after):.4f}")
        print(
            "Before update:",
            {k: float(v) for k, v in metrics_before.items() if k != "mean_value"},
        )
        print(
            "After update:",
            {k: float(v) for k, v in metrics_after.items() if k != "mean_value"},
        )
        print(f"Parameter L2 delta: {float(param_delta):.6f}")
        print(f"Repeated update loss trace: {sanity['losses'][:5]} ... -> {sanity['losses'][-1]:.4f}")
        print(
            f"Repeated update critic trace: {sanity['critic_losses'][:5]} ... -> "
            f"{sanity['critic_losses'][-1]:.4f}"
        )
        print("Sampled actions:", [action_dict[step["action"]] for step in trajectory[:5]])
    
    # Close the environment
    env.close()
    print("Training complete!")
    #TODO need to add a message to close the environment/exit the python code when you exit the simulator
    


if __name__ == "__main__":
    main()

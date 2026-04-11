from pathlib import Path
import sys

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.beliefs.dummy_belief import DummyBelief
from core.envs.dummy_env import DummyEnv
from core.models.ppo_netowork import ActorCritic
from core.utils.obs_encoder import ObservationEncoder

import jax
import jax.numpy as jnp



NUM_ACTIONS = 9
USE_BELIEF_INPUT = True


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
    obs = env.reset()

    for step in range(max_steps):
        obs_vec = encoder.encode(obs)
        belief_vec = belief.update(obs_vec)
        obs_belief_vec = build_policy_input(obs_vec, belief_vec, use_belief_input=False)
        # observation only uses obs_vec
        # obs + belief uses obs_belief_vec

        action, logprob, value, rng = select_action(
            model,
            params,
            rng,
            obs_vec,
            belief_vec,
            use_belief_input=USE_BELIEF_INPUT,
        )

        obs_next, reward, done, info = env.step(action)

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


def main():
    env = DummyEnv()
    encoder = ObservationEncoder()
    belief = DummyBelief()
    model, params, rng = init_policy_model(
        obs_dim=encoder.obs_dim,
        belief_dim=belief.num_strategies,
        action_dim=NUM_ACTIONS,
        use_belief_input=USE_BELIEF_INPUT,
    )

    trajectory = collect_rollout(
        env,
        encoder,
        belief,
        model=model,
        params=params,
        rng=rng,
        max_steps=50,
    )

    print(f"Collected {len(trajectory)} transitions.")
    action_dict = {0: "move forward", 1:"move left", 2: "move right", 3:"move back", 4: "attack hero", 5: "attack nearest minions", 6:"retreat", 7:"attack tower", 8: "hold"}
    if trajectory:
        print(trajectory[0]["obs_belief"].shape)
        for i in range(len(trajectory)):
            print(action_dict[trajectory[i]['action']])


if __name__ == "__main__":
    main()

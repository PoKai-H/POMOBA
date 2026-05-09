from collections import Counter
import argparse
from pathlib import Path
import sys

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.neutralExpert_neutral import basic_config, training_config
from core.envs.dummy_env import DummyEnv
from core.strategy.basic_strategy import NeutralStrategy
from core.utils.obs_encoder import unwrap_obs
from godot_rl.core.godot_env import GodotEnv


ACTION_NAMES = {
    0: "move_up",
    1: "move_left",
    2: "move_right",
    3: "move_down",
    4: "move_up_left",
    5: "move_up_right",
    6: "move_down_left",
    7: "move_down_right",
    8: "hold",
    9: "attack_hero",
    10: "attack_nearest_minion",
    11: "attack_tower",
    12: "retreat",
}


def build_test_config():
    config = {
        **basic_config,
        **training_config,
        "core": {
            **basic_config.get("core", {}),
            "opponent_strategy": "neutral",
            "strategy_switch_mode": "fixed",
        },
        "extensions": {
            **basic_config.get("extensions", {}),
        },
        "DEFAULT_NPC_POLICY": "neutral",
        "NPC_POLICIES": {
            1: "neutral",
        },
        "RANDOMIZE_NPC_POLICY_EACH_EPISODE": False,
    }
    return config


def make_env(config):
    if config.get("USE_DUMMY_ENV", False):
        return DummyEnv(seed=config.get("SEED", 42))

    GodotEnv.DEFAULT_TIMEOUT = config.get("GODOT_TIMEOUT", 180)
    return GodotEnv(
        env_path=config.get("GODOT_ENV_PATH", None),
        port=config.get("GODOT_PORT", 11008),
        show_window=config.get("SHOW_WINDOW", False),
    )


def scripted_rollout(env, config, max_steps):
    obs_list, _ = env.reset(config)
    policies = [NeutralStrategy() for _ in obs_list]
    trajectory = []
    episode_rewards = np.zeros(len(obs_list), dtype=np.float32)

    for step in range(max_steps):
        try:
            actions = [
                policy.select_action(unwrap_obs(agent_obs))
                for policy, agent_obs in zip(policies, obs_list)
            ]
            action_for_env = [np.asarray(actions, dtype=np.int32)]
            obs_next_list, reward_list, done_list, truncated_list, info_list = env.step(action_for_env)
            print(action_for_env)
        except Exception as exc:
            trajectory.append(
                {
                    "step": step,
                    "actions": [],
                    "action_names": [],
                    "rewards": [],
                    "dones": [],
                    "truncated": [],
                    "obs": obs_list,
                    "infos": [],
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            break

        rewards = np.asarray(reward_list, dtype=np.float32)
        episode_rewards += rewards
        trajectory.append(
            {
                "step": step,
                "actions": actions,
                "action_names": [ACTION_NAMES.get(action, str(action)) for action in actions],
                "rewards": rewards.tolist(),
                "dones": done_list,
                "truncated": truncated_list,
                "obs": obs_list,
                "infos": info_list,
            }
        )

        obs_list = obs_next_list
        if any(done_list) or any(truncated_list):
            break

    return trajectory, episode_rewards


def summarize(trajectory, episode_rewards):
    action_counts = Counter()
    for step in trajectory:
        action_counts.update(step["action_names"])

    print(f"Collected {len(trajectory)} scripted steps.")
    print("Episode rewards:", [round(float(value), 3) for value in episode_rewards])
    print("Action counts:", dict(action_counts))
    print("First actions:", [step["action_names"] for step in trajectory[:5]])
    if trajectory:
        print("First rewards:", [step["rewards"] for step in trajectory[:5]])
        if "error" in trajectory[-1]:
            print("Stopped after env error:", trajectory[-1]["error"])


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a scripted-vs-scripted neutral pipeline test."
    )
    parser.add_argument("--dummy", action="store_true", help="Use DummyEnv instead of Godot.")
    parser.add_argument("--max-steps", type=int, default=None, help="Override rollout length.")
    return parser.parse_args()


def main():
    args = parse_args()
    config = build_test_config()
    if args.dummy:
        config["USE_DUMMY_ENV"] = True
    if args.max_steps is not None:
        config["MAX_STEPS_PER_EPISODE"] = args.max_steps
        config["core"] = {
            **config["core"],
            "max_steps": args.max_steps,
        }

    env = make_env(config)
    try:
        trajectory, episode_rewards = scripted_rollout(
            env,
            config,
            max_steps=int(config["MAX_STEPS_PER_EPISODE"]),
        )
        summarize(trajectory, episode_rewards)
    finally:
        env.close()


if __name__ == "__main__":
    main()

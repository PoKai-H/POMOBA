from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.basic_config import basic_config, training_config
from core.beliefs.bayesian_belief import BayesianBelief
from core.envs.dummy_env import DummyEnv
from core.models.ppo import PPO
from core.utils.obs_encoder import ObservationEncoder
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


def build_run_config():
    return {
        **basic_config,
        **training_config,
        "core": {
            **basic_config.get("core", {}),
            "opponent_strategy": training_config["opponent_strategy"],
            "strategy_switch_mode": training_config["strategy_switch_mode"],
        },
        "extensions": {
            **basic_config.get("extensions", {}),
        },
    }


def config_for_update(base_config, update_idx):
    del update_idx
    return {
        **base_config,
        "core": {
            **base_config.get("core", {}),
            "opponent_strategy": base_config["opponent_strategy"],
            "strategy_switch_mode": base_config["strategy_switch_mode"],
        },
        "extensions": {
            **base_config.get("extensions", {}),
        },
    }


def make_env(config):
    if config.get("USE_DUMMY_ENV", False):
        return DummyEnv(seed=config.get("SEED", 42))

    GodotEnv.DEFAULT_TIMEOUT = config.get("GODOT_TIMEOUT", 180)
    return GodotEnv(
        env_path=config.get("GODOT_ENV_PATH", None),
        port=config.get("GODOT_PORT", 11008),
        show_window=config.get("SHOW_WINDOW", False),
    )


def format_metrics(metrics):
    return {
        key: round(float(value), 4)
        for key, value in metrics.items()
    }


def latest_episode_summary(agent):
    if not agent.episode_logs:
        return "episode_reward=n/a episode_length=n/a"

    latest = agent.episode_logs[-1]
    return (
        f"episode_reward={latest['episode_reward']:.2f} "
        f"episode_length={latest['episode_length']}"
    )


def train():
    config = build_run_config()
    env = make_env(config)
    encoder = ObservationEncoder()
    belief = BayesianBelief()
    agent = PPO(env=env, encoder=encoder, belief=belief, config=config)

    total_timesteps = int(config["TOTAL_TIMESTEPS"])
    timesteps_per_batch = int(config["TIMESTEP_PER_BATCH"])
    num_updates = max(1, total_timesteps // timesteps_per_batch)

    try:
        for update_idx in range(1, num_updates + 1):
            current_config = config_for_update(config, update_idx)
            trajectory, last_value = agent.collect_rollout(config=current_config)
            trajectory = agent.attach_returns_and_advantages(
                trajectory,
                last_value=last_value,
            )
            batch = agent.build_ppo_batch(trajectory)
            loss, metrics = agent.update(batch)

            sampled_actions = [
                ACTION_NAMES.get(step["action"], str(step["action"]))
                for step in trajectory[:5]
            ]

            print(
                f"[update {update_idx}/{num_updates}] "
                f"steps={len(trajectory)} "
                f"loss={float(loss):.4f} "
                f"{latest_episode_summary(agent)}"
            )
            print("  metrics:", format_metrics(metrics))
            print("  sampled_actions:", sampled_actions)

    finally:
        env.close()

    print("Training complete!")
    return agent


def main():
    train()


if __name__ == "__main__":
    main()

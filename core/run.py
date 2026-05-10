from collections import Counter
import argparse
from datetime import datetime
import importlib
import importlib.util
import json
import pickle
from pathlib import Path
from pprint import pformat
import re
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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

ANALYSIS_OUTPUT_ROOT = Path("outputs/training_analysis")
DEFAULT_CONFIG_MODULE = "config.neutralExpert_neutral"


def _sanitize_run_name(name):
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._")
    return sanitized or "run"


def _config_name_from_arg(config_arg):
    config_path = Path(config_arg)
    if config_path.suffix == ".py":
        return config_path.stem
    return config_arg.rsplit(".", 1)[-1]


def _module_name_from_arg(config_arg):
    if config_arg.endswith(".py") or "/" in config_arg:
        return None
    if "." in config_arg:
        return config_arg
    return f"config.{config_arg}"


def load_config_module(config_arg):
    module_name = _module_name_from_arg(config_arg)
    if module_name is not None:
        return importlib.import_module(module_name)

    config_path = Path(config_arg)
    if not config_path.is_absolute():
        config_path = Path.cwd() / config_path
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    module_name = f"runtime_config_{config_path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, config_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load config file: {config_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_run_config(config_arg):
    module = load_config_module(config_arg)
    if not hasattr(module, "basic_config") or not hasattr(module, "training_config"):
        raise AttributeError(
            "Config module must define both `basic_config` and `training_config`."
        )
    return (
        module.basic_config,
        module.training_config,
        _sanitize_run_name(_config_name_from_arg(config_arg)),
    )


def build_run_config(basic_config, training_config, config_name):
    return {
        **basic_config,
        **training_config,
        "CONFIG_NAME": config_name,
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
    initial_ratio = float(base_config.get("EXPERT_MIX_INITIAL_RATIO", 0.0))
    final_ratio = float(base_config.get("EXPERT_MIX_FINAL_RATIO", 0.0))
    decay_updates = max(1, int(base_config.get("EXPERT_MIX_DECAY_UPDATES", 1)))
    decay_progress = min(max(update_idx - 1, 0), decay_updates) / decay_updates
    expert_ratio = initial_ratio + (final_ratio - initial_ratio) * decay_progress

    return {
        **base_config,
        "EXPERT_MIX_RATIO": expert_ratio,
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
        from core.envs.dummy_env import DummyEnv

        return DummyEnv(seed=config.get("SEED", 42))

    from godot_rl.core.godot_env import GodotEnv

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


def _mean(values):
    return sum(values) / len(values) if values else 0.0


def _quantiles(values):
    if not values:
        return {}

    ordered = sorted(float(value) for value in values)
    last_index = len(ordered) - 1
    return {
        "min": round(ordered[0], 4),
        "p25": round(ordered[int(last_index * 0.25)], 4),
        "mean": round(_mean(ordered), 4),
        "p75": round(ordered[int(last_index * 0.75)], 4),
        "max": round(ordered[-1], 4),
        "sum": round(sum(ordered), 4),
    }


def _top_counts(values, limit=6):
    total = len(values)
    counts = Counter(values)
    return [
        {
            "name": name,
            "count": count,
            "pct": round(count / total, 3) if total else 0.0,
        }
        for name, count in counts.most_common(limit)
    ]


def _summary_value(summary, path, default=0.0):
    current = summary
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    if current is None:
        return default
    return current


def _unwrap_step_obs(step, agent_id):
    obs_list = step.get("all_raw_obs", [])
    if agent_id >= len(obs_list):
        return {}
    raw_obs = obs_list[agent_id]
    if isinstance(raw_obs, dict) and isinstance(raw_obs.get("obs"), dict):
        return raw_obs["obs"]
    return raw_obs if isinstance(raw_obs, dict) else {}


def _object_summary(obs):
    objects = obs.get("objects", [])
    visible_enemy_minions = 0
    visible_ally_minions = 0
    visible_enemy_towers = []
    visible_ally_towers = []

    for obj in objects:
        if not obj.get("visible"):
            continue
        obj_type = obj.get("type")
        team = obj.get("team")
        hp = obj.get("status", {}).get("hp")
        if obj_type == "minion":
            if team == "enemy":
                visible_enemy_minions += 1
            elif team == "ally":
                visible_ally_minions += 1
        elif obj_type == "tower":
            entry = {
                "id": obj.get("id"),
                "hp": round(float(hp), 2) if hp is not None else None,
                "team": team,
            }
            if team == "enemy":
                visible_enemy_towers.append(entry)
            elif team == "ally":
                visible_ally_towers.append(entry)

    return {
        "visible_ally_minions": visible_ally_minions,
        "visible_enemy_minions": visible_enemy_minions,
        "visible_ally_towers": visible_ally_towers,
        "visible_enemy_towers": visible_enemy_towers,
    }


def _last_frame_summary(trajectory):
    if not trajectory:
        return {}

    last = trajectory[-1]
    all_actions = last.get("all_actions", [])
    all_policy_names = last.get("all_policy_names", [])
    all_rewards = last.get("all_rewards", [])
    all_dones = last.get("all_dones", [])
    all_truncated = last.get("all_truncated", [])
    agents = []

    for agent_id, action in enumerate(all_actions):
        obs = _unwrap_step_obs(last, agent_id)
        self_obs = obs.get("self", {})
        status = self_obs.get("status", {})
        agents.append(
            {
                "id": agent_id,
                "policy": all_policy_names[agent_id] if agent_id < len(all_policy_names) else None,
                "action": ACTION_NAMES.get(action, str(action)),
                "reward": round(float(all_rewards[agent_id]), 4) if agent_id < len(all_rewards) else None,
                "done": bool(all_dones[agent_id]) if agent_id < len(all_dones) else None,
                "truncated": bool(all_truncated[agent_id]) if agent_id < len(all_truncated) else None,
                "hp": round(float(status.get("hp", 0.0)), 2),
                "objects": _object_summary(obs),
            }
        )

    return {
        "step": last.get("step"),
        "agents": agents,
    }


def _event_counts(trajectory):
    totals = {
        "takedown_enemy_agents": 0,
        "takedown_enemy_minions": 0,
        "deaths": 0,
    }
    for step in trajectory:
        event_counts = step.get("info", {}).get("event_counts", {})
        for key in totals:
            totals[key] += int(event_counts.get(key, 0))
    return totals


def _latest_episode_details(agent):
    if not agent.episode_logs:
        return None

    latest = agent.episode_logs[-1]
    core_config = latest.get("core", {})
    return {
        "reward": round(float(latest["episode_reward"]), 4),
        "length": int(latest["episode_length"]),
        "terminated": bool(latest.get("terminated")),
        "truncated": bool(latest.get("truncated")),
        "opponent_strategy": core_config.get("opponent_strategy"),
        "strategy_switch_mode": core_config.get("strategy_switch_mode"),
    }


def rollout_summary(trajectory, agent):
    rewards = [float(step["reward"]) for step in trajectory]
    values = [float(step["value"]) for step in trajectory]
    returns = [float(step["return"]) for step in trajectory]
    advantages = [float(step["advantage"]) for step in trajectory]
    actions = [ACTION_NAMES.get(step["action"], str(step["action"])) for step in trajectory]
    dones = [step for step in trajectory if step.get("done")]
    truncated = [step for step in trajectory if step.get("truncated")]
    expert_steps = [step for step in trajectory if step.get("expert_mask", 0.0)]
    event_counts = _event_counts(trajectory)

    return {
        "reward": _quantiles(rewards),
        "value": _quantiles(values),
        "return": _quantiles(returns),
        "advantage": _quantiles(advantages),
        "positive_reward_steps": sum(value > 0.0 for value in rewards),
        "negative_reward_steps": sum(value < 0.0 for value in rewards),
        "actions": _top_counts(actions),
        "attack_rate": round(
            sum(action.startswith("attack") for action in actions) / len(actions),
            3,
        ) if actions else 0.0,
        "movement_rate": round(
            sum(action.startswith("move") for action in actions) / len(actions),
            3,
        ) if actions else 0.0,
        "retreat_rate": round(actions.count("retreat") / len(actions), 3) if actions else 0.0,
        "expert_action_rate": round(len(expert_steps) / len(actions), 3) if actions else 0.0,
        "configured_expert_ratio": round(float(agent.expert_mix_ratio), 3),
        "takedown_enemy_agents": event_counts["takedown_enemy_agents"],
        "takedown_enemy_minions": event_counts["takedown_enemy_minions"],
        "death_count": event_counts["deaths"],
        "done_count": len(dones),
        "truncated_count": len(truncated),
        "done_steps": [step["step"] for step in dones[-3:]],
        "truncated_steps": [step["step"] for step in truncated[-3:]],
        "latest_episode": _latest_episode_details(agent),
        "last_frame": _last_frame_summary(trajectory),
    }


def training_history_entry(update_idx, trajectory, loss, metrics, summary):
    latest_episode = summary.get("latest_episode") or {}
    actions = [ACTION_NAMES.get(step["action"], str(step["action"])) for step in trajectory]
    action_counts = Counter(actions)
    total_actions = len(actions)
    return {
        "update": update_idx,
        "steps": len(trajectory),
        "loss": round(float(loss), 6),
        "metrics": format_metrics(metrics),
        "episode_reward": latest_episode.get("reward"),
        "episode_length": latest_episode.get("length"),
        "terminated": latest_episode.get("terminated"),
        "truncated": latest_episode.get("truncated"),
        "opponent_strategy": latest_episode.get("opponent_strategy"),
        "strategy_switch_mode": latest_episode.get("strategy_switch_mode"),
        "reward_mean": _summary_value(summary, ("reward", "mean")),
        "reward_sum": _summary_value(summary, ("reward", "sum")),
        "reward_min": _summary_value(summary, ("reward", "min")),
        "reward_max": _summary_value(summary, ("reward", "max")),
        "return_mean": _summary_value(summary, ("return", "mean")),
        "advantage_mean": _summary_value(summary, ("advantage", "mean")),
        "advantage_min": _summary_value(summary, ("advantage", "min")),
        "advantage_max": _summary_value(summary, ("advantage", "max")),
        "value_mean": _summary_value(summary, ("value", "mean")),
        "positive_reward_steps": summary.get("positive_reward_steps", 0),
        "negative_reward_steps": summary.get("negative_reward_steps", 0),
        "attack_rate": summary.get("attack_rate", 0.0),
        "movement_rate": summary.get("movement_rate", 0.0),
        "retreat_rate": summary.get("retreat_rate", 0.0),
        "expert_action_rate": summary.get("expert_action_rate", 0.0),
        "configured_expert_ratio": summary.get("configured_expert_ratio", 0.0),
        "takedown_enemy_agents": summary.get("takedown_enemy_agents", 0),
        "takedown_enemy_minions": summary.get("takedown_enemy_minions", 0),
        "death_count": summary.get("death_count", 0),
        "done_count": summary.get("done_count", 0),
        "truncated_count": summary.get("truncated_count", 0),
        "actions": summary.get("actions", []),
        "action_distribution": {
            name: round(action_counts.get(name, 0) / total_actions, 4)
            for name in ACTION_NAMES.values()
        },
    }


def _save_json(path, data):
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def create_output_dir(run_name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = ANALYSIS_OUTPUT_ROOT / f"{run_name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_checkpoint(agent, output_dir, update_idx, final=False):
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    filename = "final.pkl" if final else f"update_{update_idx:04d}.pkl"
    checkpoint_path = checkpoints_dir / filename
    checkpoint = {
        "update": int(update_idx),
        "final": bool(final),
        "params": agent.params,
        "opt_state": agent.opt_state,
        "network_signature": agent._network_signature,
        "optimizer_signature": agent._optimizer_signature,
        "config": agent.config,
        "episode_logs": agent.episode_logs,
    }
    with checkpoint_path.open("wb") as file:
        pickle.dump(checkpoint, file)
    return checkpoint_path


def save_training_artifacts(history, output_dir):
    if not history:
        return None

    _save_json(output_dir / "history.json", history)
    return output_dir


def latest_episode_summary(agent):
    if not agent.episode_logs:
        return "episode_reward=n/a episode_length=n/a"

    latest = agent.episode_logs[-1]
    return (
        f"episode_reward={latest['episode_reward']:.2f} "
        f"episode_length={latest['episode_length']}"
    )


def print_training_stop(reason, update_idx, num_updates, agent):
    print(
        f"Training stopped during update {update_idx}/{num_updates}: {reason}"
    )
    print("Latest episode:", latest_episode_summary(agent))


def train(config_arg=DEFAULT_CONFIG_MODULE):
    from core.beliefs.dummy_belief import DummyBelief
    from core.beliefs.bayesian_belief import BayesianBelief
    from core.models.ppo import PPO
    from core.utils.obs_encoder import ObservationEncoder

    basic_config, training_config, config_name = load_run_config(config_arg)
    config = build_run_config(basic_config, training_config, config_name)
    print(f"Using config: {config_name} ({config_arg})")
    output_dir = create_output_dir(config_name)
    print(f"Output directory: {output_dir}")
    _save_json(output_dir / "run_config.json", config)
    env = make_env(config)
    encoder = ObservationEncoder()
    belief = BayesianBelief()
    agent = PPO(env=env, encoder=encoder, belief=belief, config=config)

    total_timesteps = int(config["TOTAL_TIMESTEPS"])
    timesteps_per_batch = int(config["TIMESTEP_PER_BATCH"])
    num_updates = max(1, total_timesteps // timesteps_per_batch)
    checkpoint_every = int(config.get("CHECKPOINT_EVERY_UPDATES", 0))
    history = []
    last_update_idx = 0

    try:
        for update_idx in range(1, num_updates + 1):
            last_update_idx = update_idx
            current_config = config_for_update(config, update_idx)
            try:
                trajectory, last_value = agent.collect_rollout(config=current_config)
            except SystemExit as exc:
                print_training_stop(
                    f"environment exited with code {exc.code}",
                    update_idx,
                    num_updates,
                    agent,
                )
                return agent
            except (ConnectionError, ConnectionResetError, BrokenPipeError, OSError) as exc:
                print_training_stop(
                    f"{type(exc).__name__}: {exc}",
                    update_idx,
                    num_updates,
                    agent,
                )
                return agent
            except KeyboardInterrupt:
                print_training_stop("keyboard interrupt", update_idx, num_updates, agent)
                return agent

            if not trajectory:
                print_training_stop(
                    "no rollout steps were collected",
                    update_idx,
                    num_updates,
                    agent,
                )
                return agent

            trajectory = agent.attach_returns_and_advantages(
                trajectory,
                last_value=last_value,
            )
            batch = agent.build_ppo_batch(trajectory)
            loss, metrics = agent.update(batch)
            summary = rollout_summary(trajectory, agent)
            history.append(
                training_history_entry(
                    update_idx=update_idx,
                    trajectory=trajectory,
                    loss=loss,
                    metrics=metrics,
                    summary=summary,
                )
            )

            print(
                f"[update {update_idx}/{num_updates}] "
                f"steps={len(trajectory)} "
                f"loss={float(loss):.4f} "
                f"{latest_episode_summary(agent)}"
            )
            print("  metrics:", format_metrics(metrics))
            print("  rollout:")
            for line in pformat(summary, sort_dicts=False, width=100).splitlines():
                print(f"    {line}")

            if checkpoint_every > 0 and update_idx % checkpoint_every == 0:
                checkpoint_path = save_checkpoint(
                    agent,
                    output_dir,
                    update_idx=update_idx,
                )
                print(f"  checkpoint: {checkpoint_path}")

    finally:
        env.close()
        if history:
            final_update = history[-1]["update"]
            checkpoint_path = save_checkpoint(
                agent,
                output_dir,
                update_idx=final_update,
                final=True,
            )
            print(f"Final checkpoint saved to: {checkpoint_path}")
        elif last_update_idx > 0:
            print("No completed update was available for checkpointing.")

        if save_training_artifacts(history, output_dir) is not None:
            print(f"Training history saved to: {output_dir / 'history.json'}")

    print("Training complete!")
    return agent


def parse_args():
    parser = argparse.ArgumentParser(description="Run PPO training.")
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_MODULE,
        help=(
            "Config module or file path. Examples: "
            "`ppo_neutral`, `config.ppo_neutral`, or `config/ppo_neutral.py`."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    train(config_arg=args.config)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Evaluate one trained PPO checkpoint against scripted opponent settings.

This is for robustness/generalization testing. For example, a policy trained
against neutral can be evaluated against neutral, aggressive, and farming
without retraining. It can also evaluate against a config that changes the
opponent strategy during each episode.

Example:
    python evaluation/robustness_test.py \
        --checkpoint outputs/training_analysis/aggressiveExpert_neutral_x/checkpoints/final.pkl

    python evaluation/robustness_test.py \
        --checkpoint outputs/training_analysis/aggressiveExpert_neutral_x/checkpoints/final.pkl \
        --eval-config strategySwitching_onlyOBS
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import json
import pickle
from pathlib import Path
import sys

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.run import ACTION_NAMES, build_run_config, load_run_config, make_env


DEFAULT_OPPONENTS = ["aggressive", "neutral", "farming"]
DEFAULT_OUTPUT_ROOT = Path("evaluation/robustness_results")
ACTION_GROUPS = {
    "attack": {"attack_hero", "attack_nearest_minion", "attack_tower"},
    "movement": {
        "move_up",
        "move_left",
        "move_right",
        "move_down",
        "move_up_left",
        "move_up_right",
        "move_down_left",
        "move_down_right",
    },
    "retreat": {"retreat"},
}


def _mean(values):
    values = [float(value) for value in values if value is not None]
    return sum(values) / len(values) if values else 0.0


def _std(values):
    values = [float(value) for value in values if value is not None]
    if not values:
        return 0.0
    center = _mean(values)
    variance = sum((value - center) ** 2 for value in values) / len(values)
    return variance ** 0.5


def _load_checkpoint(path):
    with Path(path).open("rb") as file:
        return pickle.load(file)


def _eval_config(base_config, opponent):
    core_config = {
        **base_config.get("core", {}),
        "opponent_strategy": opponent,
        "strategy_switch_mode": "time_based",
    }
    return {
        **base_config,
        "core": core_config,
        "opponent_strategy": opponent,
        "strategy_switch_mode": "time_based",
        "DEFAULT_NPC_POLICY": opponent,
        "NPC_POLICIES": {1: opponent},
        "NPC_POLICY_SCHEDULE": {},
        "RANDOMIZE_NPC_POLICY_EACH_EPISODE": False,
        "EXPERT_MIX_STRATEGY": "neutral",
        "EXPERT_MIX_INITIAL_RATIO": 0.0,
        "EXPERT_MIX_FINAL_RATIO": 0.0,
        "EXPERT_MIX_RATIO": 0.0,
    }


def _load_eval_config(config_arg):
    basic_config, training_config, config_name = load_run_config(config_arg)
    config = build_run_config(basic_config, training_config, config_name)
    return config_name, config


def _select_policy_action(agent, obs, deterministic):
    obs_vec = agent.encoder.encode(obs)
    belief_vec = agent.belief.update(obs)

    if not deterministic:
        action, _, _ = agent.select_action(
            obs_vec,
            belief_vec,
            use_belief_input=agent.use_belief_input,
        )
        return action

    import jax.numpy as jnp

    policy_input = agent.build_policy_input(
        obs_vec,
        belief_vec,
        use_belief_input=agent.use_belief_input,
    )
    pi, _ = agent.network.apply(
        agent.params,
        jnp.asarray(policy_input, dtype=jnp.float32),
    )
    return int(pi.mode())


def _event_counts(info):
    event_counts = info.get("event_counts", {}) if isinstance(info, dict) else {}
    return {
        "takedown_enemy_agents": int(event_counts.get("takedown_enemy_agents", 0)),
        "takedown_enemy_minions": int(event_counts.get("takedown_enemy_minions", 0)),
        "deaths": int(event_counts.get("deaths", 0)),
    }


def _action_rate(records, group_name):
    if not records:
        return 0.0
    action_names = ACTION_GROUPS[group_name]
    count = sum(1 for record in records if record["action"] in action_names)
    return count / len(records)


def _action_distribution(records):
    if not records:
        return {}
    counts = Counter(record["action"] for record in records)
    total = sum(counts.values())
    return {
        action: round(count / total, 4)
        for action, count in sorted(counts.items())
    }


def _distribution_l1(left, right):
    actions = set(left) | set(right)
    return sum(abs(float(left.get(action, 0.0)) - float(right.get(action, 0.0))) for action in actions)


def _records_in_window(step_trace, start_step, end_step):
    return [
        record
        for record in step_trace
        if start_step <= int(record["step"]) <= end_step
    ]


def _switch_analysis(step_trace, switch_step, window_size):
    if switch_step is None:
        return None

    switch_step = int(switch_step)
    window_size = int(window_size)
    pre_records = _records_in_window(
        step_trace,
        max(0, switch_step - window_size),
        switch_step - 1,
    )
    post_records = _records_in_window(
        step_trace,
        switch_step,
        switch_step + window_size - 1,
    )
    pre_actions = _action_distribution(pre_records)
    post_actions = _action_distribution(post_records)
    pre_reward_mean = _mean(record["reward"] for record in pre_records)
    post_reward_mean = _mean(record["reward"] for record in post_records)
    pre_reward_std = _std(record["reward"] for record in pre_records)
    post_reward_std = _std(record["reward"] for record in post_records)

    return {
        "window_size": window_size,
        "pre_steps": len(pre_records),
        "post_steps": len(post_records),
        "pre_reward_mean": round(pre_reward_mean, 4),
        "post_reward_mean": round(post_reward_mean, 4),
        "reward_mean_delta": round(post_reward_mean - pre_reward_mean, 4),
        "pre_reward_std": round(pre_reward_std, 4),
        "post_reward_std": round(post_reward_std, 4),
        "reward_std_delta": round(post_reward_std - pre_reward_std, 4),
        "pre_attack_rate": round(_action_rate(pre_records, "attack"), 4),
        "post_attack_rate": round(_action_rate(post_records, "attack"), 4),
        "pre_movement_rate": round(_action_rate(pre_records, "movement"), 4),
        "post_movement_rate": round(_action_rate(post_records, "movement"), 4),
        "pre_retreat_rate": round(_action_rate(pre_records, "retreat"), 4),
        "post_retreat_rate": round(_action_rate(post_records, "retreat"), 4),
        "action_distribution_shift": round(_distribution_l1(pre_actions, post_actions), 4),
        "pre_action_distribution": pre_actions,
        "post_action_distribution": post_actions,
        "pre_events": {
            "takedown_enemy_agents": sum(record["events"]["takedown_enemy_agents"] for record in pre_records),
            "takedown_enemy_minions": sum(record["events"]["takedown_enemy_minions"] for record in pre_records),
            "deaths": sum(record["events"]["deaths"] for record in pre_records),
        },
        "post_events": {
            "takedown_enemy_agents": sum(record["events"]["takedown_enemy_agents"] for record in post_records),
            "takedown_enemy_minions": sum(record["events"]["takedown_enemy_minions"] for record in post_records),
            "deaths": sum(record["events"]["deaths"] for record in post_records),
        },
    }


def _run_episode(agent, env, config, opponent, deterministic, switch_window):
    agent.set_config(config)
    episode_config = agent._config_for_episode_reset()
    agent.set_config(episode_config)
    obs_list, _ = env.reset(episode_config)
    learning_agent_id = int(episode_config.get("LEARNING_AGENT_ID", 0))
    max_steps = int(episode_config.get("MAX_STEPS_PER_EPISODE", 1000))

    episode_reward = 0.0
    action_counts = Counter()
    totals = {
        "takedown_enemy_agents": 0,
        "takedown_enemy_minions": 0,
        "deaths": 0,
    }
    terminated = False
    truncated = False
    step_trace = []

    for step in range(max_steps):
        learning_obs = obs_list[learning_agent_id]
        learning_action = _select_policy_action(agent, learning_obs, deterministic)
        learning_action_name = ACTION_NAMES.get(learning_action, str(learning_action))
        all_actions, all_policy_names = agent.select_env_actions(
            obs_list,
            learning_agent_id,
            learning_action,
            step,
            learning_policy_name="ppo_eval",
        )

        action_counts[learning_action_name] += 1
        action_for_env = [np.asarray(all_actions, dtype=np.int32)]
        obs_list, rewards, dones, truncated_list, infos = env.step(action_for_env)

        step_reward = float(rewards[learning_agent_id])
        episode_reward += step_reward
        info = infos[learning_agent_id] if learning_agent_id < len(infos) else {}
        step_events = _event_counts(info)
        for key, value in step_events.items():
            totals[key] += value

        terminated = bool(dones[learning_agent_id])
        truncated = bool(truncated_list[learning_agent_id]) or step == max_steps - 1
        step_trace.append(
            {
                "step": step,
                "action": learning_action_name,
                "reward": round(step_reward, 4),
                "cumulative_reward": round(episode_reward, 4),
                "events": step_events,
                "terminated": terminated,
                "truncated": truncated and not terminated,
            }
        )
        if terminated or truncated:
            core_config = episode_config.get("core", {})
            switch_step = core_config.get("strategy_switch_step")
            switch_analysis = _switch_analysis(step_trace, switch_step, switch_window)
            return {
                "opponent": opponent,
                "initial_opponent_strategy": core_config.get("opponent_strategy", opponent),
                "next_opponent_strategy": core_config.get("next_opponent_strategy"),
                "strategy_switch_step": switch_step,
                "episode_reward": round(episode_reward, 4),
                "terminal_reward": round(step_reward, 4),
                "episode_length": step + 1,
                "terminated": terminated,
                "truncated": truncated and not terminated,
                "takedown_enemy_agents": totals["takedown_enemy_agents"],
                "takedown_enemy_minions": totals["takedown_enemy_minions"],
                "death_count": totals["deaths"],
                "top_actions": action_counts.most_common(6),
                "all_policy_names": all_policy_names,
                "switch_analysis": switch_analysis,
                "switch_trace": (
                    _records_in_window(
                        step_trace,
                        max(0, int(switch_step) - int(switch_window)),
                        int(switch_step) + int(switch_window) - 1,
                    )
                    if switch_step is not None
                    else []
                ),
            }

    raise RuntimeError("Episode loop exited unexpectedly.")


def _episode_is_win(episode):
    if not episode.get("terminated"):
        return False
    if "terminal_reward" in episode:
        return float(episode["terminal_reward"]) > 0.0
    return float(episode.get("episode_reward", 0.0)) > 0.0


def _summarize_episodes(opponent, episodes):
    wins = sum(1 for episode in episodes if _episode_is_win(episode))
    switch_analyses = [
        episode.get("switch_analysis")
        for episode in episodes
        if episode.get("switch_analysis") is not None
    ]
    initial_counts = Counter(
        episode.get("initial_opponent_strategy", opponent)
        for episode in episodes
    )
    next_counts = Counter(
        episode.get("next_opponent_strategy")
        for episode in episodes
        if episode.get("next_opponent_strategy") is not None
    )
    return {
        "opponent": opponent,
        "episodes": len(episodes),
        "mean_reward": round(_mean(ep["episode_reward"] for ep in episodes), 4),
        "mean_length": round(_mean(ep["episode_length"] for ep in episodes), 4),
        "win_rate": round(wins / len(episodes), 4) if episodes else 0.0,
        "mean_deaths": round(_mean(ep["death_count"] for ep in episodes), 4),
        "mean_enemy_agent_takedowns": round(
            _mean(ep["takedown_enemy_agents"] for ep in episodes),
            4,
        ),
        "mean_enemy_minion_takedowns": round(
            _mean(ep["takedown_enemy_minions"] for ep in episodes),
            4,
        ),
        "initial_strategy_counts": dict(initial_counts),
        "next_strategy_counts": dict(next_counts),
        "mean_switch_reward_delta": round(
            _mean(item.get("reward_mean_delta") for item in switch_analyses),
            4,
        ),
        "mean_switch_reward_std_delta": round(
            _mean(item.get("reward_std_delta") for item in switch_analyses),
            4,
        ),
        "mean_switch_action_shift": round(
            _mean(item.get("action_distribution_shift") for item in switch_analyses),
            4,
        ),
        "mean_pre_switch_attack_rate": round(
            _mean(item.get("pre_attack_rate") for item in switch_analyses),
            4,
        ),
        "mean_post_switch_attack_rate": round(
            _mean(item.get("post_attack_rate") for item in switch_analyses),
            4,
        ),
        "mean_pre_switch_retreat_rate": round(
            _mean(item.get("pre_retreat_rate") for item in switch_analyses),
            4,
        ),
        "mean_post_switch_retreat_rate": round(
            _mean(item.get("post_retreat_rate") for item in switch_analyses),
            4,
        ),
    }


def _markdown_report(checkpoint_path, deterministic, eval_config, summaries):
    lines = [
        "# Robustness Evaluation",
        "",
        f"- Checkpoint: `{checkpoint_path}`",
        f"- Policy mode: `{'deterministic' if deterministic else 'stochastic'}`",
        f"- Evaluation config: `{eval_config or 'fixed_opponents'}`",
        "",
        "## Summary",
        "",
    ]
    headers = [
        "opponent",
        "episodes",
        "mean_reward",
        "mean_length",
        "win_rate",
        "mean_deaths",
        "mean_enemy_agent_takedowns",
        "mean_enemy_minion_takedowns",
        "initial_strategy_counts",
        "next_strategy_counts",
        "mean_switch_reward_delta",
        "mean_switch_reward_std_delta",
        "mean_switch_action_shift",
        "mean_pre_switch_attack_rate",
        "mean_post_switch_attack_rate",
        "mean_pre_switch_retreat_rate",
        "mean_post_switch_retreat_rate",
    ]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for summary in summaries:
        lines.append("| " + " | ".join(str(summary[h]) for h in headers) + " |")
    lines.append("")
    return "\n".join(lines)


def run_robustness_test(args):
    from core.beliefs.dummy_belief import DummyBelief
    from core.models.ppo import PPO
    from core.utils.obs_encoder import ObservationEncoder

    checkpoint = _load_checkpoint(args.checkpoint)
    if args.eval_config:
        eval_config_name, base_config = _load_eval_config(args.eval_config)
    else:
        eval_config_name = None
        base_config = dict(checkpoint["config"])

    env = make_env(base_config)
    encoder = ObservationEncoder()
    belief = DummyBelief()
    agent = PPO(env=env, encoder=encoder, belief=belief, config=base_config)
    agent.params = checkpoint["params"]
    agent.opt_state = checkpoint.get("opt_state", agent.opt_state)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_name = Path(args.checkpoint).parents[1].name
    suffix = eval_config_name or "fixed_opponents"
    output_dir = Path(args.output_dir) / f"{checkpoint_name}_{suffix}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    summaries = []
    try:
        eval_items = [eval_config_name] if args.eval_config else args.opponents
        for opponent in eval_items:
            config = base_config if args.eval_config else _eval_config(base_config, opponent)
            agent.set_config(config)
            episodes = []
            for episode_idx in range(1, args.episodes + 1):
                result = _run_episode(
                    agent,
                    env,
                    config,
                    opponent,
                    deterministic=not args.stochastic,
                    switch_window=args.switch_window,
                )
                result["episode"] = episode_idx
                episodes.append(result)
                print(
                    f"[{opponent}] episode={episode_idx}/{args.episodes} "
                    f"reward={result['episode_reward']:.2f} "
                    f"length={result['episode_length']} "
                    f"deaths={result['death_count']}"
                )

            all_results[opponent] = episodes
            summaries.append(_summarize_episodes(opponent, episodes))
    finally:
        env.close()

    result_payload = {
        "checkpoint": str(args.checkpoint),
        "eval_config": args.eval_config,
        "deterministic": not args.stochastic,
        "switch_window": args.switch_window,
        "summaries": summaries,
        "episodes": all_results,
    }
    json_path = output_dir / "robustness_results.json"
    json_path.write_text(json.dumps(result_payload, indent=2, default=str), encoding="utf-8")

    print(f"Saved robustness JSON to: {json_path}")
    print()
    print(_markdown_report(args.checkpoint, not args.stochastic, eval_config_name, summaries))


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate checkpoint robustness across opponents.")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoints/final.pkl.")
    parser.add_argument(
        "--eval-config",
        default=None,
        help=(
            "Optional config module/file for evaluation, e.g. "
            "`strategySwitching_onlyOBS`. When set, --opponents is ignored."
        ),
    )
    parser.add_argument(
        "--opponents",
        nargs="+",
        default=DEFAULT_OPPONENTS,
        help="Opponent strategies to evaluate against.",
    )
    parser.add_argument("--episodes", type=int, default=50, help="Episodes per opponent.")
    parser.add_argument(
        "--switch-window",
        type=int,
        default=50,
        help="Number of steps before and after a strategy switch to analyze.",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample actions instead of using deterministic policy mode.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Root output directory for robustness results.",
    )
    return parser.parse_args()


def main():
    run_robustness_test(parse_args())


if __name__ == "__main__":
    main()

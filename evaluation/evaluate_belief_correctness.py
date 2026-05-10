#!/usr/bin/env python3
"""Evaluate BayesianBelief correctness without a trained PPO checkpoint.

This script is a sanity check for belief propagation. It runs scripted agents in
the simulator, feeds the learning agent's observations into BayesianBelief, and
compares the inferred strategy against the known scripted opponent strategy.

Examples:
    python evaluation/evaluate_belief_correctness.py \
        --eval-config strategySwitching_onlyOBS \
        --episodes 50

    python evaluation/evaluate_belief_correctness.py \
        --opponents aggressive neutral farming \
        --episodes 20
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import json
import math
from pathlib import Path
import sys

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.run import ACTION_NAMES, build_run_config, load_run_config, make_env
from core.strategy.basic_strategy import AggressiveStrategy, FarmingStrategy, NeutralStrategy
from core.utils.obs_encoder import unwrap_obs


STRATEGIES = ["aggressive", "neutral", "farming"]
DEFAULT_OUTPUT_ROOT = Path("evaluation/belief_correctness_results")
DEFAULT_OPPONENTS = ["aggressive", "neutral", "farming"]
STRATEGY_REGISTRY = {
    AggressiveStrategy.name: AggressiveStrategy,
    FarmingStrategy.name: FarmingStrategy,
    NeutralStrategy.name: NeutralStrategy,
}


def _mean(values):
    values = [float(value) for value in values if value is not None]
    return sum(values) / len(values) if values else 0.0


def _load_eval_config(config_arg):
    basic_config, training_config, config_name = load_run_config(config_arg)
    config = build_run_config(basic_config, training_config, config_name)
    return config_name, config


def _fixed_opponent_config(base_config, opponent):
    return {
        **base_config,
        "core": {
            **base_config.get("core", {}),
            "opponent_strategy": opponent,
            "strategy_switch_mode": "time_based",
            "strategy_switch_step": None,
            "next_opponent_strategy": None,
        },
        "opponent_strategy": opponent,
        "strategy_switch_mode": "time_based",
        "DEFAULT_NPC_POLICY": opponent,
        "NPC_POLICIES": {1: opponent},
        "NPC_POLICY_SCHEDULE": {},
        "RANDOMIZE_NPC_POLICY_EACH_EPISODE": False,
    }


def _episode_config(base_config, rng):
    if not base_config.get("RANDOMIZE_NPC_POLICY_EACH_EPISODE", False):
        return base_config

    strategy_pool = list(base_config.get("NPC_STRATEGY_POOL", STRATEGIES))
    if len(strategy_pool) < 2:
        return base_config

    max_steps = int(base_config.get("MAX_STEPS_PER_EPISODE", 1000))
    switch_min, switch_max = base_config.get("NPC_SWITCH_STEP_RANGE", [1, max_steps - 1])
    switch_min = max(1, int(switch_min))
    switch_max = min(int(switch_max), max_steps - 1)
    switch_step = int(rng.integers(switch_min, switch_max + 1))
    initial_strategy, next_strategy = rng.choice(
        strategy_pool,
        size=2,
        replace=False,
    ).tolist()

    npc_policy_ids = base_config.get("NPC_POLICY_IDS")
    if npc_policy_ids is None:
        npc_policy_ids = list(base_config.get("NPC_POLICIES", {}).keys()) or [1]

    npc_policies = {policy_id: initial_strategy for policy_id in npc_policy_ids}
    npc_schedule = {
        policy_id: [
            {"start_step": 0, "strategy": initial_strategy},
            {"start_step": switch_step, "strategy": next_strategy},
        ]
        for policy_id in npc_policy_ids
    }
    core_config = {
        **base_config.get("core", {}),
        "opponent_strategy": initial_strategy,
        "strategy_switch_mode": "random_time",
        "strategy_switch_step": switch_step,
        "next_opponent_strategy": next_strategy,
    }
    return {
        **base_config,
        "core": core_config,
        "NPC_POLICIES": npc_policies,
        "NPC_POLICY_SCHEDULE": npc_schedule,
    }


def _strategy_for_step(episode_config, step, fallback):
    core_config = episode_config.get("core", {})
    initial = core_config.get("opponent_strategy", fallback)
    switch_step = core_config.get("strategy_switch_step")
    next_strategy = core_config.get("next_opponent_strategy")
    if switch_step is not None and next_strategy is not None and step >= int(switch_step):
        return next_strategy
    return initial


def _policy_for_strategy(strategy_name, cache):
    if strategy_name not in cache:
        strategy_cls = STRATEGY_REGISTRY[strategy_name]
        cache[strategy_name] = strategy_cls()
    return cache[strategy_name]


def _belief_dict(belief_vec):
    values = np.asarray(belief_vec, dtype=np.float32).reshape(-1)
    total = float(values.sum())
    if total <= 0:
        values = np.full(len(STRATEGIES), 1.0 / len(STRATEGIES), dtype=np.float32)
    else:
        values = values / total
    return {
        strategy: round(float(value), 6)
        for strategy, value in zip(STRATEGIES, values)
    }


def _predicted_strategy(belief):
    return max(belief.items(), key=lambda item: item[1])[0]


def _entropy(belief):
    value = 0.0
    for probability in belief.values():
        p = max(float(probability), 1e-8)
        value -= p * math.log(p)
    return value


def _detection_delay(trace, switch_step, next_strategy):
    if switch_step is None or next_strategy is None:
        return None
    for record in trace:
        if int(record["step"]) >= int(switch_step) and record["predicted_strategy"] == next_strategy:
            return int(record["step"]) - int(switch_step)
    return None


def _belief_step_shift(trace):
    if len(trace) < 2:
        return 0.0
    shifts = []
    for previous, current in zip(trace[:-1], trace[1:]):
        shifts.append(
            sum(
                abs(current["belief"][strategy] - previous["belief"][strategy])
                for strategy in STRATEGIES
            )
        )
    return _mean(shifts)


def _confusion_matrix(trace):
    matrix = {
        true_strategy: {predicted: 0 for predicted in STRATEGIES}
        for true_strategy in STRATEGIES
    }
    for record in trace:
        true_strategy = record["true_strategy"]
        predicted = record["predicted_strategy"]
        if true_strategy in matrix and predicted in matrix[true_strategy]:
            matrix[true_strategy][predicted] += 1
    return matrix


def _summarize_trace(trace, switch_step, next_strategy):
    pre_trace = [
        record for record in trace
        if switch_step is None or int(record["step"]) < int(switch_step)
    ]
    post_trace = [
        record for record in trace
        if switch_step is not None and int(record["step"]) >= int(switch_step)
    ]

    def accuracy(records):
        if not records:
            return 0.0
        return sum(1 for record in records if record["belief_correct"]) / len(records)

    return {
        "overall_accuracy": round(accuracy(trace), 4),
        "pre_switch_accuracy": round(accuracy(pre_trace), 4),
        "post_switch_accuracy": round(accuracy(post_trace), 4),
        "switch_detection_delay": _detection_delay(trace, switch_step, next_strategy),
        "mean_entropy": round(_mean(record["entropy"] for record in trace), 4),
        "mean_belief_step_shift": round(_belief_step_shift(trace), 4),
        "confusion_matrix": _confusion_matrix(trace),
    }


def _run_episode(env, base_config, label, learning_strategy_name, rng):
    from core.beliefs.bayesian_belief import BayesianBelief

    belief = BayesianBelief()
    episode_config = _episode_config(base_config, rng)
    obs_list, _ = env.reset(episode_config)

    max_steps = int(episode_config.get("MAX_STEPS_PER_EPISODE", 1000))
    learning_agent_id = int(episode_config.get("LEARNING_AGENT_ID", 0))
    opponent_agent_id = 1 if learning_agent_id == 0 else 0
    core_config = episode_config.get("core", {})
    switch_step = core_config.get("strategy_switch_step")
    next_strategy = core_config.get("next_opponent_strategy")

    policy_cache = {}
    trace = []
    episode_reward = 0.0
    action_counts = Counter()

    for step in range(max_steps):
        learning_obs = obs_list[learning_agent_id]
        opponent_obs = obs_list[opponent_agent_id]
        true_strategy = _strategy_for_step(episode_config, step, label)

        belief_vec = belief.update(learning_obs)
        belief_distribution = _belief_dict(belief_vec)
        predicted = _predicted_strategy(belief_distribution)

        learning_policy = _policy_for_strategy(learning_strategy_name, policy_cache)
        opponent_policy = _policy_for_strategy(true_strategy, policy_cache)
        learning_action = learning_policy.select_action(unwrap_obs(learning_obs))
        opponent_action = opponent_policy.select_action(unwrap_obs(opponent_obs))

        all_actions = [None] * len(obs_list)
        all_actions[learning_agent_id] = learning_action
        all_actions[opponent_agent_id] = opponent_action

        obs_list, rewards, dones, truncated_list, infos = env.step(
            [np.asarray(all_actions, dtype=np.int32)]
        )

        reward = float(rewards[learning_agent_id])
        episode_reward += reward
        action_counts[ACTION_NAMES.get(learning_action, str(learning_action))] += 1
        terminated = bool(dones[learning_agent_id])
        truncated = bool(truncated_list[learning_agent_id]) or step == max_steps - 1

        trace.append(
            {
                "step": step,
                "true_strategy": true_strategy,
                "predicted_strategy": predicted,
                "belief_correct": predicted == true_strategy,
                "belief": belief_distribution,
                "entropy": round(_entropy(belief_distribution), 4),
                "learning_action": ACTION_NAMES.get(learning_action, str(learning_action)),
                "opponent_action": ACTION_NAMES.get(opponent_action, str(opponent_action)),
                "reward": round(reward, 4),
                "cumulative_reward": round(episode_reward, 4),
                "terminated": terminated,
                "truncated": truncated and not terminated,
            }
        )

        if terminated or truncated:
            summary = _summarize_trace(trace, switch_step, next_strategy)
            return {
                "label": label,
                "initial_opponent_strategy": core_config.get("opponent_strategy", label),
                "next_opponent_strategy": next_strategy,
                "strategy_switch_step": switch_step,
                "episode_reward": round(episode_reward, 4),
                "episode_length": step + 1,
                "terminated": terminated,
                "truncated": truncated and not terminated,
                "top_learning_actions": action_counts.most_common(6),
                "belief_summary": summary,
                "belief_trace": trace,
            }

    raise RuntimeError("Episode loop exited unexpectedly.")


def _aggregate_confusion(episodes):
    matrix = {
        true_strategy: {predicted: 0 for predicted in STRATEGIES}
        for true_strategy in STRATEGIES
    }
    for episode in episodes:
        episode_matrix = episode["belief_summary"]["confusion_matrix"]
        for true_strategy in STRATEGIES:
            for predicted in STRATEGIES:
                matrix[true_strategy][predicted] += episode_matrix[true_strategy][predicted]
    return matrix


def _summarize_episodes(label, episodes):
    delays = [
        episode["belief_summary"]["switch_detection_delay"]
        for episode in episodes
        if episode["belief_summary"]["switch_detection_delay"] is not None
    ]
    return {
        "label": label,
        "episodes": len(episodes),
        "mean_episode_reward": round(_mean(ep["episode_reward"] for ep in episodes), 4),
        "mean_episode_length": round(_mean(ep["episode_length"] for ep in episodes), 4),
        "overall_accuracy": round(
            _mean(ep["belief_summary"]["overall_accuracy"] for ep in episodes),
            4,
        ),
        "pre_switch_accuracy": round(
            _mean(ep["belief_summary"]["pre_switch_accuracy"] for ep in episodes),
            4,
        ),
        "post_switch_accuracy": round(
            _mean(ep["belief_summary"]["post_switch_accuracy"] for ep in episodes),
            4,
        ),
        "mean_switch_detection_delay": round(_mean(delays), 4) if delays else None,
        "mean_entropy": round(
            _mean(ep["belief_summary"]["mean_entropy"] for ep in episodes),
            4,
        ),
        "mean_belief_step_shift": round(
            _mean(ep["belief_summary"]["mean_belief_step_shift"] for ep in episodes),
            4,
        ),
        "confusion_matrix": _aggregate_confusion(episodes),
    }


def _markdown_report(eval_config_name, learning_strategy, summaries):
    headers = [
        "label",
        "episodes",
        "overall_accuracy",
        "pre_switch_accuracy",
        "post_switch_accuracy",
        "mean_switch_detection_delay",
        "mean_entropy",
        "mean_belief_step_shift",
        "mean_episode_reward",
    ]
    lines = [
        "# Belief Correctness Evaluation",
        "",
        f"- Evaluation config: `{eval_config_name or 'fixed_opponents'}`",
        f"- Learning agent scripted strategy: `{learning_strategy}`",
        "",
        "## Summary",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for summary in summaries:
        lines.append("| " + " | ".join(str(summary.get(header)) for header in headers) + " |")

    lines.extend(["", "## Confusion Matrices", ""])
    for summary in summaries:
        lines.append(f"### {summary['label']}")
        lines.append("")
        lines.append("| true \\ predicted | aggressive | neutral | farming |")
        lines.append("| --- | --- | --- | --- |")
        matrix = summary["confusion_matrix"]
        for true_strategy in STRATEGIES:
            row = [str(matrix[true_strategy][predicted]) for predicted in STRATEGIES]
            lines.append(f"| {true_strategy} | " + " | ".join(row) + " |")
        lines.append("")
    return "\n".join(lines)


def run_belief_correctness(args):
    if args.learning_strategy not in STRATEGY_REGISTRY:
        raise ValueError(f"Unknown learning strategy: {args.learning_strategy}")

    if args.eval_config:
        eval_config_name, base_config = _load_eval_config(args.eval_config)
        labels = [eval_config_name]
    else:
        eval_config_name = None
        _, base_config = _load_eval_config(args.base_config)
        labels = args.opponents

    rng = np.random.default_rng(args.seed)
    env = make_env(base_config)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = eval_config_name or "fixed_opponents"
    output_dir = Path(args.output_dir) / f"{suffix}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    summaries = []
    try:
        for label in labels:
            config = base_config if args.eval_config else _fixed_opponent_config(base_config, label)
            episodes = []
            for episode_idx in range(1, args.episodes + 1):
                result = _run_episode(
                    env=env,
                    base_config=config,
                    label=label,
                    learning_strategy_name=args.learning_strategy,
                    rng=rng,
                )
                result["episode"] = episode_idx
                episodes.append(result)
                summary = result["belief_summary"]
                print(
                    f"[{label}] episode={episode_idx}/{args.episodes} "
                    f"acc={summary['overall_accuracy']:.3f} "
                    f"post={summary['post_switch_accuracy']:.3f} "
                    f"delay={summary['switch_detection_delay']}"
                )
            all_results[label] = episodes
            summaries.append(_summarize_episodes(label, episodes))
    finally:
        env.close()

    payload = {
        "eval_config": args.eval_config,
        "base_config": args.base_config if not args.eval_config else None,
        "learning_strategy": args.learning_strategy,
        "episodes_per_label": args.episodes,
        "summaries": summaries,
        "episodes": all_results,
    }
    json_path = output_dir / "belief_correctness_results.json"
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"Saved belief correctness JSON to: {json_path}")
    print()
    print(_markdown_report(eval_config_name, args.learning_strategy, summaries))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate BayesianBelief correctness using scripted opponents.",
    )
    parser.add_argument(
        "--eval-config",
        default=None,
        help="Optional switching config, e.g. strategySwitching_onlyOBS.",
    )
    parser.add_argument(
        "--base-config",
        default="aggressiveExpert_neutral",
        help="Base config used for fixed-opponent tests when --eval-config is omitted.",
    )
    parser.add_argument(
        "--opponents",
        nargs="+",
        default=DEFAULT_OPPONENTS,
        help="Fixed opponents to evaluate when --eval-config is omitted.",
    )
    parser.add_argument("--episodes", type=int, default=50, help="Episodes per label.")
    parser.add_argument(
        "--learning-strategy",
        default="neutral",
        choices=sorted(STRATEGY_REGISTRY),
        help="Scripted policy used by the observed learning-side agent.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Root output directory.",
    )
    return parser.parse_args()


def main():
    run_belief_correctness(parse_args())


if __name__ == "__main__":
    main()

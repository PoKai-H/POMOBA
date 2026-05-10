#!/usr/bin/env python3
"""Evaluate checkpoint robustness while logging belief inference traces.

This script is intended for the belief-specific evaluation in the project plan.
It records the inferred belief distribution at each step, compares it with the
ground-truth scripted opponent strategy, and reports strategy inference and
switch-detection metrics alongside downstream reward metrics.

Examples:
    python evaluation/robustness_belief_test.py \
        --checkpoint outputs/training_analysis/belief_run_x/checkpoints/final.pkl \
        --eval-config strategySwitching_onlyOBS

    python evaluation/robustness_belief_test.py \
        --checkpoint outputs/training_analysis/belief_run_x/checkpoints/final.pkl \
        --eval-config strategySwitching_onlyOBS \
        --belief-mode shuffled
"""

from __future__ import annotations

import argparse
from collections import Counter, deque
from datetime import datetime
import json
import math
import pickle
from pathlib import Path
import sys

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.run import ACTION_NAMES, build_run_config, load_run_config, make_env


STRATEGIES = ["aggressive", "neutral", "farming"]
DEFAULT_OUTPUT_ROOT = Path("evaluation/belief_results")
DEFAULT_OPPONENTS = ["aggressive", "neutral", "farming"]


def _mean(values):
    values = [float(value) for value in values if value is not None]
    return sum(values) / len(values) if values else 0.0


def _load_checkpoint(path):
    with Path(path).open("rb") as file:
        return pickle.load(file)


def _load_eval_config(config_arg):
    basic_config, training_config, config_name = load_run_config(config_arg)
    config = build_run_config(basic_config, training_config, config_name)
    return config_name, config


def _merge_eval_config(checkpoint_config, eval_config, eval_config_name):
    """Use eval opponent settings while preserving checkpoint model shape."""
    merged = dict(checkpoint_config)
    merged["CONFIG_NAME"] = eval_config_name
    merged["core"] = {
        **checkpoint_config.get("core", {}),
        **eval_config.get("core", {}),
    }
    merged["extensions"] = {
        **checkpoint_config.get("extensions", {}),
        **eval_config.get("extensions", {}),
    }

    eval_keys = [
        "opponent_strategy",
        "strategy_switch_mode",
        "DEFAULT_NPC_POLICY",
        "NPC_POLICIES",
        "NPC_POLICY_SCHEDULE",
        "RANDOMIZE_NPC_POLICY_EACH_EPISODE",
        "NPC_POLICY_IDS",
        "NPC_STRATEGY_POOL",
        "NPC_SWITCH_STEP_RANGE",
        "MAX_STEPS_PER_EPISODE",
    ]
    for key in eval_keys:
        if key in eval_config:
            merged[key] = eval_config[key]

    # Keep the checkpoint architecture unchanged.
    merged["USE_BELIEF_INPUT"] = checkpoint_config.get("USE_BELIEF_INPUT", False)
    return merged


def _fixed_opponent_config(base_config, opponent):
    return {
        **base_config,
        "core": {
            **base_config.get("core", {}),
            "opponent_strategy": opponent,
            "strategy_switch_mode": "time_based",
            "next_opponent_strategy": None,
            "strategy_switch_step": None,
        },
        "opponent_strategy": opponent,
        "strategy_switch_mode": "time_based",
        "DEFAULT_NPC_POLICY": opponent,
        "NPC_POLICIES": {1: opponent},
        "NPC_POLICY_SCHEDULE": {},
        "RANDOMIZE_NPC_POLICY_EACH_EPISODE": False,
        "EXPERT_MIX_RATIO": 0.0,
        "EXPERT_MIX_INITIAL_RATIO": 0.0,
        "EXPERT_MIX_FINAL_RATIO": 0.0,
    }


def _event_counts(info):
    event_counts = info.get("event_counts", {}) if isinstance(info, dict) else {}
    return {
        "takedown_enemy_agents": int(event_counts.get("takedown_enemy_agents", 0)),
        "takedown_enemy_minions": int(event_counts.get("takedown_enemy_minions", 0)),
        "deaths": int(event_counts.get("deaths", 0)),
    }


def _to_belief_dict(belief_vec):
    values = np.asarray(belief_vec, dtype=np.float32).reshape(-1)
    if values.size != len(STRATEGIES):
        raise ValueError(f"Expected {len(STRATEGIES)} belief values, got {values.size}.")
    total = float(values.sum())
    if total <= 0:
        values = np.full(len(STRATEGIES), 1.0 / len(STRATEGIES), dtype=np.float32)
    else:
        values = values / total
    return {
        strategy: round(float(value), 6)
        for strategy, value in zip(STRATEGIES, values)
    }


def _belief_entropy(belief_dict):
    entropy = 0.0
    for value in belief_dict.values():
        p = max(float(value), 1e-8)
        entropy -= p * math.log(p)
    return entropy


def _predicted_strategy(belief_dict):
    return max(belief_dict.items(), key=lambda item: item[1])[0]


def _true_strategy_for_step(core_config, fallback_opponent, step):
    initial = core_config.get("opponent_strategy", fallback_opponent)
    switch_step = core_config.get("strategy_switch_step")
    next_strategy = core_config.get("next_opponent_strategy")
    if switch_step is not None and next_strategy is not None and step >= int(switch_step):
        return next_strategy
    return initial


def _policy_belief_vector(correct_belief_vec, mode, rng, lag_buffer, lag_steps):
    correct = np.asarray(correct_belief_vec, dtype=np.float32)
    if mode == "correct":
        policy_belief = correct
    elif mode == "random":
        policy_belief = rng.dirichlet(np.ones(len(STRATEGIES))).astype(np.float32)
    elif mode == "shuffled":
        policy_belief = correct.copy()
        rng.shuffle(policy_belief)
    elif mode == "lagged":
        if len(lag_buffer) >= lag_steps:
            policy_belief = np.asarray(lag_buffer[0], dtype=np.float32)
        else:
            policy_belief = np.full(len(STRATEGIES), 1.0 / len(STRATEGIES), dtype=np.float32)
    else:
        raise ValueError(f"Unknown belief mode: {mode}")

    lag_buffer.append(correct.copy())
    return policy_belief


def _select_policy_action(agent, obs, belief_vec, deterministic):
    obs_vec = agent.encoder.encode(obs)
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


def _episode_is_win(episode):
    return bool(episode.get("terminated")) and float(episode.get("terminal_reward", 0.0)) > 0.0


def _detection_delay(trace, switch_step, next_strategy):
    if switch_step is None or next_strategy is None:
        return None
    for record in trace:
        if int(record["step"]) >= int(switch_step) and record["predicted_strategy"] == next_strategy:
            return int(record["step"]) - int(switch_step)
    return None


def _belief_step_stability(trace):
    if len(trace) < 2:
        return 0.0
    shifts = []
    for prev, curr in zip(trace[:-1], trace[1:]):
        prev_b = prev["belief"]
        curr_b = curr["belief"]
        shifts.append(sum(abs(curr_b[name] - prev_b[name]) for name in STRATEGIES))
    return _mean(shifts)


def _belief_summary(trace, switch_step, next_strategy):
    if not trace:
        return {
            "belief_accuracy": 0.0,
            "pre_switch_accuracy": 0.0,
            "post_switch_accuracy": 0.0,
            "switch_detection_delay": None,
            "mean_belief_entropy": 0.0,
            "mean_belief_step_shift": 0.0,
        }

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
        "belief_accuracy": round(accuracy(trace), 4),
        "pre_switch_accuracy": round(accuracy(pre_trace), 4),
        "post_switch_accuracy": round(accuracy(post_trace), 4),
        "switch_detection_delay": _detection_delay(trace, switch_step, next_strategy),
        "mean_belief_entropy": round(_mean(record["belief_entropy"] for record in trace), 4),
        "mean_belief_step_shift": round(_belief_step_stability(trace), 4),
    }


def _run_episode(
    agent,
    env,
    config,
    opponent_label,
    deterministic,
    belief_mode,
    lag_steps,
    rng,
):
    from core.beliefs.bayesian_belief import BayesianBelief

    agent.belief = BayesianBelief()
    agent.set_config(config)
    episode_config = agent._config_for_episode_reset()
    agent.set_config(episode_config)
    obs_list, _ = env.reset(episode_config)

    learning_agent_id = int(episode_config.get("LEARNING_AGENT_ID", 0))
    max_steps = int(episode_config.get("MAX_STEPS_PER_EPISODE", 1000))
    core_config = episode_config.get("core", {})
    switch_step = core_config.get("strategy_switch_step")
    next_strategy = core_config.get("next_opponent_strategy")

    episode_reward = 0.0
    action_counts = Counter()
    totals = {
        "takedown_enemy_agents": 0,
        "takedown_enemy_minions": 0,
        "deaths": 0,
    }
    belief_trace = []
    lag_buffer = deque(maxlen=max(1, int(lag_steps)))

    for step in range(max_steps):
        learning_obs = obs_list[learning_agent_id]
        correct_belief_vec = agent.belief.update(learning_obs)
        correct_belief = _to_belief_dict(correct_belief_vec)
        policy_belief_vec = _policy_belief_vector(
            correct_belief_vec,
            belief_mode,
            rng,
            lag_buffer,
            lag_steps,
        )
        policy_belief = _to_belief_dict(policy_belief_vec)
        predicted = _predicted_strategy(correct_belief)
        true_strategy = _true_strategy_for_step(core_config, opponent_label, step)

        learning_action = _select_policy_action(
            agent,
            learning_obs,
            policy_belief_vec,
            deterministic,
        )
        learning_action_name = ACTION_NAMES.get(learning_action, str(learning_action))
        all_actions, all_policy_names = agent.select_env_actions(
            obs_list,
            learning_agent_id,
            learning_action,
            step,
            learning_policy_name=f"ppo_{belief_mode}_belief",
        )

        action_counts[learning_action_name] += 1
        obs_list, rewards, dones, truncated_list, infos = env.step(
            [np.asarray(all_actions, dtype=np.int32)]
        )

        step_reward = float(rewards[learning_agent_id])
        episode_reward += step_reward
        info = infos[learning_agent_id] if learning_agent_id < len(infos) else {}
        step_events = _event_counts(info)
        for key, value in step_events.items():
            totals[key] += value

        terminated = bool(dones[learning_agent_id])
        truncated = bool(truncated_list[learning_agent_id]) or step == max_steps - 1
        belief_trace.append(
            {
                "step": step,
                "true_strategy": true_strategy,
                "predicted_strategy": predicted,
                "belief_correct": predicted == true_strategy,
                "belief": correct_belief,
                "policy_belief": policy_belief,
                "belief_entropy": round(_belief_entropy(correct_belief), 4),
                "action": learning_action_name,
                "reward": round(step_reward, 4),
                "cumulative_reward": round(episode_reward, 4),
                "events": step_events,
                "terminated": terminated,
                "truncated": truncated and not terminated,
            }
        )

        if terminated or truncated:
            summary = _belief_summary(belief_trace, switch_step, next_strategy)
            return {
                "opponent": opponent_label,
                "initial_opponent_strategy": core_config.get("opponent_strategy", opponent_label),
                "next_opponent_strategy": next_strategy,
                "strategy_switch_step": switch_step,
                "belief_mode": belief_mode,
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
                "belief_summary": summary,
                "belief_trace": belief_trace,
            }

    raise RuntimeError("Episode loop exited unexpectedly.")


def _summarize_episodes(label, episodes):
    delays = [
        ep["belief_summary"].get("switch_detection_delay")
        for ep in episodes
        if ep["belief_summary"].get("switch_detection_delay") is not None
    ]
    return {
        "label": label,
        "episodes": len(episodes),
        "mean_reward": round(_mean(ep["episode_reward"] for ep in episodes), 4),
        "win_rate": round(
            sum(1 for ep in episodes if _episode_is_win(ep)) / len(episodes),
            4,
        ) if episodes else 0.0,
        "mean_length": round(_mean(ep["episode_length"] for ep in episodes), 4),
        "mean_deaths": round(_mean(ep["death_count"] for ep in episodes), 4),
        "mean_enemy_agent_takedowns": round(
            _mean(ep["takedown_enemy_agents"] for ep in episodes),
            4,
        ),
        "mean_enemy_minion_takedowns": round(
            _mean(ep["takedown_enemy_minions"] for ep in episodes),
            4,
        ),
        "belief_accuracy": round(
            _mean(ep["belief_summary"]["belief_accuracy"] for ep in episodes),
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
        "mean_belief_entropy": round(
            _mean(ep["belief_summary"]["mean_belief_entropy"] for ep in episodes),
            4,
        ),
        "mean_belief_step_shift": round(
            _mean(ep["belief_summary"]["mean_belief_step_shift"] for ep in episodes),
            4,
        ),
    }


def _markdown_report(checkpoint_path, eval_config, belief_mode, summaries):
    headers = [
        "label",
        "episodes",
        "mean_reward",
        "win_rate",
        "mean_deaths",
        "belief_accuracy",
        "pre_switch_accuracy",
        "post_switch_accuracy",
        "mean_switch_detection_delay",
        "mean_belief_entropy",
        "mean_belief_step_shift",
    ]
    lines = [
        "# Belief Robustness Evaluation",
        "",
        f"- Checkpoint: `{checkpoint_path}`",
        f"- Evaluation config: `{eval_config or 'fixed_opponents'}`",
        f"- Belief mode used by policy: `{belief_mode}`",
        "",
        "## Summary",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for summary in summaries:
        lines.append("| " + " | ".join(str(summary.get(header)) for header in headers) + " |")
    lines.append("")
    return "\n".join(lines)


def run_belief_test(args):
    from core.beliefs.bayesian_belief import BayesianBelief
    from core.models.ppo import PPO
    from core.utils.obs_encoder import ObservationEncoder

    checkpoint = _load_checkpoint(args.checkpoint)
    checkpoint_config = dict(checkpoint["config"])
    if args.eval_config:
        eval_config_name, eval_config = _load_eval_config(args.eval_config)
        base_config = _merge_eval_config(checkpoint_config, eval_config, eval_config_name)
    else:
        eval_config_name = None
        base_config = checkpoint_config

    env = make_env(base_config)
    agent = PPO(
        env=env,
        encoder=ObservationEncoder(),
        belief=BayesianBelief(),
        config=base_config,
    )
    agent.params = checkpoint["params"]
    agent.opt_state = checkpoint.get("opt_state", agent.opt_state)

    rng = np.random.default_rng(args.seed)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_name = Path(args.checkpoint).parents[1].name
    suffix = eval_config_name or "fixed_opponents"
    output_dir = (
        Path(args.output_dir)
        / f"{checkpoint_name}_{suffix}_{args.belief_mode}_{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    summaries = []
    try:
        labels = [eval_config_name] if args.eval_config else args.opponents
        for label in labels:
            config = base_config if args.eval_config else _fixed_opponent_config(base_config, label)
            episodes = []
            for episode_idx in range(1, args.episodes + 1):
                result = _run_episode(
                    agent=agent,
                    env=env,
                    config=config,
                    opponent_label=label,
                    deterministic=not args.stochastic,
                    belief_mode=args.belief_mode,
                    lag_steps=args.lag_steps,
                    rng=rng,
                )
                result["episode"] = episode_idx
                episodes.append(result)
                belief_summary = result["belief_summary"]
                print(
                    f"[{label}] episode={episode_idx}/{args.episodes} "
                    f"reward={result['episode_reward']:.2f} "
                    f"belief_acc={belief_summary['belief_accuracy']:.3f} "
                    f"delay={belief_summary['switch_detection_delay']}"
                )
            all_results[label] = episodes
            summaries.append(_summarize_episodes(label, episodes))
    finally:
        env.close()

    payload = {
        "checkpoint": str(args.checkpoint),
        "eval_config": args.eval_config,
        "belief_mode": args.belief_mode,
        "lag_steps": args.lag_steps,
        "deterministic": not args.stochastic,
        "summaries": summaries,
        "episodes": all_results,
    }
    json_path = output_dir / "belief_robustness_results.json"
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"Saved belief robustness JSON to: {json_path}")
    print()
    print(_markdown_report(args.checkpoint, args.eval_config, args.belief_mode, summaries))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a checkpoint and log belief inference traces.",
    )
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoints/final.pkl.")
    parser.add_argument(
        "--eval-config",
        default=None,
        help="Optional switching evaluation config, e.g. strategySwitching_onlyOBS.",
    )
    parser.add_argument(
        "--opponents",
        nargs="+",
        default=DEFAULT_OPPONENTS,
        help="Fixed opponents to evaluate when --eval-config is not provided.",
    )
    parser.add_argument("--episodes", type=int, default=50, help="Episodes per setting.")
    parser.add_argument(
        "--belief-mode",
        choices=["correct", "random", "shuffled", "lagged"],
        default="correct",
        help="Belief vector fed to the policy. The logged belief trace always records Bayesian belief.",
    )
    parser.add_argument(
        "--lag-steps",
        type=int,
        default=25,
        help="Number of steps to lag belief input when --belief-mode lagged is used.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for perturbed belief modes.")
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample actions instead of using deterministic policy mode.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Root output directory for belief robustness results.",
    )
    return parser.parse_args()


def main():
    run_belief_test(parse_args())


if __name__ == "__main__":
    main()

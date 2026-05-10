#!/usr/bin/env python3
"""Generate paper-ready belief-vs-observation analysis tables and figures.

This script is intentionally separate from the existing evaluation summaries.
It reads already-generated robustness and belief-ablation JSON files, then
prints a compact report intended for the final paper.

Outputs:
    evaluation/paper_belief_analysis/
        downstream_fixed.csv
        switching_stability.csv
        belief_ablation.csv
        downstream_fixed.png
        switching_stability.png
        belief_ablation.png

Example:
    python evaluation/paper_belief_analysis.py
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean


DEFAULT_ROBUSTNESS_GLOB = "evaluation/robustness_results/*/robustness_results.json"
DEFAULT_BELIEF_GLOB = "evaluation/belief_results/*/belief_robustness_results.json"
DEFAULT_OUTPUT_DIR = "evaluation/paper_belief_analysis"
DEFAULT_OBS_POLICY = "aggressiveExpert_neutral"
DEFAULT_BELIEF_POLICY = "aggressiveExpert_neutral_belief"


def _mean(values):
    values = [float(value) for value in values if value is not None]
    return mean(values) if values else 0.0


def _policy_name(checkpoint_path):
    run_name = Path(checkpoint_path).parents[1].name
    return run_name.rsplit("_2026", 1)[0]


def _input_type(policy):
    return "obs+belief" if "belief" in policy.lower() else "obs"


def _read_json(path):
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def _load_robustness(path):
    data = _read_json(path)
    policy = _policy_name(data["checkpoint"])
    eval_config = data.get("eval_config")
    summaries = data.get("summaries", [])
    rewards = [item.get("mean_reward") for item in summaries]
    return {
        "policy": policy,
        "input": _input_type(policy),
        "eval_type": "switching" if eval_config else "fixed",
        "eval_config": eval_config or "fixed_opponents",
        "avg_reward": round(_mean(rewards), 4),
        "worst_reward": round(min(rewards), 4) if rewards else 0.0,
        "reward_gap": round(max(rewards) - min(rewards), 4) if rewards else 0.0,
        "avg_win_rate": round(_mean(item.get("win_rate") for item in summaries), 4),
        "avg_deaths": round(_mean(item.get("mean_deaths") for item in summaries), 4),
        "avg_agent_takedowns": round(
            _mean(item.get("mean_enemy_agent_takedowns") for item in summaries),
            4,
        ),
        "avg_minion_takedowns": round(
            _mean(item.get("mean_enemy_minion_takedowns") for item in summaries),
            4,
        ),
        "switch_reward_delta": round(
            _mean(item.get("mean_switch_reward_delta") for item in summaries),
            4,
        ),
        "switch_reward_std_delta": round(
            _mean(item.get("mean_switch_reward_std_delta") for item in summaries),
            4,
        ),
        "switch_action_shift": round(
            _mean(item.get("mean_switch_action_shift") for item in summaries),
            4,
        ),
        "pre_switch_attack_rate": round(
            _mean(item.get("mean_pre_switch_attack_rate") for item in summaries),
            4,
        ),
        "post_switch_attack_rate": round(
            _mean(item.get("mean_post_switch_attack_rate") for item in summaries),
            4,
        ),
        "path": str(path),
    }


def _load_belief_ablation(path):
    data = _read_json(path)
    policy = _policy_name(data["checkpoint"])
    summaries = data.get("summaries", [])
    return {
        "policy": policy,
        "belief_mode": data.get("belief_mode", "unknown"),
        "eval_config": data.get("eval_config") or "fixed_opponents",
        "avg_reward": round(_mean(item.get("mean_reward") for item in summaries), 4),
        "avg_win_rate": round(_mean(item.get("win_rate") for item in summaries), 4),
        "avg_deaths": round(_mean(item.get("mean_deaths") for item in summaries), 4),
        "avg_agent_takedowns": round(
            _mean(item.get("mean_enemy_agent_takedowns") for item in summaries),
            4,
        ),
        "avg_minion_takedowns": round(
            _mean(item.get("mean_enemy_minion_takedowns") for item in summaries),
            4,
        ),
        "belief_accuracy": round(_mean(item.get("belief_accuracy") for item in summaries), 4),
        "post_switch_accuracy": round(
            _mean(item.get("post_switch_accuracy") for item in summaries),
            4,
        ),
        "switch_detection_delay": round(
            _mean(
                item.get("mean_switch_detection_delay")
                for item in summaries
                if item.get("mean_switch_detection_delay") is not None
            ),
            4,
        ),
        "belief_entropy": round(
            _mean(item.get("mean_belief_entropy") for item in summaries),
            4,
        ),
        "path": str(path),
    }


def _discover(pattern):
    return sorted(Path.cwd().glob(pattern))


def _write_csv(path, rows, headers):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _markdown_table(rows, headers):
    if not rows:
        return "_No data._\n"
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")
    return "\n".join(lines) + "\n"


def _paired_delta(obs_row, belief_row):
    if not obs_row or not belief_row:
        return None
    return {
        "comparison": "belief - observation",
        "reward_delta": round(belief_row["avg_reward"] - obs_row["avg_reward"], 4),
        "death_delta": round(belief_row["avg_deaths"] - obs_row["avg_deaths"], 4),
        "agent_takedown_delta": round(
            belief_row["avg_agent_takedowns"] - obs_row["avg_agent_takedowns"],
            4,
        ),
        "minion_takedown_delta": round(
            belief_row["avg_minion_takedowns"] - obs_row["avg_minion_takedowns"],
            4,
        ),
        "switch_reward_delta_delta": round(
            belief_row["switch_reward_delta"] - obs_row["switch_reward_delta"],
            4,
        ),
        "switch_std_delta_delta": round(
            belief_row["switch_reward_std_delta"] - obs_row["switch_reward_std_delta"],
            4,
        ),
        "action_shift_delta": round(
            belief_row["switch_action_shift"] - obs_row["switch_action_shift"],
            4,
        ),
    }


def _plot_or_warn(output_dir, fixed_rows, switching_rows, ablation_rows):
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        return [f"Matplotlib unavailable; figures were not generated: {exc}"]

    warnings = []
    if fixed_rows:
        labels = [row["input"] for row in fixed_rows]
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
        metrics = [
            ("avg_reward", "Average reward"),
            ("avg_deaths", "Deaths"),
            ("avg_agent_takedowns", "Enemy agent takedowns"),
        ]
        colors = ["#4C78A8" if row["input"] == "obs" else "#F58518" for row in fixed_rows]
        for ax, (metric, title) in zip(axes, metrics):
            ax.bar(labels, [row[metric] for row in fixed_rows], color=colors)
            ax.set_title(title)
            ax.grid(axis="y", alpha=0.25)
        fig.suptitle("Fixed-opponent downstream performance")
        fig.tight_layout()
        fig.savefig(output_dir / "downstream_fixed.png", dpi=180)
        plt.close(fig)

    if switching_rows:
        labels = [row["input"] for row in switching_rows]
        fig, axes = plt.subplots(1, 4, figsize=(14, 3.4))
        metrics = [
            ("avg_reward", "Average reward"),
            ("switch_reward_delta", "Post-pre reward"),
            ("switch_reward_std_delta", "Reward volatility change"),
            ("switch_action_shift", "Action distribution shift"),
        ]
        colors = ["#4C78A8" if row["input"] == "obs" else "#F58518" for row in switching_rows]
        for ax, (metric, title) in zip(axes, metrics):
            ax.bar(labels, [row[metric] for row in switching_rows], color=colors)
            ax.axhline(0, color="black", linewidth=0.8)
            ax.set_title(title)
            ax.grid(axis="y", alpha=0.25)
        fig.suptitle("Strategy-switching performance and stability")
        fig.tight_layout()
        fig.savefig(output_dir / "switching_stability.png", dpi=180)
        plt.close(fig)

    if ablation_rows:
        order = ["correct", "random", "shuffled", "lagged"]
        rows = sorted(
            ablation_rows,
            key=lambda row: order.index(row["belief_mode"]) if row["belief_mode"] in order else 99,
        )
        labels = [row["belief_mode"] for row in rows]
        fig, axes = plt.subplots(1, 4, figsize=(14, 3.4))
        metrics = [
            ("avg_reward", "Average reward"),
            ("avg_deaths", "Deaths"),
            ("belief_accuracy", "Belief accuracy"),
            ("switch_detection_delay", "Detection delay"),
        ]
        for ax, (metric, title) in zip(axes, metrics):
            ax.bar(labels, [row[metric] for row in rows], color="#54A24B")
            ax.set_title(title)
            ax.tick_params(axis="x", rotation=30)
            ax.grid(axis="y", alpha=0.25)
        fig.suptitle("Belief input ablation")
        fig.tight_layout()
        fig.savefig(output_dir / "belief_ablation.png", dpi=180)
        plt.close(fig)

    return warnings


def _analysis_text(obs_fixed, belief_fixed, obs_switch, belief_switch, ablation_rows):
    lines = []
    fixed_delta = _paired_delta(obs_fixed, belief_fixed)
    switch_delta = _paired_delta(obs_switch, belief_switch)

    if fixed_delta:
        lines.append(
            f"- On fixed opponents, the belief-conditioned policy changed average reward by "
            f"`{fixed_delta['reward_delta']}` relative to the observation-only baseline. "
            f"Deaths changed by `{fixed_delta['death_delta']}`."
        )
    if switch_delta:
        lines.append(
            f"- Under strategy switching, the belief-conditioned policy changed average reward by "
            f"`{switch_delta['reward_delta']}`. The post-switch reward delta changed by "
            f"`{switch_delta['switch_reward_delta_delta']}`, reward volatility changed by "
            f"`{switch_delta['switch_std_delta_delta']}`, and action-shift changed by "
            f"`{switch_delta['action_shift_delta']}`."
        )

    if ablation_rows:
        correct = next((row for row in ablation_rows if row["belief_mode"] == "correct"), None)
        random = next((row for row in ablation_rows if row["belief_mode"] == "random"), None)
        shuffled = next((row for row in ablation_rows if row["belief_mode"] == "shuffled"), None)
        lagged = next((row for row in ablation_rows if row["belief_mode"] == "lagged"), None)
        if correct and random:
            lines.append(
                f"- Correct belief reward vs random belief reward: `{correct['avg_reward']}` "
                f"vs `{random['avg_reward']}`. This tests whether the policy uses belief "
                f"semantics rather than only the extra input dimensions."
            )
        if correct and shuffled:
            lines.append(
                f"- Correct belief reward vs shuffled belief reward: `{correct['avg_reward']}` "
                f"vs `{shuffled['avg_reward']}`. A shuffled drop would indicate that the "
                f"meaning of each belief dimension matters."
            )
        if correct and lagged:
            lines.append(
                f"- Correct belief reward vs lagged belief reward: `{correct['avg_reward']}` "
                f"vs `{lagged['avg_reward']}`. This tests whether current strategy inference "
                f"is more useful than stale inference."
            )

    lines.append(
        "- Interpretation note: belief can still be scientifically useful even when it does "
        "not improve final reward, because the ablations test whether the policy actually "
        "uses the inferred opponent-state signal."
    )
    return "\n".join(lines)


def generate(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    robustness_rows = [_load_robustness(path) for path in _discover(args.robustness_glob)]
    robustness_rows = [
        row for row in robustness_rows
        if row["policy"] in {args.obs_policy, args.belief_policy}
    ]
    belief_rows = [_load_belief_ablation(path) for path in _discover(args.belief_glob)]
    belief_rows = [row for row in belief_rows if row["policy"] == args.belief_policy]

    obs_fixed = next(
        (row for row in robustness_rows if row["policy"] == args.obs_policy and row["eval_type"] == "fixed"),
        None,
    )
    belief_fixed = next(
        (row for row in robustness_rows if row["policy"] == args.belief_policy and row["eval_type"] == "fixed"),
        None,
    )
    obs_switch = next(
        (row for row in robustness_rows if row["policy"] == args.obs_policy and row["eval_type"] == "switching"),
        None,
    )
    belief_switch = next(
        (row for row in robustness_rows if row["policy"] == args.belief_policy and row["eval_type"] == "switching"),
        None,
    )

    fixed_rows = [row for row in [obs_fixed, belief_fixed] if row]
    switching_rows = [row for row in [obs_switch, belief_switch] if row]
    delta_rows = [
        row for row in [
            _paired_delta(obs_fixed, belief_fixed),
            _paired_delta(obs_switch, belief_switch),
        ]
        if row
    ]

    fixed_headers = [
        "policy",
        "input",
        "avg_reward",
        "worst_reward",
        "reward_gap",
        "avg_win_rate",
        "avg_deaths",
        "avg_agent_takedowns",
        "avg_minion_takedowns",
    ]
    switch_headers = [
        "policy",
        "input",
        "eval_config",
        "avg_reward",
        "avg_win_rate",
        "avg_deaths",
        "avg_agent_takedowns",
        "avg_minion_takedowns",
        "switch_reward_delta",
        "switch_reward_std_delta",
        "switch_action_shift",
    ]
    delta_headers = [
        "comparison",
        "reward_delta",
        "death_delta",
        "agent_takedown_delta",
        "minion_takedown_delta",
        "switch_reward_delta_delta",
        "switch_std_delta_delta",
        "action_shift_delta",
    ]
    ablation_headers = [
        "policy",
        "belief_mode",
        "eval_config",
        "avg_reward",
        "avg_win_rate",
        "avg_deaths",
        "avg_agent_takedowns",
        "avg_minion_takedowns",
        "belief_accuracy",
        "post_switch_accuracy",
        "switch_detection_delay",
        "belief_entropy",
    ]

    _write_csv(output_dir / "downstream_fixed.csv", fixed_rows, fixed_headers)
    _write_csv(output_dir / "switching_stability.csv", switching_rows, switch_headers)
    _write_csv(output_dir / "belief_ablation.csv", belief_rows, ablation_headers)
    _write_csv(output_dir / "belief_vs_obs_deltas.csv", delta_rows, delta_headers)
    warnings = _plot_or_warn(output_dir, fixed_rows, switching_rows, belief_rows)

    md = (
        "# Paper Belief Analysis\n\n"
        "This report is generated from existing evaluation JSON files and is intended "
        "to provide paper-ready tables, figures, and interpretation notes.\n\n"
        "## Table 1: Downstream Performance on Fixed Opponents\n\n"
        + _markdown_table(fixed_rows, fixed_headers)
        + "\n![Fixed downstream performance](downstream_fixed.png)\n\n"
        "## Table 2: Strategy-Switching Stability\n\n"
        + _markdown_table(switching_rows, switch_headers)
        + "\n![Switching stability](switching_stability.png)\n\n"
        "## Table 3: Belief Minus Observation Deltas\n\n"
        + _markdown_table(delta_rows, delta_headers)
        + "\n## Table 4: Perturbed-Belief Ablation\n\n"
        + _markdown_table(belief_rows, ablation_headers)
        + "\n![Belief ablation](belief_ablation.png)\n\n"
        "## Analysis Summary\n\n"
        + _analysis_text(obs_fixed, belief_fixed, obs_switch, belief_switch, belief_rows)
        + "\n"
    )
    if warnings:
        md += "\n## Generation Warnings\n\n" + "\n".join(f"- {warning}" for warning in warnings) + "\n"

    print()
    print(md)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate paper-ready belief analysis.")
    parser.add_argument("--robustness-glob", default=DEFAULT_ROBUSTNESS_GLOB)
    parser.add_argument("--belief-glob", default=DEFAULT_BELIEF_GLOB)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--obs-policy", default=DEFAULT_OBS_POLICY)
    parser.add_argument("--belief-policy", default=DEFAULT_BELIEF_POLICY)
    return parser.parse_args()


def main():
    generate(parse_args())


if __name__ == "__main__":
    main()

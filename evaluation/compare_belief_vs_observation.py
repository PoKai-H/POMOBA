#!/usr/bin/env python3
"""Compare observation-only and belief-conditioned policy evaluations.

This script focuses on two questions:

1. Downstream performance:
   Does the belief-conditioned policy achieve better reward, deaths, and
   takedowns than the observation-only baseline?

2. Switch ability:
   Under strategy-switching evaluation, does the belief-conditioned policy show
   lower harmful reward/action instability around the switch?

It reads existing robustness outputs and optionally belief-ablation outputs.

Example:
    python evaluation/compare_belief_vs_observation.py

    python evaluation/compare_belief_vs_observation.py \
      --include aggressiveExpert_neutral aggressiveExpert_neutral_belief
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean


DEFAULT_ROBUSTNESS_GLOB = "evaluation/robustness_results/*/robustness_results.json"
DEFAULT_BELIEF_GLOB = "evaluation/belief_results/*/belief_robustness_results.json"

def _mean(values):
    values = [float(value) for value in values if value is not None]
    return mean(values) if values else 0.0


def _config_name_from_checkpoint(checkpoint_path):
    run_name = Path(checkpoint_path).parents[1].name
    return run_name.rsplit("_2026", 1)[0]


def _input_type(policy_name):
    return "obs+belief" if "belief" in policy_name.lower() else "obs"


def _eval_type(eval_config):
    return "switching" if eval_config else "fixed"


def _load_robustness_row(path):
    with Path(path).open("r", encoding="utf-8") as file:
        data = json.load(file)

    summaries = data.get("summaries", [])
    rewards = [summary.get("mean_reward") for summary in summaries]
    eval_config = data.get("eval_config")
    policy = _config_name_from_checkpoint(data["checkpoint"])

    return {
        "policy": policy,
        "input": _input_type(policy),
        "eval_type": _eval_type(eval_config),
        "eval_config": eval_config or "fixed_opponents",
        "avg_reward": round(_mean(rewards), 4),
        "worst_reward": round(min(rewards), 4) if rewards else 0.0,
        "reward_gap": round(max(rewards) - min(rewards), 4) if rewards else 0.0,
        "avg_win_rate": round(_mean(summary.get("win_rate") for summary in summaries), 4),
        "avg_deaths": round(_mean(summary.get("mean_deaths") for summary in summaries), 4),
        "avg_enemy_agent_takedowns": round(
            _mean(summary.get("mean_enemy_agent_takedowns") for summary in summaries),
            4,
        ),
        "avg_enemy_minion_takedowns": round(
            _mean(summary.get("mean_enemy_minion_takedowns") for summary in summaries),
            4,
        ),
        "avg_switch_reward_delta": round(
            _mean(summary.get("mean_switch_reward_delta") for summary in summaries),
            4,
        ),
        "avg_switch_reward_std_delta": round(
            _mean(summary.get("mean_switch_reward_std_delta") for summary in summaries),
            4,
        ),
        "avg_switch_action_shift": round(
            _mean(summary.get("mean_switch_action_shift") for summary in summaries),
            4,
        ),
        "result_path": str(path),
    }


def _load_belief_ablation_row(path):
    with Path(path).open("r", encoding="utf-8") as file:
        data = json.load(file)

    summaries = data.get("summaries", [])
    policy = _config_name_from_checkpoint(data["checkpoint"])
    belief_mode = data.get("belief_mode", "unknown")
    eval_config = data.get("eval_config") or "fixed_opponents"

    return {
        "policy": policy,
        "belief_mode": belief_mode,
        "eval_config": eval_config,
        "avg_reward": round(_mean(summary.get("mean_reward") for summary in summaries), 4),
        "avg_win_rate": round(_mean(summary.get("win_rate") for summary in summaries), 4),
        "avg_deaths": round(_mean(summary.get("mean_deaths") for summary in summaries), 4),
        "avg_enemy_agent_takedowns": round(
            _mean(summary.get("mean_enemy_agent_takedowns") for summary in summaries),
            4,
        ),
        "avg_enemy_minion_takedowns": round(
            _mean(summary.get("mean_enemy_minion_takedowns") for summary in summaries),
            4,
        ),
        "belief_accuracy": round(_mean(summary.get("belief_accuracy") for summary in summaries), 4),
        "post_switch_accuracy": round(
            _mean(summary.get("post_switch_accuracy") for summary in summaries),
            4,
        ),
        "mean_switch_detection_delay": round(
            _mean(
                summary.get("mean_switch_detection_delay")
                for summary in summaries
                if summary.get("mean_switch_detection_delay") is not None
            ),
            4,
        ),
        "mean_belief_entropy": round(
            _mean(summary.get("mean_belief_entropy") for summary in summaries),
            4,
        ),
        "result_path": str(path),
    }


def _discover(paths, pattern):
    if paths:
        return [Path(path) for path in paths]
    return sorted(Path.cwd().glob(pattern))


def _filter_rows(rows, include):
    if not include:
        return rows
    include_set = set(include)
    return [row for row in rows if row.get("policy") in include_set]


def _markdown_table(rows, headers):
    if not rows:
        return "_No results found._\n"
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")
    return "\n".join(lines) + "\n"


def _delta_rows(rows):
    """Create obs+belief minus obs deltas for matching eval types/configs."""
    deltas = []
    obs_rows = [row for row in rows if row["input"] == "obs"]
    belief_rows = [row for row in rows if row["input"] == "obs+belief"]
    for belief in belief_rows:
        candidates = [
            row for row in obs_rows
            if row["eval_type"] == belief["eval_type"]
            and row["eval_config"] == belief["eval_config"]
        ]
        if not candidates:
            continue

        # Prefer the same curriculum name with the belief suffix removed.
        expected_obs_name = belief["policy"].replace("_belief", "")
        baseline = next(
            (row for row in candidates if row["policy"] == expected_obs_name),
            candidates[0],
        )
        deltas.append(
            {
                "belief_policy": belief["policy"],
                "obs_baseline": baseline["policy"],
                "eval_type": belief["eval_type"],
                "eval_config": belief["eval_config"],
                "reward_delta": round(belief["avg_reward"] - baseline["avg_reward"], 4),
                "death_delta": round(belief["avg_deaths"] - baseline["avg_deaths"], 4),
                "agent_takedown_delta": round(
                    belief["avg_enemy_agent_takedowns"] - baseline["avg_enemy_agent_takedowns"],
                    4,
                ),
                "minion_takedown_delta": round(
                    belief["avg_enemy_minion_takedowns"] - baseline["avg_enemy_minion_takedowns"],
                    4,
                ),
                "switch_reward_delta_delta": round(
                    belief["avg_switch_reward_delta"] - baseline["avg_switch_reward_delta"],
                    4,
                ),
                "switch_std_delta_delta": round(
                    belief["avg_switch_reward_std_delta"]
                    - baseline["avg_switch_reward_std_delta"],
                    4,
                ),
                "action_shift_delta": round(
                    belief["avg_switch_action_shift"] - baseline["avg_switch_action_shift"],
                    4,
                ),
            }
        )
    return deltas


def _analysis_text(delta_rows):
    if not delta_rows:
        return (
            "No matching observation-only / belief-conditioned pairs were found. "
            "Use `--include` or make sure both checkpoints have fixed and switching "
            "robustness results.\n"
        )

    lines = []
    for row in delta_rows:
        if row["eval_type"] == "fixed":
            lines.append(
                f"- Fixed eval: `{row['belief_policy']}` vs `{row['obs_baseline']}` "
                f"reward delta = `{row['reward_delta']}`, death delta = `{row['death_delta']}`."
            )
        else:
            lines.append(
                f"- Switching eval: `{row['belief_policy']}` vs `{row['obs_baseline']}` "
                f"reward delta = `{row['reward_delta']}`, "
                f"switch reward-delta improvement = `{row['switch_reward_delta_delta']}`, "
                f"switch std-delta change = `{row['switch_std_delta_delta']}`, "
                f"action-shift change = `{row['action_shift_delta']}`."
            )
    lines.append("")
    lines.append(
        "Interpret switch metrics jointly: lower action shift is not always better. "
        "A large action shift with stable reward can indicate adaptation, while a large "
        "action shift with worse reward indicates harmful instability."
    )
    return "\n".join(lines) + "\n"


def build_report(robustness_rows, belief_rows):
    fixed_rows = [row for row in robustness_rows if row["eval_type"] == "fixed"]
    switching_rows = [row for row in robustness_rows if row["eval_type"] == "switching"]
    deltas = _delta_rows(robustness_rows)

    fixed_headers = [
        "policy",
        "input",
        "avg_reward",
        "worst_reward",
        "reward_gap",
        "avg_win_rate",
        "avg_deaths",
        "avg_enemy_agent_takedowns",
        "avg_enemy_minion_takedowns",
    ]
    switching_headers = [
        "policy",
        "input",
        "eval_config",
        "avg_reward",
        "avg_win_rate",
        "avg_deaths",
        "avg_enemy_agent_takedowns",
        "avg_enemy_minion_takedowns",
        "avg_switch_reward_delta",
        "avg_switch_reward_std_delta",
        "avg_switch_action_shift",
    ]
    delta_headers = [
        "belief_policy",
        "obs_baseline",
        "eval_type",
        "eval_config",
        "reward_delta",
        "death_delta",
        "agent_takedown_delta",
        "minion_takedown_delta",
        "switch_reward_delta_delta",
        "switch_std_delta_delta",
        "action_shift_delta",
    ]
    belief_headers = [
        "policy",
        "belief_mode",
        "eval_config",
        "avg_reward",
        "avg_win_rate",
        "avg_deaths",
        "belief_accuracy",
        "post_switch_accuracy",
        "mean_switch_detection_delay",
        "mean_belief_entropy",
    ]

    content = (
        "# Belief vs Observation Evaluation\n\n"
        "This report compares downstream performance and strategy-switching stability "
        "between observation-only and belief-conditioned policies.\n\n"
        "## Downstream Performance: Fixed Opponents\n\n"
        + _markdown_table(fixed_rows, fixed_headers)
        + "\n## Switch Ability: Strategy-Switching Opponents\n\n"
        + _markdown_table(switching_rows, switching_headers)
        + "\n## Belief Minus Observation Deltas\n\n"
        "`reward_delta > 0` means the belief-conditioned policy has higher average reward. "
        "`death_delta < 0` means fewer deaths. For switch metrics, positive "
        "`switch_reward_delta_delta` means the post-switch reward drop improved.\n\n"
        + _markdown_table(deltas, delta_headers)
        + "\n## Belief Ablation Results\n\n"
        "These rows come from `robustness_belief_test.py` and compare correct, random, "
        "shuffled, and lagged belief inputs for the same belief-conditioned checkpoint.\n\n"
        + _markdown_table(belief_rows, belief_headers)
        + "\n## Interpretation Notes\n\n"
        + _analysis_text(deltas)
    )

    return content


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare observation-only and belief-conditioned evaluation results.",
    )
    parser.add_argument(
        "--robustness",
        nargs="*",
        default=None,
        help="Optional robustness_results.json paths. Defaults to all robustness results.",
    )
    parser.add_argument(
        "--belief-results",
        nargs="*",
        default=None,
        help="Optional belief_robustness_results.json paths. Defaults to all belief results.",
    )
    parser.add_argument(
        "--include",
        nargs="*",
        default=None,
        help="Optional policy names to include, e.g. aggressiveExpert_neutral aggressiveExpert_neutral_belief.",
    )
    parser.add_argument("--save-md", default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def main():
    args = parse_args()
    robustness_paths = _discover(args.robustness, DEFAULT_ROBUSTNESS_GLOB)
    belief_paths = _discover(args.belief_results, DEFAULT_BELIEF_GLOB)

    robustness_rows = [_load_robustness_row(path) for path in robustness_paths]
    robustness_rows = _filter_rows(robustness_rows, args.include)
    robustness_rows.sort(
        key=lambda row: (row["eval_type"], row["input"], -row["avg_reward"], row["policy"]),
    )

    belief_rows = [_load_belief_ablation_row(path) for path in belief_paths]
    belief_rows = _filter_rows(belief_rows, args.include)
    belief_rows.sort(key=lambda row: (row["policy"], row["eval_config"], row["belief_mode"]))

    print(build_report(robustness_rows, belief_rows))
    if args.save_md:
        print("Markdown files are no longer written; reports are printed to stdout.")


if __name__ == "__main__":
    main()

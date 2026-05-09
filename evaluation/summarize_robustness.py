#!/usr/bin/env python3
"""Summarize robustness test outputs into separate comparison tables.

Example:
    python evaluation/summarize_robustness.py
    python evaluation/summarize_robustness.py \
        --save-fixed-md evaluation/robustness_fixed_summary.md \
        --save-switching-md evaluation/robustness_switching_summary.md
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean


DEFAULT_RESULTS_GLOB = "evaluation/robustness_results/*/robustness_results.json"
DEFAULT_FIXED_MD_PATH = "evaluation/robustness_fixed_summary.md"
DEFAULT_SWITCHING_MD_PATH = "evaluation/robustness_switching_summary.md"


def _mean(values):
    values = [float(value) for value in values if value is not None]
    return mean(values) if values else 0.0


def _config_name_from_checkpoint(checkpoint_path):
    run_name = Path(checkpoint_path).parents[1].name
    return run_name.rsplit("_2026", 1)[0]


def load_summary(path):
    with Path(path).open("r", encoding="utf-8") as file:
        data = json.load(file)

    summaries = data.get("summaries", [])
    rewards = [item["mean_reward"] for item in summaries]

    return {
        "policy_checkpoint": _config_name_from_checkpoint(data["checkpoint"]),
        "eval_config": data.get("eval_config") or "fixed_opponents",
        "avg_reward_across_opponents": round(_mean(rewards), 4),
        "worst_opponent_reward": round(min(rewards), 4) if rewards else 0.0,
        "robustness_gap": round(max(rewards) - min(rewards), 4) if rewards else 0.0,
        "avg_win_rate": round(_mean(item.get("win_rate") for item in summaries), 4),
        "avg_deaths": round(_mean(item.get("mean_deaths") for item in summaries), 4),
        "avg_enemy_agent_takedowns": round(
            _mean(item.get("mean_enemy_agent_takedowns") for item in summaries),
            4,
        ),
        "avg_enemy_minion_takedowns": round(
            _mean(item.get("mean_enemy_minion_takedowns") for item in summaries),
            4,
        ),
        "avg_switch_reward_delta": round(
            _mean(item.get("mean_switch_reward_delta") for item in summaries),
            4,
        ),
        "avg_switch_reward_std_delta": round(
            _mean(item.get("mean_switch_reward_std_delta") for item in summaries),
            4,
        ),
        "avg_switch_action_shift": round(
            _mean(item.get("mean_switch_action_shift") for item in summaries),
            4,
        ),
        "result_path": str(path),
    }


def discover_result_paths(paths):
    if paths:
        return [Path(path) for path in paths]
    return sorted(Path.cwd().glob(DEFAULT_RESULTS_GLOB))


FIXED_HEADERS = [
    "policy_checkpoint",
    "avg_reward_across_opponents",
    "worst_opponent_reward",
    "robustness_gap",
    "avg_win_rate",
    "avg_deaths",
    "avg_enemy_agent_takedowns",
    "avg_enemy_minion_takedowns",
]

SWITCHING_HEADERS = [
    "policy_checkpoint",
    "eval_config",
    "avg_reward_across_opponents",
    "avg_win_rate",
    "avg_deaths",
    "avg_enemy_agent_takedowns",
    "avg_enemy_minion_takedowns",
    "avg_switch_reward_delta",
    "avg_switch_reward_std_delta",
    "avg_switch_action_shift",
]


def _is_switching_row(row):
    return row.get("eval_config") != "fixed_opponents"


def print_table(title, rows, headers):
    if not rows:
        print(f"No {title} results found.")
        return

    widths = {
        header: max(len(header), *(len(str(row.get(header, ""))) for row in rows))
        for header in headers
    }
    print(f"\n{title}")
    print(" ".join(header.ljust(widths[header]) for header in headers))
    print(" ".join("-" * widths[header] for header in headers))
    for row in rows:
        print(" ".join(str(row.get(header, "")).ljust(widths[header]) for header in headers))


def _markdown_table(rows, headers):
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")
    return "\n".join(lines)


def save_fixed_markdown(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = (
        "# Fixed-Opponent Robustness Summary\n\n"
        "This table compares each trained checkpoint against fixed scripted opponents.\n\n"
        "- `avg_reward_across_opponents`: mean reward averaged over opponent strategies.\n"
        "- `worst_opponent_reward`: lowest mean reward among evaluated opponents.\n"
        "- `robustness_gap`: best opponent reward minus worst opponent reward. Lower means more consistent.\n"
        "- `avg_win_rate`: mean terminal-win rate across opponents. New results use terminal-step reward; older result files may have used total episode reward.\n"
        "- `avg_deaths`: mean deaths across opponents.\n"
        "- `avg_enemy_agent_takedowns`: mean enemy agent takedowns credited directly to the evaluated PPO agent.\n"
        "- `avg_enemy_minion_takedowns`: mean enemy minion takedowns credited directly to the evaluated PPO agent.\n\n"
        + _markdown_table(rows, FIXED_HEADERS)
        + "\n"
    )
    path.write_text(content, encoding="utf-8")
    print(f"Saved fixed-opponent markdown summary to: {path}")


def save_switching_markdown(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = (
        "# Strategy-Switching Robustness Summary\n\n"
        "This table compares each trained checkpoint when the opponent strategy changes during an episode.\n\n"
        "- `avg_reward_across_opponents`: mean reward across switching evaluation configs.\n"
        "- `avg_win_rate`: mean terminal-win rate.\n"
        "- `avg_deaths`: mean deaths.\n"
        "- `avg_enemy_agent_takedowns`: mean enemy agent takedowns credited directly to the evaluated PPO agent.\n"
        "- `avg_enemy_minion_takedowns`: mean enemy minion takedowns credited directly to the evaluated PPO agent.\n"
        "- `avg_switch_reward_delta`: post-switch mean reward minus pre-switch mean reward.\n"
        "- `avg_switch_reward_std_delta`: post-switch reward volatility minus pre-switch volatility.\n"
        "- `avg_switch_action_shift`: L1 distance between pre-switch and post-switch action distributions.\n\n"
        + _markdown_table(rows, SWITCHING_HEADERS)
        + "\n"
    )
    path.write_text(content, encoding="utf-8")
    print(f"Saved strategy-switching markdown summary to: {path}")


def save_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved CSV summary to: {path}")


def save_json(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Saved JSON summary to: {path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize robustness test results.")
    parser.add_argument(
        "results",
        nargs="*",
        help="Optional robustness_results.json paths. Defaults to all robustness result folders.",
    )
    parser.add_argument(
        "--save-fixed-md",
        default=DEFAULT_FIXED_MD_PATH,
        help="Markdown output path for fixed-opponent results.",
    )
    parser.add_argument(
        "--save-switching-md",
        default=DEFAULT_SWITCHING_MD_PATH,
        help="Markdown output path for strategy-switching results.",
    )
    parser.add_argument("--save-csv", default=None, help="Optional CSV output path.")
    parser.add_argument("--save-json", default=None, help="Optional JSON output path.")
    return parser.parse_args()


def main():
    args = parse_args()
    rows = [load_summary(path) for path in discover_result_paths(args.results)]
    fixed_rows = [row for row in rows if not _is_switching_row(row)]
    switching_rows = [row for row in rows if _is_switching_row(row)]
    fixed_rows.sort(key=lambda row: row["avg_reward_across_opponents"], reverse=True)
    switching_rows.sort(key=lambda row: row["avg_reward_across_opponents"], reverse=True)

    print_table("Fixed-Opponent Robustness", fixed_rows, FIXED_HEADERS)
    print_table("Strategy-Switching Robustness", switching_rows, SWITCHING_HEADERS)

    if args.save_fixed_md:
        save_fixed_markdown(args.save_fixed_md, fixed_rows)
    if args.save_switching_md:
        save_switching_markdown(args.save_switching_md, switching_rows)
    if args.save_csv:
        save_csv(args.save_csv, rows)
    if args.save_json:
        save_json(args.save_json, rows)


if __name__ == "__main__":
    main()

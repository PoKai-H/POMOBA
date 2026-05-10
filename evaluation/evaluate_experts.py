#!/usr/bin/env python3
"""Rank expert-curriculum runs from saved training histories.

Example:
    python evaluation/evaluate_experts.py
    python evaluation/evaluate_experts.py outputs/training_analysis/*/history.json
    python evaluation/evaluate_experts.py --tail-updates 20 --opponent neutral
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean


DEFAULT_HISTORY_GLOB = "outputs/training_analysis/*/history.json"
SCORE_EXPLANATION = """# Expert Evaluation Score

The score is a heuristic ranking metric for comparing which expert curriculum
helped PPO learn the strongest final policy.

Formula:

```text
score =
  final_reward
  + 0.5 * post_expert_reward
  - 5.0 * death_count
  + 2.0 * enemy_minion_takedowns
  + 10.0 * enemy_agent_takedowns
```

Term meanings:

- `final_reward`: mean episode reward over the final evaluation window.
- `post_expert_reward`: mean episode reward after the expert ratio has faded below the threshold.
- `death_count`: mean number of PPO agent deaths per update in the final window.
- `enemy_minion_takedowns`: mean number of enemy minion takedowns per update in the final window.
- `enemy_agent_takedowns`: mean number of enemy agent takedowns per update in the final window.

Interpretation:

- Higher score is better.
- The score rewards final performance, post-expert stability, lane progress, and combat success.
- The score penalizes deaths.
- Behavior-cloning diagnostics such as `bc_loss_delta` are reported separately and are not part of the score, because they measure imitation quality rather than final task performance.
"""


def _mean(values, default=0.0):
    values = [float(value) for value in values if value is not None]
    return mean(values) if values else default


def _first_update_at_or_above(history, key, threshold):
    for item in history:
        value = item.get(key)
        if value is not None and float(value) >= threshold:
            return item.get("update")
    return None


def _safe_float(item, key, default=0.0):
    value = item.get(key)
    return default if value is None else float(value)


def load_history(path):
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def load_config_name(history_path):
    run_dir = Path(history_path).parent
    run_config_path = run_dir / "run_config.json"
    if not run_config_path.exists():
        return run_dir.name

    with run_config_path.open("r", encoding="utf-8") as file:
        run_config = json.load(file)
    return run_config.get("CONFIG_NAME", run_dir.name)


def infer_expert_name(config_name):
    lower_name = config_name.lower()
    if lower_name.startswith("ppo_") or lower_name.startswith("noexpert"):
        return "none"
    if "neutralexpert" in lower_name:
        return "neutral"
    if "farmingexpert" in lower_name:
        return "farming"
    if "aggressiveexpert" in lower_name or "aggresiveexpert" in lower_name:
        return "aggressive"
    return config_name.split("_", 1)[0]


def infer_opponent_name(config_name, history):
    if history:
        latest = history[-1].get("opponent_strategy")
        if latest:
            return latest

    parts = config_name.split("_")
    return parts[-1] if len(parts) > 1 else "unknown"


def _active_expert_window(history, fade_threshold):
    return [
        item
        for item in history
        if _safe_float(item, "configured_expert_ratio") > fade_threshold
    ]


def _window_delta(history, key_fn, window_size):
    if not history:
        return 0.0, 0.0, 0.0

    window_size = min(max(1, window_size), len(history))
    early = _mean(key_fn(item) for item in history[:window_size])
    late = _mean(key_fn(item) for item in history[-window_size:])
    return early, late, late - early


def summarize_run(history_path, tail_updates, fade_threshold, reward_threshold):
    history = load_history(history_path)
    config_name = load_config_name(history_path)
    expert = infer_expert_name(config_name)
    opponent = infer_opponent_name(config_name, history)

    tail = history[-tail_updates:] if tail_updates > 0 else history
    faded = [
        item
        for item in history
        if _safe_float(item, "configured_expert_ratio") <= fade_threshold
    ]
    post_expert_window = faded if faded else tail
    active_expert_window = _active_expert_window(history, fade_threshold)
    active_window_size = min(tail_updates, len(active_expert_window)) if tail_updates > 0 else len(active_expert_window)

    final_reward = _mean(item.get("episode_reward") for item in tail)
    post_expert_reward = _mean(item.get("episode_reward") for item in post_expert_window)
    death_count = _mean(item.get("death_count") for item in tail)
    minion_takedowns = _mean(item.get("takedown_enemy_minions") for item in tail)
    agent_takedowns = _mean(item.get("takedown_enemy_agents") for item in tail)
    entropy = _mean(item.get("metrics", {}).get("entropy") for item in tail)
    early_bc_loss, late_bc_loss, bc_loss_delta = _window_delta(
        active_expert_window,
        lambda item: item.get("metrics", {}).get("bc_loss"),
        active_window_size,
    )
    early_active_reward, late_active_reward, active_reward_delta = _window_delta(
        active_expert_window,
        lambda item: item.get("episode_reward"),
        active_window_size,
    )

    score = (
        final_reward
        + 0.5 * post_expert_reward
        - 5.0 * death_count
        + 2.0 * minion_takedowns
        + 10.0 * agent_takedowns
    )

    return {
        "run": Path(history_path).parent.name,
        "config": config_name,
        "expert": expert,
        "opponent": opponent,
        "updates": len(history),
        "final_reward": round(final_reward, 4),
        "post_expert_reward": round(post_expert_reward, 4),
        "post_expert_updates": len(post_expert_window),
        "death_count": round(death_count, 4),
        "enemy_minion_takedowns": round(minion_takedowns, 4),
        "enemy_agent_takedowns": round(agent_takedowns, 4),
        "early_bc_loss": round(early_bc_loss, 4),
        "late_bc_loss": round(late_bc_loss, 4),
        "bc_loss_delta": round(bc_loss_delta, 4),
        "early_active_reward": round(early_active_reward, 4),
        "late_active_reward": round(late_active_reward, 4),
        "active_reward_delta": round(active_reward_delta, 4),
        "entropy": round(entropy, 4),
        "first_reward_threshold_update": _first_update_at_or_above(
            history,
            "episode_reward",
            reward_threshold,
        ),
        "score": round(score, 4),
        "history_path": str(history_path),
    }


def discover_history_paths(paths):
    if paths:
        return [Path(path) for path in paths]
    return sorted(Path.cwd().glob(DEFAULT_HISTORY_GLOB))


def print_table(rows):
    if not rows:
        print("No histories found.")
        return

    headers = [
        "rank",
        "config",
        "expert",
        "opponent",
        "score",
        "final_reward",
        "post_expert_reward",
        "post_expert_updates",
        "death_count",
        "enemy_minion_takedowns",
        "enemy_agent_takedowns",
        "bc_loss_delta",
        "active_reward_delta",
        "entropy",
    ]
    widths = {
        header: max(len(header), *(len(str(row.get(header, ""))) for row in rows))
        for header in headers
    }

    print(" ".join(header.ljust(widths[header]) for header in headers))
    print(" ".join("-" * widths[header] for header in headers))
    for rank, row in enumerate(rows, start=1):
        row = {"rank": rank, **row}
        print(" ".join(str(row.get(header, "")).ljust(widths[header]) for header in headers))


def save_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _markdown_table(rows):
    headers = [
        "rank",
        "config",
        "expert",
        "opponent",
        "score",
        "final_reward",
        "post_expert_reward",
        "post_expert_updates",
        "death_count",
        "enemy_minion_takedowns",
        "enemy_agent_takedowns",
        "bc_loss_delta",
        "active_reward_delta",
        "entropy",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for rank, row in enumerate(rows, start=1):
        values = []
        row = {"rank": rank, **row}
        for header in headers:
            values.append(str(row.get(header, "")))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def markdown_report(rows):
    report = (
        SCORE_EXPLANATION.rstrip()
        + "\n\n"
        + "## Ranking Table\n\n"
        + _markdown_table(rows)
        + "\n"
    )
    return report


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate which expert helped PPO most.")
    parser.add_argument(
        "histories",
        nargs="*",
        help="Optional history.json paths. Defaults to outputs/training_analysis/*/history.json.",
    )
    parser.add_argument(
        "--tail-updates",
        type=int,
        default=20,
        help="Number of final updates used for final averages.",
    )
    parser.add_argument(
        "--fade-threshold",
        type=float,
        default=0.05,
        help="Expert ratio threshold treated as expert faded out.",
    )
    parser.add_argument(
        "--reward-threshold",
        type=float,
        default=0.0,
        help="Reward threshold used for learning-speed reporting.",
    )
    parser.add_argument(
        "--opponent",
        default=None,
        help="Optional opponent strategy filter, e.g. neutral.",
    )
    parser.add_argument("--save-json", default=None, help="Optional JSON output path.")
    parser.add_argument("--save-csv", default=None, help="Optional CSV output path.")
    parser.add_argument(
        "--save-md",
        default=None,
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def main():
    args = parse_args()
    rows = [
        summarize_run(path, args.tail_updates, args.fade_threshold, args.reward_threshold)
        for path in discover_history_paths(args.histories)
    ]

    if args.opponent is not None:
        rows = [row for row in rows if row["opponent"] == args.opponent]

    rows.sort(key=lambda row: row["score"], reverse=True)
    if args.save_json:
        with Path(args.save_json).open("w", encoding="utf-8") as file:
            json.dump(rows, file, indent=2)
    if args.save_csv:
        save_csv(args.save_csv, rows)
    print(markdown_report(rows))
    if args.save_md:
        print("Markdown files are no longer written; reports are printed to stdout.")


if __name__ == "__main__":
    main()

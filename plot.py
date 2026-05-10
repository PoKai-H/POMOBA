#!/usr/bin/env python3
"""Generate training analysis plots from an existing history.json file.

Examples:
    python plot.py --folder outputs/training_analysis/aggressiveExpert_neutral_20260507_174053
    python plot.py outputs/training_analysis/aggressiveExpert_neutral_20260507_174053
    python plot.py --aggressiveExpert_neutral_20260507_174053
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ACTION_NAMES = [
    "move_up",
    "move_left",
    "move_right",
    "move_down",
    "move_up_left",
    "move_up_right",
    "move_down_left",
    "move_down_right",
    "hold",
    "attack_hero",
    "attack_nearest_minion",
    "attack_tower",
    "retreat",
]

TRAINING_OUTPUT_ROOT = Path("outputs/training_analysis")


def _series(history, key, default=0.0):
    return [default if item.get(key) is None else item.get(key) for item in history]


def _metric_series(history, key, default=0.0):
    return [
        default if item.get("metrics", {}).get(key) is None else item["metrics"][key]
        for item in history
    ]


def _load_history(folder):
    history_path = folder / "history.json"
    if not history_path.exists():
        raise FileNotFoundError(f"history.json not found in: {folder}")
    with history_path.open("r", encoding="utf-8") as file:
        history = json.load(file)
    if not history:
        raise ValueError(f"history.json is empty: {history_path}")
    return history


def _resolve_folder(folder_arg):
    folder = Path(folder_arg).expanduser()
    candidates = []

    if folder.is_absolute() or folder.exists():
        candidates.append(folder)
    else:
        candidates.extend(
            [
                Path.cwd() / folder,
                TRAINING_OUTPUT_ROOT / folder_arg,
            ]
        )
        candidates.extend(sorted(TRAINING_OUTPUT_ROOT.glob(f"{folder_arg}*")))

    for candidate in candidates:
        if candidate.is_dir() and (candidate / "history.json").exists():
            return candidate

    searched = "\n".join(f"- {candidate}" for candidate in candidates)
    raise FileNotFoundError(
        f"Could not find a training output folder for `{folder_arg}`.\n"
        f"Searched:\n{searched}"
    )


def _folder_from_unknown_args(unknown):
    folder_flags = [arg for arg in unknown if arg.startswith("--")]
    if len(folder_flags) != 1:
        return None
    return folder_flags[0][2:]


def _plot_training_curves(plt, history, output_dir):
    updates = _series(history, "update")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes[0, 0].plot(updates, _series(history, "episode_reward", None), marker="o")
    axes[0, 0].set_title("Episode Reward")
    axes[0, 0].set_xlabel("Update")
    axes[0, 0].set_ylabel("Reward")

    axes[0, 1].plot(updates, _series(history, "episode_length", None), marker="o")
    axes[0, 1].set_title("Episode Length")
    axes[0, 1].set_xlabel("Update")
    axes[0, 1].set_ylabel("Steps")

    axes[1, 0].plot(updates, _series(history, "loss"), marker="o", label="total")
    axes[1, 0].plot(updates, _metric_series(history, "critic_loss"), marker="o", label="critic")
    axes[1, 0].plot(updates, _metric_series(history, "actor_loss"), marker="o", label="actor")
    axes[1, 0].set_title("PPO Loss")
    axes[1, 0].set_xlabel("Update")
    axes[1, 0].legend()

    axes[1, 1].plot(updates, _metric_series(history, "entropy"), marker="o")
    axes[1, 1].set_title("Policy Entropy")
    axes[1, 1].set_xlabel("Update")
    axes[1, 1].set_ylabel("Entropy")

    path = output_dir / "training_curves.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _plot_action_diagnostics(plt, history, output_dir):
    updates = _series(history, "update")

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), constrained_layout=True)
    axes[0].plot(updates, _series(history, "attack_rate"), marker="o", label="attack")
    axes[0].plot(updates, _series(history, "movement_rate"), marker="o", label="movement")
    axes[0].plot(updates, _series(history, "retreat_rate"), marker="o", label="retreat")
    axes[0].plot(updates, _series(history, "expert_action_rate"), marker="o", label="expert")
    axes[0].plot(
        updates,
        _series(history, "configured_expert_ratio"),
        marker="o",
        linestyle="--",
        label="expert target",
    )
    axes[0].set_title("Action Group Rates")
    axes[0].set_xlabel("Update")
    axes[0].set_ylabel("Rate")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].legend()

    action_rates = []
    for item in history:
        counts = item.get("action_distribution", {})
        action_rates.append([counts.get(name, 0.0) for name in ACTION_NAMES])
    image = axes[1].imshow(action_rates, aspect="auto", interpolation="nearest")
    axes[1].set_title("Most Used Actions by Update")
    axes[1].set_xlabel("Action")
    axes[1].set_ylabel("Update")
    axes[1].set_xticks(range(len(ACTION_NAMES)))
    axes[1].set_xticklabels(ACTION_NAMES, rotation=45, ha="right")
    axes[1].set_yticks(range(len(updates)))
    axes[1].set_yticklabels(updates)
    fig.colorbar(image, ax=axes[1], label="Rate")

    path = output_dir / "action_diagnostics.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _plot_reward_diagnostics(plt, history, output_dir):
    updates = _series(history, "update")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes[0, 0].plot(updates, _series(history, "reward_mean"), marker="o", label="mean")
    axes[0, 0].plot(updates, _series(history, "reward_sum"), marker="o", label="sum")
    axes[0, 0].set_title("Reward per Rollout")
    axes[0, 0].set_xlabel("Update")
    axes[0, 0].legend()

    axes[0, 1].plot(updates, _series(history, "positive_reward_steps"), marker="o", label="positive")
    axes[0, 1].plot(updates, _series(history, "negative_reward_steps"), marker="o", label="negative")
    axes[0, 1].set_title("Reward Signal Frequency")
    axes[0, 1].set_xlabel("Update")
    axes[0, 1].set_ylabel("Steps")
    axes[0, 1].legend()

    axes[1, 0].plot(updates, _series(history, "advantage_mean"), marker="o", label="mean")
    axes[1, 0].plot(updates, _series(history, "advantage_min"), marker="o", label="min")
    axes[1, 0].plot(updates, _series(history, "advantage_max"), marker="o", label="max")
    axes[1, 0].set_title("Advantage")
    axes[1, 0].set_xlabel("Update")
    axes[1, 0].legend()

    axes[1, 1].plot(updates, _series(history, "return_mean"), marker="o", label="return")
    axes[1, 1].plot(updates, _series(history, "value_mean"), marker="o", label="value")
    axes[1, 1].set_title("Return vs Value")
    axes[1, 1].set_xlabel("Update")
    axes[1, 1].legend()

    path = output_dir / "reward_diagnostics.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _plot_combat_events(plt, history, output_dir):
    updates = _series(history, "update")

    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    ax.plot(updates, _series(history, "takedown_enemy_agents"), marker="o", label="enemy agents")
    ax.plot(updates, _series(history, "takedown_enemy_minions"), marker="o", label="enemy minions")
    ax.plot(updates, _series(history, "death_count"), marker="o", label="deaths")
    ax.set_title("Combat Events per Update")
    ax.set_xlabel("Update")
    ax.set_ylabel("Count")
    ax.legend()

    path = output_dir / "combat_events.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def generate_plots(folder):
    history = _load_history(folder)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = [
        _plot_training_curves(plt, history, folder),
        _plot_action_diagnostics(plt, history, folder),
        _plot_reward_diagnostics(plt, history, folder),
        _plot_combat_events(plt, history, folder),
    ]
    return paths


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate training plots from outputs/training_analysis/<folder>/history.json.",
    )
    parser.add_argument("folder", nargs="?", help="Training output folder path or folder name.")
    parser.add_argument("--folder", dest="folder_option", help="Training output folder path or folder name.")
    args, unknown = parser.parse_known_args(argv)

    folder = args.folder_option or args.folder or _folder_from_unknown_args(unknown)
    if not folder:
        parser.error("provide a folder, e.g. `python plot.py --aggressiveExpert_neutral_20260507_174053`")
    if len(unknown) > 1 or (unknown and folder != _folder_from_unknown_args(unknown)):
        parser.error(f"unrecognized arguments: {' '.join(unknown)}")

    args.folder = folder
    return args


def main(argv=None):
    args = parse_args(argv)
    folder = _resolve_folder(args.folder)
    paths = generate_plots(folder)
    print(f"Generated plots in: {folder}")
    for path in paths:
        print(f"- {path}")


if __name__ == "__main__":
    main(sys.argv[1:])


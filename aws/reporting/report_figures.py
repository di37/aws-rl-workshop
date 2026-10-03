"""Renders the report figures from report-table rows (headless, PNG)."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

BEFORE_COLOR = "#9e9e9e"
AFTER_COLOR = "#1f77b4"
HEADLINE_METRICS = ("eval/reward/pass_at_1", "eval/reward/pass_at_2", "eval/reward/mean")
DPI = 150


def training_reward_curve(rows: Sequence[Mapping[str, Any]], path: Path) -> Path:
    """Plots the mean rollout reward of every training step.

    Args:
        rows: Rows from :func:`report_tables.training_curve`.
        path: Output PNG path.

    Returns:
        The written path.
    """
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot([r["step"] for r in rows], [_value(r["mean_reward"]) for r in rows], marker="o", color=AFTER_COLOR)
    for row in [row for row in rows[:1] + rows[-1:] if row["mean_reward"] is not None]:
        ax.annotate(f"{row['mean_reward']:.3f}", (row["step"], row["mean_reward"]),
                    textcoords="offset points", xytext=(0, -14), ha="center", fontsize=8)
    ax.set(title="Mean rollout reward per training step (32 tickets x 4 rollouts)",
           xlabel="Training step", ylabel="Mean reward (partial credit)", ylim=(0, 1.05))
    ax.set_xticks([r["step"] for r in rows])
    ax.grid(alpha=0.3)
    return _save(fig, path)


def evaluation_before_after(rows: Sequence[Mapping[str, Any]], path: Path) -> Path:
    """Compares base and fine-tuned headline metrics on the held-out tickets.

    Args:
        rows: Rows from :func:`report_tables.evaluation_comparison`.
        path: Output PNG path.

    Returns:
        The written path.
    """
    chosen = [row for row in rows if row["metric"] in HEADLINE_METRICS]
    positions = range(len(chosen))
    width = 0.38
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for offset, key, label, color in ((-width / 2, "base", "Base model", BEFORE_COLOR),
                                      (width / 2, "fine_tuned", "Fine-tuned model", AFTER_COLOR)):
        values = [_value(row[key]) for row in chosen]
        bars = ax.bar([p + offset for p in positions], values, width, label=label, color=color)
        ax.bar_label(bars, labels=["n/a" if row[key] is None else f"{row[key]:.2f}" for row in chosen],
                     padding=2, fontsize=8)
    ax.set_xticks(list(positions), [row["label"] for row in chosen])
    ax.set(title="Held-out tickets: before vs after training (32 tickets x 2 rollouts per model)",
           ylabel="Score", ylim=(0, 1.12))
    ax.legend(loc="lower right")
    return _save(fig, path)


def per_ticket_scatter(rows: Sequence[Mapping[str, Any]], path: Path) -> Path:
    """Plots each held-out ticket's mean reward before against after training.

    Points above the diagonal improved. Point size counts tickets that share
    the same pair of means.

    Args:
        rows: Rows from :func:`report_tables.per_ticket_rewards`.
        path: Output PNG path.

    Returns:
        The written path.
    """
    counts = Counter((row["before_mean"], row["after_mean"]) for row in rows
                     if row["before_mean"] is not None and row["after_mean"] is not None)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1.05], [0, 1.05], linestyle="--", color=BEFORE_COLOR, linewidth=1)
    for (before, after), count in counts.items():
        ax.scatter(before, after, s=60 * count, alpha=0.6, color=AFTER_COLOR)
        ax.annotate(str(count), (before, after), ha="center", va="center", fontsize=8)
    ax.set(title="Per held-out ticket: mean reward\n(above the line = better after training)",
           xlabel="Before training (base model)", ylabel="After training (fine-tuned)",
           xlim=(0, 1.08), ylim=(0, 1.08))
    ax.grid(alpha=0.3)
    return _save(fig, path)


def live_inference_outcomes(rows: Sequence[Mapping[str, Any]], path: Path) -> Path:
    """Compares the live before/after reward of each replayed ticket.

    Args:
        rows: Rows from :func:`report_tables.live_inference`.
        path: Output PNG path.

    Returns:
        The written path.
    """
    positions = range(len(rows))
    fig, ax = plt.subplots(figsize=(9, 0.7 * len(rows) + 2))
    for offset, key, label, color in ((-0.2, "before_reward", "Before training (base model)", BEFORE_COLOR),
                                      (0.2, "after_reward", "After training (fine-tuned)", AFTER_COLOR)):
        ax.barh([p + offset for p in positions], [_value(row[key]) for row in rows], 0.4,
                label=label, color=color)
    ax.set_yticks(list(positions), [_shorten(row["ticket"]) for row in rows])
    ax.invert_yaxis()
    ax.set(title="Live inference on held-out tickets (1.0 = fully solved)", xlabel="Reward", xlim=(0, 1.05))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=8)
    return _save(fig, path)


def _value(number: float | None) -> float:
    """Plots a missing or failed value as a gap (NaN), never as a zero."""
    return float("nan") if number is None else float(number)


def _shorten(text: str, limit: int = 60) -> str:
    """Shortens a ticket for an axis label."""
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _save(fig: Any, path: Path) -> Path:
    """Writes and closes a figure.

    Args:
        fig: Matplotlib figure.
        path: Output PNG path; its folder is created if missing.

    Returns:
        The written path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path

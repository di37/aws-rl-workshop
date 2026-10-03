"""Smoke tests: every report figure renders to a non-empty PNG."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from aws.reporting.report_figures import (  # noqa: E402
    evaluation_before_after,
    live_inference_outcomes,
    per_ticket_scatter,
    training_reward_curve,
)


class FigureTests(unittest.TestCase):
    """Verifies each figure writer produces an image file."""

    def test_each_figure_writes_a_png(self) -> None:
        comparison = [
            {"metric": "eval/reward/pass_at_1", "label": "pass@1", "base": 0.5, "fine_tuned": 0.75},
            {"metric": "eval/reward/pass_at_2", "label": "pass@2", "base": 0.8, "fine_tuned": 0.9},
            {"metric": "eval/reward/mean", "label": "mean", "base": 0.85, "fine_tuned": 0.95},
        ]
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            paths = [
                training_reward_curve([{"step": 1, "mean_reward": 0.8}, {"step": 2, "mean_reward": 0.9}],
                                      out / "curve.png"),
                evaluation_before_after(comparison, out / "bars.png"),
                per_ticket_scatter([{"before_mean": 0.8, "after_mean": 1.0},
                                    {"before_mean": None, "after_mean": 0.5}], out / "scatter.png"),
                live_inference_outcomes([{"ticket": "router blinking", "before_reward": 0.75,
                                          "after_reward": None}], out / "live.png"),
            ]
            sizes = [path.stat().st_size for path in paths]

        self.assertTrue(all(size > 1_000 for size in sizes), sizes)

    def test_missing_values_and_empty_rows_do_not_crash(self) -> None:
        comparison = [{"metric": "eval/reward/pass_at_1", "label": "pass@1", "base": None, "fine_tuned": 0.7}]
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            paths = [
                training_reward_curve([], out / "empty.png"),
                training_reward_curve([{"step": 1, "mean_reward": None}], out / "none.png"),
                evaluation_before_after(comparison, out / "bars.png"),
            ]

            self.assertTrue(all(path.stat().st_size > 0 for path in paths))


if __name__ == "__main__":
    unittest.main()

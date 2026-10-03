"""Unit tests for the hard-capped, measurement-based budget guard."""

from __future__ import annotations

import unittest
from decimal import Decimal

from aws.config import DemoConfig
from aws.costs.cost_guard import MtrlCostGuard

RATES = {
    "training_prefill": Decimal("0.12"),
    "training_sample": Decimal("0.30"),
    "training_train": Decimal("0.36"),
    "evaluation_prefill": Decimal("0.12"),
    "evaluation_sample": Decimal("0.30"),
}
MEASURED = {"prefill": 197_217, "sample": 34_617, "rollouts": 64}


class MtrlCostGuardTests(unittest.TestCase):
    """Verifies cost arithmetic and the hard ceiling."""

    def setUp(self) -> None:
        self.guard = MtrlCostGuard(DemoConfig(), RATES, Decimal("13.1158"))

    def test_billed_token_cost_uses_component_rates(self) -> None:
        cost = self.guard.billed_cost(
            "Evaluation", {"PrefillTokenCount": 1_000_000, "SampleTokenCount": 1_000_000}
        )

        self.assertEqual(cost, Decimal("0.42"))

    def test_training_cost_includes_train_tokens(self) -> None:
        cost = self.guard.billed_cost(
            "Finetuning",
            {
                "PrefillTokenCount": 1_000_000,
                "SampleTokenCount": 1_000_000,
                "TrainTokenCount": 1_000_000,
            },
        )

        self.assertEqual(cost, Decimal("0.78"))

    def test_planned_training_scales_measured_usage_to_all_rollouts(self) -> None:
        cost = self.guard.planned_training_cost(MEASURED)

        rollouts = Decimal(10 * 32 * 4)
        per_prefill = Decimal(197_217) / 64
        per_sample = Decimal(34_617) / 64
        expected = (
            rollouts * Decimal("1.5") / Decimal(1_000_000)
        ) * (per_prefill * RATES["training_prefill"] + per_sample * RATES["training_sample"]
             + (per_prefill + per_sample) * RATES["training_train"])
        self.assertAlmostEqual(float(cost), float(expected), places=6)

    def test_endpoint_cost_covers_maximum_lifetime(self) -> None:
        self.assertEqual(
            self.guard.endpoint_cost(minutes=75), Decimal("13.1158") * Decimal(75) / 60
        )

    def test_planned_endpoint_includes_deletion_tail(self) -> None:
        report = self.guard.evaluate(
            measured=MEASURED, spent=[],
            remaining={"training": False, "comparison": False, "endpoint": True},
        )

        line = next(item for item in report["lines"] if item["item"] == "Endpoint (planned)")
        minutes = 60 + MtrlCostGuard.ENDPOINT_TAIL_MINUTES
        self.assertAlmostEqual(line["usd"], float(Decimal("13.1158") * minutes / 60), places=3)

    def test_report_within_cap(self) -> None:
        report = self.guard.evaluate(
            measured=MEASURED,
            spent=[{"label": "base eval", "component": "Evaluation",
                    "usage": {"PrefillTokenCount": 197_217, "SampleTokenCount": 34_617}}],
            remaining={"training": True, "comparison": True, "endpoint": True},
        )

        self.assertTrue(report["within_budget"])
        self.assertLessEqual(report["projected_total_usd"], 25)
        self.assertEqual(report["budget_cap_usd"], 25)
        self.assertEqual(
            [line["item"] for line in report["lines"]][-1], "Contingency (AgentCore, CodeBuild, S3, logs)"
        )

    def test_completed_stages_are_not_planned_again(self) -> None:
        report = self.guard.evaluate(
            measured=MEASURED,
            spent=[],
            remaining={"training": False, "comparison": False, "endpoint": False},
        )

        items = [line["item"] for line in report["lines"]]
        self.assertNotIn("Training (planned)", items)
        self.assertNotIn("Endpoint (planned)", items)

    def test_report_over_cap_raises(self) -> None:
        guard = MtrlCostGuard(DemoConfig(), RATES, Decimal("40"))

        with self.assertRaisesRegex(PermissionError, r"\$25"):
            guard.evaluate(
                measured=MEASURED,
                spent=[],
                remaining={"training": True, "comparison": True, "endpoint": True},
            )

    def test_measured_usage_requires_rollouts(self) -> None:
        with self.assertRaises(ValueError):
            self.guard.planned_training_cost({"prefill": 1, "sample": 1, "rollouts": 0})


if __name__ == "__main__":
    unittest.main()

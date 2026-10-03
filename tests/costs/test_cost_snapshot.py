"""Unit tests for freezing the study's cost accounting into evidence."""

from __future__ import annotations

import unittest
from decimal import Decimal

from aws.config import DemoConfig
from aws.costs.cost_guard import MtrlCostGuard
from aws.costs.cost_snapshot import (
    SOURCE_ALLOWANCE,
    SOURCE_RECORDED,
    SOURCE_TOKENS,
    build_cost_snapshot,
)

RATES = {
    "training_prefill": Decimal("0.10"),
    "training_sample": Decimal("1.00"),
    "training_train": Decimal("0.50"),
    "evaluation_prefill": Decimal("0.20"),
    "evaluation_sample": Decimal("2.00"),
}


class CostSnapshotTests(unittest.TestCase):
    """Verifies priced lines keep their inputs and totals separate the allowance."""

    def setUp(self) -> None:
        self.guard = MtrlCostGuard(DemoConfig(), RATES, Decimal("13.00"))
        self.records = [
            {"label": "Training", "component": "Finetuning",
             "usage": {"PrefillTokenCount": 2_000_000, "SampleTokenCount": 1_000_000,
                       "TrainTokenCount": 2_000_000}},
            {"label": "Evaluation", "component": "Evaluation",
             "usage": {"PrefillTokenCount": 1_000_000, "SampleTokenCount": 500_000}},
            {"label": "Cross-region copy", "usd": 0.84},
        ]

    def test_token_lines_keep_usage_and_are_priced_from_rates(self) -> None:
        snapshot = build_cost_snapshot(self.records, self.guard, captured_at="2026-10-02T12:00:00+00:00")

        training, evaluation, copy, allowance = snapshot["items"]
        self.assertEqual(training["usd"], 2.2)
        self.assertEqual(training["usage"]["TrainTokenCount"], 2_000_000)
        self.assertEqual(training["source"], SOURCE_TOKENS)
        self.assertEqual(evaluation["usd"], 1.2)
        self.assertEqual((copy["usd"], copy["source"]), (0.84, SOURCE_RECORDED))
        self.assertEqual((allowance["usd"], allowance["source"]), (2.0, SOURCE_ALLOWANCE))

    def test_totals_separate_billed_tokens_estimates_and_the_allowance(self) -> None:
        snapshot = build_cost_snapshot(self.records, self.guard, captured_at="t")

        self.assertEqual(snapshot["billed_tokens_usd"], 3.4)
        self.assertEqual(snapshot["computed_or_estimated_usd"], 0.84)
        self.assertEqual(snapshot["total_usd"], 4.24)
        self.assertEqual(snapshot["total_with_allowance_usd"], 6.24)
        self.assertEqual(snapshot["hosting_usd_per_hour"], "13.00")
        self.assertEqual(snapshot["budget_cap_usd"], 25)
        self.assertEqual(snapshot["rates_usd_per_million_tokens"]["training_train"], "0.50")
        self.assertEqual(snapshot["captured_at"], "t")

    def test_missing_usage_is_priced_as_zero(self) -> None:
        records = [{"label": "Comparison: base model", "component": "Evaluation", "usage": {}}]

        snapshot = build_cost_snapshot(records, self.guard, captured_at="t")

        self.assertEqual(snapshot["items"][0]["usd"], 0.0)


if __name__ == "__main__":
    unittest.main()

"""Unit tests for the billed-spend ledger."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from aws.costs.ledger import SpendLedger
from aws.records.evidence import EvidenceStore

BASE_USAGE = {"PrefillTokenCount": 197_217, "SampleTokenCount": 34_617}


class SpendLedgerTests(unittest.TestCase):
    """Verifies spend collection from recorded jobs."""

    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.store = EvidenceStore(Path(self.directory.name))
        self.inspector = MagicMock()
        self.inspector.step_jobs.return_value = {"EvaluateBaseModel": "arn:job/base"}
        self.inspector.eval_job_details.return_value = {"billable_token_usage": BASE_USAGE}
        self.store.save("base_evaluation.json", {"arn": "arn:pipeline/v2"})
        self.store.save(
            "base_evaluation_metrics.json",
            {"eval/reward/num_prompts": 32, "eval/reward/rollouts_per_prompt": 2},
        )
        self.ledger = SpendLedger(self.store, self.inspector)

    def tearDown(self) -> None:
        self.directory.cleanup()

    def test_measured_reference_uses_base_evaluation_usage(self) -> None:
        self.assertEqual(
            self.ledger.measured_reference(),
            {"prefill": 197_217, "sample": 34_617, "rollouts": 64},
        )

    def test_spent_lists_only_recorded_stages(self) -> None:
        spent = self.ledger.spent()

        self.assertEqual([item["label"] for item in spent], ["Base evaluation"])
        self.assertEqual(spent[0]["component"], "Evaluation")

    def test_spent_includes_training_comparison_and_endpoint(self) -> None:
        self.store.save(
            "training_job.json",
            {"job_name": "j", "billable_token_usage": {"TrainTokenCount": 9}},
        )
        self.store.save(
            "comparison_evaluation_metrics.json",
            {"base": {"billable_token_usage": {"PrefillTokenCount": 1}},
             "fine_tuned": {"billable_token_usage": {"PrefillTokenCount": 2}}},
        )
        self.store.save("endpoint.json", {"status": "Deleted", "billed_usd": 4.5})

        labels = [item["label"] for item in self.ledger.spent()]

        self.assertEqual(
            labels,
            ["Base evaluation", "Training", "Comparison: base model",
             "Comparison: fine-tuned model", "Endpoint uptime"],
        )

    def test_spent_includes_earlier_attempts(self) -> None:
        self.store.save(
            "training_job.old-job.json",
            {"job_name": "old-job", "billable_token_usage": {"PrefillTokenCount": 5}},
        )
        self.store.save("endpoint.123.json", {"status": "Deleted", "billed_usd": 1.25})

        labels = [item["label"] for item in self.ledger.spent()]

        self.assertIn("Training (earlier attempt old-job)", labels)
        self.assertIn("Endpoint uptime (earlier attempt)", labels)

    def test_extra_costs_are_listed_as_spent(self) -> None:
        self.store.save("extra_costs.json", {"items": [
            {"label": "Cross-region copy to us-east-1", "usd": 0.84}]})

        spent = self.ledger.spent()

        self.assertIn({"label": "Cross-region copy to us-east-1", "usd": 0.84}, spent)

    def test_invalid_extra_costs_fail_loudly(self) -> None:
        for bad in (-1, "a lot"):
            self.store.save("extra_costs.json", {"items": [{"label": "x", "usd": bad}]})
            with self.assertRaises(ValueError):
                self.ledger.spent()

    def test_live_endpoint_still_counts_as_planned(self) -> None:
        self.store.save("endpoint.json", {"status": "InService"})

        self.assertTrue(self.ledger.remaining()["endpoint"])

    def test_remaining_reflects_recorded_stages(self) -> None:
        self.assertEqual(
            self.ledger.remaining(),
            {"training": True, "comparison": True, "endpoint": True},
        )
        self.store.save("training_job.json", {"job_name": "j", "job_status": "Completed"})

        self.assertFalse(self.ledger.remaining()["training"])

    def test_archived_baselines_are_counted_with_their_label(self) -> None:
        self.store.save("base_evaluation.pre_fix.json", {"arn": "arn:pipeline/v1"})
        self.store.save("base_evaluation.abc123.json", {"arn": "arn:pipeline/v0"})

        labels = [item["label"] for item in self.ledger.spent()]

        self.assertEqual(labels, ["Base evaluation (earlier attempt abc123)",
                                  "Base evaluation (first attempt)", "Base evaluation"])

    def test_pipeline_whose_step_never_ran_bills_nothing(self) -> None:
        self.inspector.step_jobs.return_value = {}

        self.assertEqual(self.ledger.spent(), [])

if __name__ == "__main__":
    unittest.main()

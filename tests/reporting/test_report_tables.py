"""Unit tests for turning recorded evidence into report tables."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from aws.records.evidence import EvidenceStore
from aws.reporting.report_tables import (
    KEY_METRICS,
    baseline_history,
    cost_table,
    evaluation_comparison,
    headline_results,
    live_inference,
    per_ticket_rewards,
    run_of_record,
    training_curve,
)

TICKET = "Resolve the customer's internet outage. Customer says: {}"


def conversation(reward: float | None) -> dict:
    """Builds one recorded evaluation conversation."""
    return {"ticket": "t", "reward": reward, "first_thought": "", "steps": []}


def episode(reward: float, actions: list[str]) -> dict:
    """Builds one live-inference episode result."""
    return {"reward": reward, "metrics": {"actions_taken": actions}, "summary": "", "trace": []}


class EvaluationTableTests(unittest.TestCase):
    """Verifies metric tables keep labels, order, and deltas."""

    def test_comparison_rows_follow_key_metric_order_with_deltas(self) -> None:
        results = {
            "base": {"metrics": {"eval/reward/pass_at_1": 0.5, "eval/reward/mean": 0.8}},
            "fine_tuned": {"metrics": {"eval/reward/pass_at_1": 0.75, "eval/reward/mean": 0.9}},
        }

        rows = evaluation_comparison(results)

        self.assertEqual([r["metric"] for r in rows], list(KEY_METRICS))
        first = rows[0]
        self.assertEqual((first["base"], first["fine_tuned"], first["delta"]), (0.5, 0.75, 0.25))
        missing = next(r for r in rows if r["metric"] == "eval/turns/mean")
        self.assertIsNone(missing["delta"])

    def test_baseline_history_shows_first_attempt_next_to_the_baseline(self) -> None:
        rows = baseline_history({"eval/reward/pass_at_1": 0}, {"eval/reward/pass_at_1": 0.5})

        self.assertEqual((rows[0]["first_attempt"], rows[0]["baseline"]), (0, 0.5))

    def test_baseline_history_without_a_first_attempt_leaves_it_empty(self) -> None:
        rows = baseline_history(None, {"eval/reward/pass_at_1": 0.5})

        self.assertIsNone(rows[0]["first_attempt"])


class TrainingCurveTests(unittest.TestCase):
    """Verifies per-step training rows are complete and ordered."""

    def test_steps_are_sorted_and_renamed(self) -> None:
        record = {"steps": [
            {"step": 2, "rollout/reward/mean": 0.9, "rollout/turns/mean": 6.5,
             "training/total_tokens": 500.0, "training/num_trajectories": 128.0},
            {"step": 1, "rollout/reward/mean": 0.8, "rollout/turns/mean": 6.0,
             "training/total_tokens": 400.0, "training/num_trajectories": 128.0},
        ]}

        rows = training_curve(record)

        self.assertEqual([r["step"] for r in rows], [1, 2])
        self.assertEqual(rows[0], {"step": 1, "mean_reward": 0.8, "mean_turns": 6.0,
                                   "total_tokens": 400, "trajectories": 128})


class PerTicketTests(unittest.TestCase):
    """Verifies per-ticket before/after rewards from recorded traces."""

    def test_rows_show_attempts_means_and_direction(self) -> None:
        recorded = {"pairs": [
            {"ticket": TICKET.format("no internet"),
             "before": [conversation(0.75), conversation(1.0)],
             "after": [conversation(1.0), conversation(1.0)]},
            {"ticket": TICKET.format("slow"),
             "before": [conversation(1.0), conversation(None)],
             "after": [conversation(0.75), conversation(0.75)]},
        ]}

        rows = per_ticket_rewards(recorded)

        self.assertEqual(rows[0]["ticket"], "no internet")
        self.assertEqual((rows[0]["before_rewards"], rows[0]["after_mean"]), ("0.75; 1.0", 1.0))
        self.assertEqual(rows[0]["change"], "improved")
        self.assertEqual((rows[1]["before_mean"], rows[1]["change"]), (1.0, "worse"))


class LiveInferenceTests(unittest.TestCase):
    """Verifies live before/after rows, including failed calls."""

    def test_rows_show_rewards_actions_and_errors(self) -> None:
        results = {"tickets": [{
            "ticket": TICKET.format("router blinking"),
            "before": episode(0.75, ["restart_router", "check_outage"]),
            "after": {"error": "ThrottlingException: slow down"},
        }]}

        row = live_inference(results)[0]

        self.assertEqual(row["ticket"], "router blinking")
        self.assertEqual(row["before_reward"], 0.75)
        self.assertEqual(row["before_actions"], "restart_router > check_outage")
        self.assertIsNone(row["after_reward"])
        self.assertIn("ThrottlingException", row["after_error"])


class EvidenceSummaryTests(unittest.TestCase):
    """Verifies run-of-record rows and headline numbers from an evidence folder."""

    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.store = EvidenceStore(Path(self.directory.name))
        self.store.save("training_job.json", {
            "job_name": "job-1", "job_status": "Completed", "duration_minutes": 19.2,
            "output_model_package_arn": "arn:model-package/g/1",
            "progress_info": {"CurrentStep": 10, "MaxSteps": 10}})
        self.store.save("comparison_evaluation_metrics.json", {
            "base": {"metrics": {"eval/reward/pass_at_1": 0.5}},
            "fine_tuned": {"metrics": {"eval/reward/pass_at_1": 0.75}}})
        self.store.save("inference_before_after.json", {"seed": 2026, "tickets": [
            {"ticket": "t", "before": episode(0.75, []), "after": episode(1.0, [])}]})

    def tearDown(self) -> None:
        self.directory.cleanup()

    def test_run_of_record_lists_recorded_stages_only(self) -> None:
        rows = run_of_record(self.store)

        stages = [r["stage"] for r in rows]
        self.assertIn("Training", stages)
        self.assertNotIn("Bedrock import", stages)
        training = next(r for r in rows if r["stage"] == "Training")
        self.assertEqual((training["identifier"], training["status"]), ("job-1", "Completed"))

    def test_endpoint_row_keeps_the_first_sentence_of_the_failure(self) -> None:
        self.store.save("endpoint.json", {
            "endpoint_name": "demo", "status": "Deleted", "billed_usd": 0.0,
            "failure_reason": ("FailedStatusError: Final Resource State: Failed. Failure Reason: Unable to "
                               "provision requested ML compute capacity due to InsufficientInstanceCapacity "
                               "error. Please retry using a different ML instance type.")})

        row = next(r for r in run_of_record(self.store) if r["stage"] == "SageMaker endpoint")

        self.assertEqual(row["detail"], "Unable to provision requested ML compute capacity due to "
                                        "InsufficientInstanceCapacity error; billed $0.0")

    def test_headline_results_count_solved_tickets(self) -> None:
        headline = headline_results(self.store)

        self.assertEqual(headline["comparison_pass_at_1"], {"base": 0.5, "fine_tuned": 0.75})
        self.assertEqual(headline["live_inference_solved"], {"before": 0, "after": 1, "tickets": 1})

    def test_cost_table_keeps_item_amount_and_source(self) -> None:
        snapshot = {"items": [{"item": "Training", "usd": 2.83, "source": "tokens", "usage": {}}]}

        self.assertEqual(cost_table(snapshot), [{"item": "Training", "usd": 2.83, "source": "tokens"}])


if __name__ == "__main__":
    unittest.main()

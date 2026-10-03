"""Unit tests for the run-or-attach MTRL training stage."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from aws.config import DemoConfig
from aws.records.evidence import EvidenceStore
from aws.rl.training import TrainingStage


def fake_job(status: str = "InProgress") -> MagicMock:
    """Builds a stand-in for an AgentRFTJob."""
    job = MagicMock()
    job.job_name = "mtrl-demo-job"
    job.job_arn = "arn:aws:sagemaker:us-west-2:1:job/AgentRFT/mtrl-demo-job"
    job.job_status = status
    job.secondary_status = "Training"
    job.failure_reason = None
    job.creation_time = datetime(2026, 10, 2, 0, 0, tzinfo=timezone.utc)
    job.end_time = datetime(2026, 10, 2, 1, 30, tzinfo=timezone.utc)
    job.output_model_package_arn = "arn:aws:sagemaker:us-west-2:1:model-package/g/1"
    job.billable_token_usage = {"PrefillTokenCount": 10, "SampleTokenCount": 2}
    job.progress_info = {"MaxSteps": 10, "CurrentStep": 10}
    return job


class TrainingStageTests(unittest.TestCase):
    """Verifies confirmation, duplicate protection, and evidence capture."""

    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.store = EvidenceStore(Path(self.directory.name))
        self.trainer = MagicMock()
        self.trainer.hyperparameters = SimpleNamespace()
        self.job = fake_job()
        self.trainer.train.return_value = self.job
        self.gate = MagicMock()
        self.attach = MagicMock(return_value=fake_job("Completed"))
        self.stage = TrainingStage(
            config=DemoConfig(),
            store=self.store,
            trainer_factory=lambda: self.trainer,
            attach=self.attach,
            signal_gate=self.gate,
        )

    def tearDown(self) -> None:
        self.directory.cleanup()

    def test_configure_applies_approved_training_size(self) -> None:
        trainer = self.stage.configure(self.trainer)
        hp = trainer.hyperparameters

        self.assertEqual(hp.max_steps, 10)
        self.assertEqual(hp.group_size, 4)
        self.assertEqual(hp.global_batch_size, 32)
        self.assertGreaterEqual(hp.max_epochs, hp.max_steps)
        self.assertEqual(hp.sampling_max_tokens, 512)

    def test_new_run_requires_exact_confirmation(self) -> None:
        with self.assertRaises(PermissionError):
            self.stage.run_or_attach("yes please")

        self.trainer.train.assert_not_called()

    def test_new_run_checks_learning_signal_then_submits_without_blocking(self) -> None:
        job = self.stage.run_or_attach(TrainingStage.CONFIRMATION)

        self.gate.assert_called_once()
        self.trainer.train.assert_called_once_with(wait=False)
        self.assertIs(job, self.job)
        self.assertEqual(self.store.load(TrainingStage.RECORD)["job_name"], "mtrl-demo-job")

    def test_gate_failure_prevents_submission(self) -> None:
        self.gate.side_effect = RuntimeError("no reward variance")

        with self.assertRaises(RuntimeError):
            self.stage.run_or_attach(TrainingStage.CONFIRMATION)

        self.trainer.train.assert_not_called()

    def test_recorded_job_is_attached_instead_of_resubmitted(self) -> None:
        self.store.save(TrainingStage.RECORD, {"job_name": "earlier-job"})

        job = self.stage.run_or_attach(TrainingStage.CONFIRMATION)

        self.attach.assert_called_once_with("earlier-job")
        self.trainer.train.assert_not_called()
        self.assertEqual(job.job_status, "Completed")

    def test_failed_record_is_archived_and_resubmitted_when_confirmed(self) -> None:
        self.store.save(TrainingStage.RECORD, {"job_name": "failed-job"})
        self.attach.return_value = fake_job("Failed")

        job = self.stage.run_or_attach(TrainingStage.RETRY_CONFIRMATION)

        self.assertIs(job, self.job)
        self.trainer.train.assert_called_once_with(wait=False)
        self.assertEqual(
            self.store.load("training_job.failed-job.json")["job_name"], "failed-job"
        )

    def test_standing_phrase_does_not_retry_a_failed_job(self) -> None:
        self.store.save(TrainingStage.RECORD, {"job_name": "failed-job"})
        self.attach.return_value = fake_job("Failed")

        job = self.stage.run_or_attach(TrainingStage.CONFIRMATION)

        self.assertEqual(job.job_status, "Failed")
        self.trainer.train.assert_not_called()

    def test_running_job_elsewhere_blocks_submission(self) -> None:
        self.stage = TrainingStage(
            config=DemoConfig(), store=self.store, trainer_factory=lambda: self.trainer,
            attach=self.attach, signal_gate=self.gate, active_jobs=lambda: ["other-job"],
        )

        with self.assertRaisesRegex(RuntimeError, "other-job"):
            self.stage.run_or_attach(TrainingStage.CONFIRMATION)

        self.trainer.train.assert_not_called()

    def test_wait_records_summary_even_when_waiting_fails(self) -> None:
        self.store.save(TrainingStage.RECORD, {"job_name": "mtrl-demo-job"})
        self.job.wait.side_effect = TimeoutError("still running")

        with self.assertRaises(TimeoutError):
            self.stage.wait(self.job, timeout_seconds=1)

        self.assertEqual(self.store.load(TrainingStage.RECORD)["job_status"], "InProgress")
        self.assertEqual(self.store.load(TrainingStage.RECORD)["job_name"], "mtrl-demo-job")

    def test_failed_record_without_confirmation_is_only_shown(self) -> None:
        self.store.save(TrainingStage.RECORD, {"job_name": "failed-job"})
        self.attach.return_value = fake_job("Failed")

        job = self.stage.run_or_attach("")

        self.assertEqual(job.job_status, "Failed")
        self.trainer.train.assert_not_called()

    def test_running_record_is_never_resubmitted(self) -> None:
        self.store.save(TrainingStage.RECORD, {"job_name": "running-job"})
        self.attach.return_value = fake_job("InProgress")

        self.stage.run_or_attach(TrainingStage.CONFIRMATION)

        self.trainer.train.assert_not_called()

    def test_replay_without_record_or_confirmation_returns_none(self) -> None:
        self.assertIsNone(self.stage.run_or_attach(""))

    def test_wait_skips_terminal_jobs_and_records_summary(self) -> None:
        self.store.save(TrainingStage.RECORD, {"job_name": "mtrl-demo-job"})
        job = fake_job("Completed")

        self.stage.wait(job, timeout_seconds=60)

        job.wait.assert_not_called()
        record = self.store.load(TrainingStage.RECORD)
        self.assertEqual(record["job_status"], "Completed")
        self.assertEqual(record["duration_minutes"], 90.0)
        self.assertEqual(record["output_model_package_arn"], job.output_model_package_arn)

    def test_wait_blocks_on_running_jobs(self) -> None:
        self.store.save(TrainingStage.RECORD, {"job_name": "mtrl-demo-job"})

        self.stage.wait(self.job, timeout_seconds=60)

        self.job.wait.assert_called_once_with(poll=30, timeout=60, max_log_lines=10)
        self.job.refresh.assert_called()

    def test_summary_tolerates_unassigned_sentinels(self) -> None:
        job = fake_job()
        job.end_time = object()

        self.assertIsNone(TrainingStage.summary(job)["duration_minutes"])

    def test_summary_tolerates_missing_end_time(self) -> None:
        job = fake_job()
        job.end_time = None

        self.assertIsNone(TrainingStage.summary(job)["duration_minutes"])

    def test_summary_records_the_jobs_own_training_and_agent_config(self) -> None:
        job = fake_job("Completed")
        job.training_config = {"BaseModelArn": "arn:hub-content/m/3.43.0",
                               "HyperParameters": {"max_steps": "10"}}
        job.agent_config = {"BedrockAgentCoreConfig": {"AgentRuntimeArn": "arn:runtime"}}

        summary = TrainingStage.summary(job)

        self.assertEqual(summary["training_config"]["HyperParameters"], {"max_steps": "10"})
        self.assertEqual(summary["agent_config"]["BedrockAgentCoreConfig"]["AgentRuntimeArn"], "arn:runtime")

    def test_summary_leaves_out_configs_that_are_not_mappings(self) -> None:
        summary = TrainingStage.summary(fake_job("Completed"))

        self.assertIsNone(summary["training_config"])
        self.assertIsNone(summary["agent_config"])

    def test_metrics_of_a_completed_job_are_recorded_per_step(self) -> None:
        job = fake_job("Completed")
        job.get_training_metrics.return_value = [
            {"step": 1, "rollout/reward/mean": 0.88},
            {"step": 2, "rollout/reward/mean": 0.9},
        ]

        steps = self.stage.record_metrics(job)

        saved = self.store.load(TrainingStage.METRICS_RECORD)
        self.assertEqual(saved["job_name"], "mtrl-demo-job")
        self.assertEqual([step["step"] for step in saved["steps"]], [1, 2])
        self.assertEqual(steps, saved["steps"])

    def test_metrics_are_not_recorded_before_the_job_completes(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "Completed"):
            self.stage.record_metrics(fake_job("InProgress"))

        self.assertIsNone(self.store.load(TrainingStage.METRICS_RECORD))


if __name__ == "__main__":
    unittest.main()

"""Unit tests for guarded SageMaker workflow behavior."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import boto3

from aws.records.evidence import EvidenceStore
from aws.workflow import MtrlDemoWorkflow, pin_sdk_region


class MtrlDemoWorkflowTests(unittest.TestCase):
    """Verifies billing gates and safe evidence serialization."""

    def setUp(self) -> None:
        """Creates a workflow without initializing AWS clients."""
        self.workflow = MtrlDemoWorkflow.__new__(MtrlDemoWorkflow)

    def test_training_signal_gate_rejects_identical_rewards(self) -> None:
        with (
            patch.object(
                self.workflow,
                "_base_evaluation_is_current",
                return_value=True,
            ),
            patch.object(
                self.workflow,
                "_fetch_base_metrics",
                return_value={
                    "eval/reward/min": 0,
                    "eval/reward/max": 0,
                },
            ),
            self.assertRaisesRegex(RuntimeError, "no reward variance"),
        ):
            self.workflow._require_learning_signal()

    def test_training_signal_gate_accepts_reward_variance(self) -> None:
        with (
            patch.object(
                self.workflow,
                "_base_evaluation_is_current",
                return_value=True,
            ),
            patch.object(
                self.workflow,
                "_fetch_base_metrics",
                return_value={
                    "eval/reward/min": 0,
                    "eval/reward/max": 1,
                },
            ),
        ):
            self.workflow._require_learning_signal()

    def test_training_signal_gate_rejects_stale_metrics(self) -> None:
        with (
            patch.object(
                self.workflow,
                "_base_evaluation_is_current",
                return_value=False,
            ),
            self.assertRaisesRegex(RuntimeError, "current dataset"),
        ):
            self.workflow._require_learning_signal()

    def test_execution_evidence_uses_allowlist(self) -> None:
        execution = SimpleNamespace(
            arn="arn:aws:sagemaker:us-west-2:123:pipeline/example",
            name="evaluation",
            mlflow_url="https://example.invalid/?authToken=secret",
            api_key="secret",
        )
        with tempfile.TemporaryDirectory() as directory:
            self.workflow.store = EvidenceStore(Path(directory))
            self.workflow._save_execution("evidence.json", execution)
            payload = json.loads((Path(directory) / "evidence.json").read_text())

        self.assertEqual(
            payload,
            {
                "arn": "arn:aws:sagemaker:us-west-2:123:pipeline/example",
                "name": "evaluation",
            },
        )

    def test_sdk_region_is_repinned_when_another_region_won(self) -> None:
        from sagemaker.core.utils.utils import SageMakerClient, SingletonMeta

        wrong = SimpleNamespace(region_name="me-central-1")
        session = boto3.Session(region_name="us-west-2")
        with patch.dict(SingletonMeta._instances, {SageMakerClient: wrong}):
            pin_sdk_region(session, "us-west-2")
            pinned = SingletonMeta._instances[SageMakerClient]

        self.assertEqual(pinned.region_name, "us-west-2")

    def test_matching_sdk_region_is_left_alone(self) -> None:
        from sagemaker.core.utils.utils import SageMakerClient, SingletonMeta

        right = SimpleNamespace(region_name="us-west-2")
        with patch.dict(SingletonMeta._instances, {SageMakerClient: right}):
            pin_sdk_region(boto3.Session(region_name="us-west-2"), "us-west-2")
            kept = SingletonMeta._instances[SageMakerClient]

        self.assertIs(kept, right)


class BaseEvaluationRunOrAttachTests(unittest.TestCase):
    """Verifies a recorded baseline is re-attached, never submitted twice."""

    RECORD = {"arn": "arn:aws:sagemaker:us-west-2:1:pipeline/p/execution/1"}
    METRICS = {"eval/reward/pass_at_1": 0.5}

    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.workflow = MtrlDemoWorkflow.__new__(MtrlDemoWorkflow)
        self.workflow.store = EvidenceStore(Path(self.directory.name))
        self.workflow._inspector = MagicMock()
        self.workflow._inspector.wait.return_value = "Succeeded"

    def tearDown(self) -> None:
        self.directory.cleanup()

    def test_nothing_recorded_and_no_confirmation_submits_nothing(self) -> None:
        with patch.object(self.workflow, "submit_base_evaluation") as submit:
            self.assertIsNone(self.workflow.run_or_attach_base_evaluation(""))

        submit.assert_not_called()

    def test_confirmation_submits_when_nothing_is_recorded(self) -> None:
        phrase = MtrlDemoWorkflow.BASE_EVAL_CONFIRMATION
        with patch.object(self.workflow, "submit_base_evaluation") as submit:
            self.workflow.run_or_attach_base_evaluation(phrase)

        submit.assert_called_once_with(phrase)

    def test_recorded_baseline_with_metrics_is_only_shown(self) -> None:
        self.workflow.store.save("base_evaluation.json", self.RECORD)
        self.workflow.store.save("base_evaluation_metrics.json", self.METRICS)
        with patch.object(self.workflow, "submit_base_evaluation") as submit:
            evidence = self.workflow.run_or_attach_base_evaluation(MtrlDemoWorkflow.BASE_EVAL_CONFIRMATION)

        submit.assert_not_called()
        self.workflow._inspector.wait.assert_not_called()
        self.assertEqual(evidence["metrics"], self.METRICS)

    def test_interrupted_baseline_is_awaited_instead_of_resubmitted(self) -> None:
        self.workflow.store.save("base_evaluation.json", self.RECORD)
        with (
            patch.object(self.workflow, "submit_base_evaluation") as submit,
            patch.object(self.workflow, "_fetch_base_metrics", return_value=self.METRICS),
        ):
            evidence = self.workflow.run_or_attach_base_evaluation("")

        submit.assert_not_called()
        self.workflow._inspector.wait.assert_called_once_with(self.RECORD["arn"], timeout_seconds=3_600)
        self.assertEqual(evidence["metrics"], self.METRICS)
        self.assertEqual(self.workflow.store.load("base_evaluation.json")["status"], "Succeeded")

    def test_failed_pipeline_is_recorded_without_metrics(self) -> None:
        self.workflow.store.save("base_evaluation.json", self.RECORD)
        self.workflow._inspector.wait.return_value = "Failed"
        with (
            patch.object(self.workflow, "_fetch_base_metrics") as fetch,
            self.assertRaisesRegex(RuntimeError, MtrlDemoWorkflow.BASE_EVAL_RETRY_CONFIRMATION),
        ):
            self.workflow.run_or_attach_base_evaluation("")

        fetch.assert_not_called()
        self.assertEqual(self.workflow.store.load("base_evaluation.json")["status"], "Failed")
        self.assertIsNone(self.workflow.store.load("base_evaluation_metrics.json"))

    def test_retry_phrase_archives_the_failed_baseline_and_submits_a_new_one(self) -> None:
        self.workflow.store.save("base_evaluation.json", {**self.RECORD, "status": "Failed"})
        with patch.object(self.workflow, "submit_base_evaluation") as submit:
            self.workflow.run_or_attach_base_evaluation(MtrlDemoWorkflow.BASE_EVAL_RETRY_CONFIRMATION)

        submit.assert_called_once_with(MtrlDemoWorkflow.BASE_EVAL_CONFIRMATION)
        self.assertEqual(self.workflow.store.load("base_evaluation.1.json")["status"], "Failed")
        self.assertIsNone(self.workflow.store.load("base_evaluation.json"))


class TrainingGateTests(unittest.TestCase):
    """Verifies training is blocked without a learning signal or above the budget."""

    def test_signal_then_budget_are_checked_before_training(self) -> None:
        workflow = MtrlDemoWorkflow.__new__(MtrlDemoWorkflow)
        calls: list[str] = []
        with (
            patch.object(workflow, "_require_learning_signal", side_effect=lambda: calls.append("signal")),
            patch.object(workflow, "budget_gate", side_effect=lambda: calls.append("budget")),
        ):
            workflow._before_training()

        self.assertEqual(calls, ["signal", "budget"])


class LiveResourceTests(unittest.TestCase):
    """Verifies setup validation asks AWS, not the local state file."""

    def setUp(self) -> None:
        self.workflow = MtrlDemoWorkflow.__new__(MtrlDemoWorkflow)
        self.workflow.state = SimpleNamespace(agent_runtime_id="runtime-1", mlflow_app_arn="arn:mlflow")
        self.clients = {"bedrock-agentcore-control": MagicMock(), "sagemaker": MagicMock()}
        self.clients["bedrock-agentcore-control"].get_agent_runtime.return_value = {"status": "READY"}
        self.clients["sagemaker"].describe_mlflow_app.return_value = {"Status": "Created"}
        self.workflow.session = SimpleNamespace(client=lambda name: self.clients[name])

    def test_ready_runtime_and_created_app_pass(self) -> None:
        self.assertEqual(self.workflow.live_resources(),
                         {"agent_runtime_status": "READY", "mlflow_app_status": "Created"})

    def test_runtime_that_is_not_ready_in_aws_fails(self) -> None:
        self.clients["bedrock-agentcore-control"].get_agent_runtime.return_value = {"status": "UPDATE_FAILED"}
        with self.assertRaisesRegex(RuntimeError, "UPDATE_FAILED"):
            self.workflow.live_resources()

    def test_missing_runtime_id_fails_without_calling_aws(self) -> None:
        self.workflow.state = SimpleNamespace(agent_runtime_id=None, mlflow_app_arn="arn:mlflow")
        with self.assertRaisesRegex(RuntimeError, "03_deploy_agent"):
            self.workflow.live_resources()

if __name__ == "__main__":
    unittest.main()

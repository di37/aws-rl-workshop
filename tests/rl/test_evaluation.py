"""Unit tests for the comparison evaluation stage and pipeline inspector."""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from aws.config import DemoConfig
from aws.records.evidence import EvidenceStore
from aws.rl.evaluation import EvaluationStage, configure_evaluator, start_quietly
from aws.rl.pipelines import PipelineInspector

EXECUTION_ARN = "arn:aws:sagemaker:us-west-2:1:pipeline/p/execution/abc"


class EvaluationStageTests(unittest.TestCase):
    """Verifies confirmation, duplicate protection, and metric collection."""

    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.store = EvidenceStore(Path(self.directory.name))
        self.execution = SimpleNamespace(arn=EXECUTION_ARN, name="eval-1")
        self.evaluator = MagicMock()
        self.evaluator.evaluate.return_value = self.execution
        self.factory = MagicMock(return_value=self.evaluator)
        self.inspector = MagicMock()
        self.inspector.step_jobs.return_value = {
            "EvaluateBaseModel": "arn:job/base",
            "EvaluateFineTunedModel": "arn:job/tuned",
        }
        self.inspector.eval_job_details.side_effect = lambda arn: {
            "run_id": f"run-{arn[-4:]}",
            "billable_token_usage": {"PrefillTokenCount": 5},
        }
        self.inspector.run_metrics.side_effect = lambda run_id: {
            "eval/reward/pass_at_1": 0.5 if run_id == "run-base" else 0.9
        }
        self.stage = EvaluationStage(DemoConfig(), self.store, self.factory, self.inspector)
        self.trained = SimpleNamespace(job_status="Completed")

    def tearDown(self) -> None:
        self.directory.cleanup()

    def test_new_evaluation_requires_exact_confirmation(self) -> None:
        with self.assertRaises(PermissionError):
            self.stage.run_or_attach("ok", self.trained)

        self.factory.assert_not_called()

    def test_evaluation_requires_completed_training(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "completed training job"):
            self.stage.run_or_attach(
                EvaluationStage.CONFIRMATION, SimpleNamespace(job_status="InProgress")
            )

    def test_confirmed_evaluation_starts_comparison_and_records_it(self) -> None:
        arn = self.stage.run_or_attach(EvaluationStage.CONFIRMATION, self.trained)

        self.factory.assert_called_once_with(self.trained)
        self.evaluator.evaluate.assert_called_once_with()
        self.assertEqual(arn, EXECUTION_ARN)
        self.assertEqual(self.store.load(EvaluationStage.RECORD)["arn"], EXECUTION_ARN)

    def test_recorded_evaluation_is_reused(self) -> None:
        self.store.save(EvaluationStage.RECORD, {"arn": "arn:previous"})

        arn = self.stage.run_or_attach(EvaluationStage.CONFIRMATION, self.trained)

        self.assertEqual(arn, "arn:previous")
        self.factory.assert_not_called()

    def test_replay_without_record_returns_none(self) -> None:
        self.assertIsNone(self.stage.run_or_attach("", self.trained))

    def test_results_pair_each_model_with_its_own_mlflow_run(self) -> None:
        results = self.stage.results(EXECUTION_ARN)

        self.assertEqual(results["base"]["metrics"]["eval/reward/pass_at_1"], 0.5)
        self.assertEqual(results["fine_tuned"]["metrics"]["eval/reward/pass_at_1"], 0.9)
        self.assertEqual(results["base"]["job_arn"], "arn:job/base")
        saved = self.store.load(EvaluationStage.METRICS)
        self.assertEqual(saved["fine_tuned"]["run_id"], "run-uned")

    def test_results_fail_clearly_when_a_step_is_missing(self) -> None:
        self.inspector.step_jobs.return_value = {"EvaluateBaseModel": "arn:job/base"}

        with self.assertRaisesRegex(RuntimeError, "EvaluateFineTunedModel"):
            self.stage.results(EXECUTION_ARN)

    def test_budget_gate_refusal_starts_nothing(self) -> None:
        gate = MagicMock(side_effect=PermissionError("over the $25 cap"))
        stage = EvaluationStage(DemoConfig(), self.store, self.factory, self.inspector, budget_gate=gate)

        with self.assertRaises(PermissionError):
            stage.run_or_attach(EvaluationStage.CONFIRMATION, self.trained)

        self.factory.assert_not_called()

    def test_recorded_pipeline_is_attached_without_the_gate(self) -> None:
        gate = MagicMock()
        stage = EvaluationStage(DemoConfig(), self.store, self.factory, self.inspector, budget_gate=gate)
        self.store.save(EvaluationStage.RECORD, {"arn": "arn:previous"})

        self.assertEqual(stage.run_or_attach(EvaluationStage.CONFIRMATION, self.trained), "arn:previous")
        gate.assert_not_called()


class EvaluatorHelperTests(unittest.TestCase):
    """Verifies shared evaluator settings and link redaction at start."""

    def test_configure_evaluator_applies_shared_settings(self) -> None:
        evaluator = SimpleNamespace(hyperparameters=SimpleNamespace())

        configure_evaluator(evaluator, DemoConfig())

        hp = evaluator.hyperparameters
        self.assertEqual((hp.eval_group_size, hp.sampling_max_tokens, hp.pass_k_values), (2, 512, [1, 2]))

    def test_start_quietly_redacts_printed_links(self) -> None:
        class NoisyEvaluator:
            def evaluate(self) -> str:
                print("MLflow: https://x.example/auth?authToken=secret-token")
                return "execution"

        with patch("sys.stdout", new_callable=io.StringIO) as shown:
            execution = start_quietly(NoisyEvaluator())

        self.assertEqual(execution, "execution")
        self.assertNotIn("secret-token", shown.getvalue())


class PipelineInspectorTests(unittest.TestCase):
    """Verifies step discovery, job details, and terminal-status waiting."""

    def setUp(self) -> None:
        self.sagemaker = MagicMock()
        self.inspector = PipelineInspector(
            sagemaker_client=self.sagemaker,
            job_loader=MagicMock(),
            metrics_reader=MagicMock(),
            sleep=lambda _: None,
        )

    def test_step_jobs_map_step_names_to_job_arns(self) -> None:
        self.sagemaker.list_pipeline_execution_steps.return_value = {
            "PipelineExecutionSteps": [
                {"StepName": "EvaluateBaseModel", "Metadata": {"Job": {"Arn": "arn:b"}}},
                {"StepName": "CreateEvaluationAction", "Metadata": {}},
            ]
        }

        self.assertEqual(
            self.inspector.step_jobs(EXECUTION_ARN), {"EvaluateBaseModel": "arn:b"}
        )

    def test_eval_job_details_read_service_output(self) -> None:
        document = {
            "ServiceOutput": {
                "MlflowDetails": {"RunId": "r1", "ExperimentName": "e"},
                "BillableTokenUsage": {"PrefillTokenCount": 7},
            }
        }
        self.inspector._job_loader.return_value = SimpleNamespace(
            job_config_document=json.dumps(document)
        )

        details = self.inspector.eval_job_details(
            "arn:aws:sagemaker:us-west-2:1:job/AgentRFTEvaluation/pipelines-x-Eval"
        )

        self.inspector._job_loader.assert_called_once_with(
            "pipelines-x-Eval", "AgentRFTEvaluation"
        )
        self.assertEqual(details["run_id"], "r1")
        self.assertEqual(details["billable_token_usage"], {"PrefillTokenCount": 7})

    def test_wait_returns_terminal_status(self) -> None:
        self.sagemaker.describe_pipeline_execution.side_effect = [
            {"PipelineExecutionStatus": "Executing"},
            {"PipelineExecutionStatus": "Succeeded"},
        ]
        self.sagemaker.list_pipeline_execution_steps.return_value = {
            "PipelineExecutionSteps": []
        }

        self.assertEqual(self.inspector.wait(EXECUTION_ARN, timeout_seconds=600), "Succeeded")

    def test_wait_times_out(self) -> None:
        self.sagemaker.describe_pipeline_execution.return_value = {
            "PipelineExecutionStatus": "Executing"
        }
        self.sagemaker.list_pipeline_execution_steps.return_value = {
            "PipelineExecutionSteps": []
        }

        with self.assertRaises(TimeoutError):
            self.inspector.wait(EXECUTION_ARN, timeout_seconds=0)

if __name__ == "__main__":
    unittest.main()

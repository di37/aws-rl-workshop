"""Base-versus-fine-tuned evaluation stage, following the official guide.

The guide's comparison is one ``MultiTurnRLEvaluator(model=trainer,
evaluate_base_model=True)`` pipeline. Both eval jobs share one S3 output path,
so each model's metrics are read from its own MLflow run instead.
"""

from __future__ import annotations

import contextlib
import io
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from aws.config import DemoConfig, ResourceState
from aws.guards import require_phrase
from aws.records.evidence import EvidenceStore
from aws.rl.pipelines import PipelineInspector


class EvaluationStage:
    """Starts, re-attaches to, and collects one comparison evaluation."""

    RECORD = "comparison_evaluation.json"
    METRICS = "comparison_evaluation_metrics.json"
    CONFIRMATION = "CONFIRM_COMPARISON_EVAL_MTRL_DEMO"
    STEPS = (("EvaluateBaseModel", "base"), ("EvaluateFineTunedModel", "fine_tuned"))

    def __init__(
        self,
        config: DemoConfig,
        store: EvidenceStore,
        evaluator_factory: Callable[[Any], Any],
        inspector: PipelineInspector,
        budget_gate: Callable[[], None] = lambda: None,
    ) -> None:
        """Initializes the stage.

        Args:
            config: Shared demo settings.
            store: Evidence store holding the execution record and metrics.
            evaluator_factory: Builds a configured evaluator for a training job.
            inspector: Reads pipeline steps, eval jobs, and MLflow metrics.
            budget_gate: Raises PermissionError when the projected spend exceeds the cap.
        """
        self.config = config
        self.store = store
        self._evaluator_factory = evaluator_factory
        self.inspector = inspector
        self._budget_gate = budget_gate

    @classmethod
    def create(
        cls,
        config: DemoConfig,
        state: ResourceState,
        sagemaker_session: Any,
        store: EvidenceStore,
        dataset_uri: str,
        inspector: PipelineInspector,
        budget_gate: Callable[[], None] = lambda: None,
    ) -> EvaluationStage:
        """Builds the stage wired to the real SageMaker SDK.

        Args:
            config: Shared demo settings.
            state: Provisioned resource identifiers.
            sagemaker_session: SageMaker SDK session.
            store: Evidence store.
            dataset_uri: S3 URI of the held-out evaluation prompts.
            inspector: Pipeline inspector.
            budget_gate: Raises PermissionError when the projected spend exceeds the cap.

        Returns:
            An evaluation stage that starts real pipelines.
        """
        from sagemaker.train.evaluate import MultiTurnRLEvaluator

        def build(training_job: Any) -> Any:
            evaluator = MultiTurnRLEvaluator(
                model=training_job,
                dataset=dataset_uri,
                s3_output_path=f"s3://{state.bucket}/comparison-evaluation/",
                evaluate_base_model=True,
                mlflow_resource_arn=state.mlflow_app_arn,
                role=state.job_role_arn,
                sagemaker_session=sagemaker_session,
                accept_eula=True,
            )
            return configure_evaluator(evaluator, config)

        return cls(config, store, build, inspector, budget_gate)

    def run_or_attach(self, confirmation: str, training_job: Any | None) -> str | None:
        """Re-uses the recorded comparison, or starts one when confirmed.

        Args:
            confirmation: Exact phrase that authorizes a new billable pipeline.
            training_job: Completed training job whose model is evaluated.

        Returns:
            Pipeline execution ARN, or None in replay mode with no record.

        Raises:
            PermissionError: If a non-empty confirmation phrase is wrong.
            RuntimeError: If the training job has not completed.
        """
        record = self.store.load(self.RECORD)
        if record:
            return record["arn"]
        if not confirmation:
            return None
        require_phrase(confirmation, self.CONFIRMATION, "evaluation")
        if training_job is None or training_job.job_status != "Completed":
            raise RuntimeError("Comparison needs a completed training job.")
        self._budget_gate()
        execution = start_quietly(self._evaluator_factory(training_job))
        self.store.save(
            self.RECORD,
            {
                "arn": execution.arn,
                "name": execution.name,
                "started_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        return execution.arn

    def wait(self, execution_arn: str, timeout_seconds: int) -> str:
        """Waits for the pipeline and records its terminal status.

        Args:
            execution_arn: Pipeline execution ARN.
            timeout_seconds: Maximum time to wait.

        Returns:
            Terminal pipeline status.
        """
        status = self.inspector.wait(execution_arn, timeout_seconds)
        record = self.store.load(self.RECORD) or {"arn": execution_arn}
        self.store.save(self.RECORD, {**record, "status": status})
        return status

    def results(self, execution_arn: str) -> dict[str, dict[str, Any]]:
        """Collects base and fine-tuned metrics from their MLflow runs.

        Args:
            execution_arn: Completed comparison pipeline execution ARN.

        Returns:
            ``{"base": {...}, "fine_tuned": {...}}`` with metrics and billed tokens.

        Raises:
            RuntimeError: If an expected evaluation step did not run a job.
        """
        jobs = self.inspector.step_jobs(execution_arn)
        results: dict[str, dict[str, Any]] = {}
        for step_name, label in self.STEPS:
            if step_name not in jobs:
                raise RuntimeError(f"Pipeline step {step_name} has no evaluation job.")
            details = self.inspector.eval_job_details(jobs[step_name])
            results[label] = {
                "job_arn": jobs[step_name],
                **details,
                "metrics": self.inspector.run_metrics(details["run_id"]),
            }
        self.store.save(self.METRICS, results)
        return results


def configure_evaluator(evaluator: Any, config: DemoConfig) -> Any:
    """Applies the evaluation settings shared by every pipeline in the demo.

    Args:
        evaluator: ``MultiTurnRLEvaluator`` to configure.
        config: Shared settings.

    Returns:
        The same evaluator.
    """
    hp = evaluator.hyperparameters
    hp.eval_group_size = config.eval_group_size
    hp.sampling_max_tokens = config.sampling_max_tokens
    hp.pass_k_values = list(config.pass_k_values)
    return evaluator


def start_quietly(evaluator: Any) -> Any:
    """Starts an evaluation, re-printing the SDK output with signed links removed.

    Args:
        evaluator: Configured ``MultiTurnRLEvaluator``.

    Returns:
        The started pipeline execution.
    """
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        execution = evaluator.evaluate()
    shown = EvidenceStore.redact(captured.getvalue()).strip()
    if shown:
        print(shown)
    return execution

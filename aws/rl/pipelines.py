"""Read-only helpers for SageMaker evaluation pipelines and their MLflow runs.

Polling is done here instead of ``execution.wait()`` because the SDK's Jupyter
renderer embeds a presigned MLflow link in the cell output.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from typing import Any

TERMINAL_PIPELINE_STATUSES = {"Succeeded", "Failed", "Stopped"}


class PipelineInspector:
    """Inspects pipeline executions, their eval jobs, and MLflow metrics."""

    def __init__(
        self,
        sagemaker_client: Any,
        job_loader: Callable[[str, str], Any],
        metrics_reader: Callable[[str], dict[str, float]],
        sleep: Callable[[float], None] = time.sleep,
        poll_seconds: int = 20,
    ) -> None:
        """Initializes the inspector with injectable AWS entry points.

        Args:
            sagemaker_client: Boto3 SageMaker client.
            job_loader: Loads a ``Job`` resource by name and category.
            metrics_reader: Returns the metrics dict for an MLflow run ID.
            sleep: Sleep function, injectable for tests.
            poll_seconds: Seconds between status polls.
        """
        self._sagemaker = sagemaker_client
        self._job_loader = job_loader
        self._metrics_reader = metrics_reader
        self._sleep = sleep
        self._poll_seconds = poll_seconds

    @classmethod
    def create(cls, session: Any, region: str, mlflow_arn: str) -> PipelineInspector:
        """Builds an inspector wired to SageMaker and the managed MLflow app.

        Args:
            session: Boto3 session.
            region: AWS region of the jobs.
            mlflow_arn: ARN of the managed MLflow app used as tracking URI.

        Returns:
            A ready inspector.
        """
        from sagemaker.core.resources import Job

        def load(job_name: str, category: str) -> Any:
            return Job.get(job_name=job_name, job_category=category, region=region)

        def read(run_id: str) -> dict[str, float]:
            import mlflow
            from mlflow.tracking import MlflowClient

            mlflow.set_tracking_uri(mlflow_arn)
            return dict(MlflowClient().get_run(run_id).data.metrics)

        return cls(session.client("sagemaker"), load, read)

    def step_jobs(self, execution_arn: str) -> dict[str, str]:
        """Maps each job-backed pipeline step to its job ARN.

        Args:
            execution_arn: Pipeline execution ARN.

        Returns:
            Step name to job ARN, for steps that ran a job.
        """
        steps = self._sagemaker.list_pipeline_execution_steps(
            PipelineExecutionArn=execution_arn
        )["PipelineExecutionSteps"]
        return {
            step["StepName"]: step["Metadata"]["Job"]["Arn"]
            for step in steps
            if step.get("Metadata", {}).get("Job", {}).get("Arn")
        }

    def eval_job_details(self, job_arn: str) -> dict[str, Any]:
        """Reads the MLflow run and billed tokens of one evaluation job.

        Args:
            job_arn: ARN such as ``...:job/AgentRFTEvaluation/<name>``.

        Returns:
            MLflow run ID, experiment name, and billable token usage.
        """
        category, job_name = job_arn.split(":job/", 1)[1].split("/", 1)
        job = self._job_loader(job_name, category)
        output = json.loads(job.job_config_document or "{}").get("ServiceOutput", {})
        mlflow_details = output.get("MlflowDetails", {})
        return {
            "run_id": mlflow_details.get("RunId"),
            "experiment_name": mlflow_details.get("ExperimentName"),
            "billable_token_usage": output.get("BillableTokenUsage"),
        }

    def run_metrics(self, run_id: str) -> dict[str, float]:
        """Reads all metrics logged to one MLflow run.

        Args:
            run_id: MLflow run ID.

        Returns:
            Metric name to latest value.
        """
        return self._metrics_reader(run_id)

    def wait(self, execution_arn: str, timeout_seconds: int) -> str:
        """Polls a pipeline execution until it reaches a terminal status.

        Args:
            execution_arn: Pipeline execution ARN.
            timeout_seconds: Maximum time to wait.

        Returns:
            Terminal status: ``Succeeded``, ``Failed``, or ``Stopped``.

        Raises:
            TimeoutError: If the execution is still running at the deadline.
        """
        started = time.monotonic()
        reported: dict[str, str] = {}
        while True:
            status = self._sagemaker.describe_pipeline_execution(
                PipelineExecutionArn=execution_arn
            )["PipelineExecutionStatus"]
            self._report_steps(execution_arn, reported)
            if status in TERMINAL_PIPELINE_STATUSES:
                print(f"Pipeline {status}.", flush=True)
                return status
            if time.monotonic() - started >= timeout_seconds:
                raise TimeoutError(f"Pipeline still {status} after {timeout_seconds}s.")
            self._sleep(self._poll_seconds)

    def _report_steps(self, execution_arn: str, reported: dict[str, str]) -> None:
        """Prints each step whose status changed since the last poll.

        Args:
            execution_arn: Pipeline execution ARN.
            reported: Last printed status per step; updated in place by design
                because it is the poll loop's private progress cursor.
        """
        steps = self._sagemaker.list_pipeline_execution_steps(
            PipelineExecutionArn=execution_arn
        )["PipelineExecutionSteps"]
        for step in reversed(steps):
            name, status = step["StepName"], step["StepStatus"]
            if reported.get(name) != status:
                reported[name] = status
                print(f"  {time.strftime('%H:%M:%S')}  {name}: {status}", flush=True)

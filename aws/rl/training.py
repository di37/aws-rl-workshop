"""Run-or-attach SageMaker MTRL training stage, following the official guide.

The guide's flow is ``trainer.train()`` → ``job.wait()`` → ``job.get_training_metrics()``.
This stage adds two safeguards around it: an exact confirmation phrase before a
billable submission, and a persisted job record so re-running the notebook
re-attaches (``MultiTurnRLTrainer.attach``) instead of paying for a second job.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from typing import Any

from aws.config import MODEL_ID, DemoConfig, ResourceState
from aws.guards import require_phrase
from aws.records.evidence import EvidenceStore

TERMINAL_STATUSES = {"Completed", "Failed", "Stopped"}
RETRYABLE_STATUSES = {"Failed", "Stopped"}


class TrainingStage:
    """Submits, re-attaches to, and summarizes one MTRL training job."""

    RECORD = "training_job.json"
    CONFIRMATION = "CONFIRM_TRAIN_MTRL_DEMO"
    RETRY_CONFIRMATION = "CONFIRM_RETRY_TRAIN_MTRL_DEMO"
    METRICS_RECORD = "training_metrics.json"

    def __init__(
        self,
        config: DemoConfig,
        store: EvidenceStore,
        trainer_factory: Callable[[], Any],
        attach: Callable[[str], Any],
        signal_gate: Callable[[], None],
        active_jobs: Callable[[], list[str]] = list,
    ) -> None:
        """Initializes the stage with injectable SDK entry points.

        Args:
            config: Shared demo settings, including the approved training size.
            store: Evidence store holding the job record.
            trainer_factory: Builds an unconfigured ``MultiTurnRLTrainer``.
            attach: Re-attaches to a job by name (``MultiTurnRLTrainer.attach``).
            signal_gate: Raises when the base evaluation has no learning signal.
            active_jobs: Returns names of MTRL training jobs still running.
        """
        self.config = config
        self.store = store
        self._trainer_factory = trainer_factory
        self._attach = attach
        self._signal_gate = signal_gate
        self._active_jobs = active_jobs

    @classmethod
    def create(
        cls,
        config: DemoConfig,
        state: ResourceState,
        sagemaker_session: Any,
        store: EvidenceStore,
        dataset_uri: str,
        signal_gate: Callable[[], None],
    ) -> TrainingStage:
        """Builds the stage wired to the real SageMaker SDK.

        Args:
            config: Shared demo settings.
            state: Provisioned resource identifiers.
            sagemaker_session: SageMaker SDK session.
            store: Evidence store holding the job record.
            dataset_uri: S3 URI of the training prompts.
            signal_gate: Raises when the base evaluation has no learning signal.

        Returns:
            A training stage that submits real jobs.
        """
        from sagemaker.train.multi_turn_rl_trainer import MultiTurnRLTrainer

        def build() -> Any:
            return MultiTurnRLTrainer(
                model=MODEL_ID,
                agent_env=state.agent_runtime_arn,
                training_dataset=dataset_uri,
                mlflow_app_arn=state.mlflow_app_arn,
                s3_output_path=f"s3://{state.bucket}/training-output/",
                role=state.job_role_arn,
                sagemaker_session=sagemaker_session,
                accept_eula=True,
                mlflow_experiment_name="mtrl-customer-support-demo",
                mlflow_run_name=(
                    f"train-{config.training_steps}-steps-"
                    f"group-{config.training_group_size}"
                ),
            )

        from sagemaker.train.agent_rft_job import AgentRFTJob

        boto_session = sagemaker_session.boto_session

        def attach(job_name: str) -> Any:
            return MultiTurnRLTrainer.attach(job_name, session=boto_session)

        def active() -> list[str]:
            return [
                job.job_name
                for job in AgentRFTJob.get_all(session=boto_session, status_equals="InProgress")
            ]

        return cls(config, store, build, attach, signal_gate, active)

    def configure(self, trainer: Any) -> Any:
        """Applies the approved training size to a trainer.

        Args:
            trainer: Unsubmitted ``MultiTurnRLTrainer``.

        Returns:
            The same trainer, ready to submit.
        """
        hp = trainer.hyperparameters
        hp.max_steps = self.config.training_steps
        hp.max_epochs = self.config.training_steps
        hp.global_batch_size = self.config.training_batch_size
        hp.group_size = self.config.training_group_size
        hp.sampling_max_tokens = self.config.sampling_max_tokens
        return trainer

    def build_trainer(self) -> Any:
        """Builds a configured trainer without submitting it.

        Returns:
            A configured ``MultiTurnRLTrainer``.
        """
        return self.configure(self._trainer_factory())

    def run_or_attach(self, confirmation: str) -> Any | None:
        """Re-attaches to the recorded job, or submits a new one when confirmed.

        A recorded job is never submitted twice. Only a recorded job that
        failed or was stopped may be replaced, and only with the separate
        retry phrase; its record is archived first. No job is submitted while
        another MTRL training job is running.

        Args:
            confirmation: Exact phrase that authorizes a new billable job.

        Returns:
            The training job, or None in replay mode with no recorded job.

        Raises:
            PermissionError: If a non-empty confirmation phrase is wrong.
        """
        record = self.store.load(self.RECORD)
        expected = self.CONFIRMATION
        if record:
            job = self._attach(record["job_name"])
            if job.job_status not in RETRYABLE_STATUSES or confirmation != self.RETRY_CONFIRMATION:
                return job
            self.store.save(f"training_job.{record['job_name']}.json", record)
            expected = self.RETRY_CONFIRMATION
        elif not confirmation:
            return None
        require_phrase(confirmation, expected, "training")
        running = self._active_jobs()
        if running:
            raise RuntimeError(f"An MTRL training job is already running: {', '.join(running)}")
        self._signal_gate()
        job = self.build_trainer().train(wait=False)
        self.store.save(
            self.RECORD,
            {
                "job_name": job.job_name,
                "submitted_at": datetime.now(timezone.utc).isoformat(),
                "training_steps": self.config.training_steps,
                "group_size": self.config.training_group_size,
                "global_batch_size": self.config.training_batch_size,
            },
        )
        return job

    def wait(self, job: Any, timeout_seconds: int) -> Any:
        """Waits for completion and records a redacted job summary.

        Args:
            job: Training job returned by :meth:`run_or_attach`.
            timeout_seconds: Maximum time to wait for a running job.

        Returns:
            The refreshed job.
        """
        try:
            if job.job_status not in TERMINAL_STATUSES:
                job.wait(poll=30, timeout=timeout_seconds, max_log_lines=10)
        finally:
            job.refresh()
            record = self.store.load(self.RECORD) or {}
            self.store.save(self.RECORD, {**record, **self.summary(job)})
        return job

    def record_metrics(self, job: Any) -> list[dict[str, Any]]:
        """Saves the per-step metrics of a completed job for offline reports.

        Args:
            job: Completed training job.

        Returns:
            One mapping per training step, in step order.

        Raises:
            RuntimeError: If the job has not completed.
        """
        if job.job_status != "Completed":
            raise RuntimeError(f"Metrics are recorded for Completed jobs only, not {job.job_status}.")
        steps = sorted((dict(step) for step in job.get_training_metrics()), key=lambda step: step["step"])
        self.store.save(self.METRICS_RECORD, {"job_name": job.job_name, "steps": steps})
        return steps

    @staticmethod
    def summary(job: Any) -> dict[str, Any]:
        """Extracts presentation-safe facts about a training job.

        Args:
            job: Training job.

        Returns:
            Status, timing, output model, billed token usage, and the
            training and agent configuration the job itself recorded.
        """
        start = _as_datetime(job.creation_time)
        end = _as_datetime(job.end_time)
        duration = round((end - start).total_seconds() / 60, 1) if start and end else None
        return {
            "job_name": job.job_name,
            "job_arn": job.job_arn,
            "job_status": job.job_status,
            "secondary_status": job.secondary_status,
            "failure_reason": job.failure_reason,
            "creation_time": str(start) if start else None,
            "end_time": str(end) if end else None,
            "duration_minutes": duration,
            "output_model_package_arn": job.output_model_package_arn,
            "billable_token_usage": job.billable_token_usage,
            "progress_info": job.progress_info,
            "training_config": _plain_mapping(getattr(job, "training_config", None)),
            "agent_config": _plain_mapping(getattr(job, "agent_config", None)),
        }


def _plain_mapping(value: Any) -> dict[str, Any] | None:
    """Returns a copy of a mapping; None for anything else, such as SDK sentinels."""
    return dict(value) if isinstance(value, Mapping) else None


def _as_datetime(value: Any) -> datetime | None:
    """Returns the value when it is a datetime, else None.

    The SageMaker core SDK uses an ``Unassigned`` sentinel for unset fields.

    Args:
        value: Timestamp field from an SDK object.

    Returns:
        The datetime, or None for unset values.
    """
    return value if isinstance(value, datetime) else None

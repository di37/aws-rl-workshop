"""Entry point for the SageMaker MTRL demo: setup, datasets, baseline, stages.

Training, comparison evaluation, deployment, and inference live in their own
modules; this facade wires them to one region-pinned session and evidence store.
"""

from __future__ import annotations

import json
from decimal import Decimal
from typing import Any

import boto3
from sagemaker.core.helper.session_helper import Session
from sagemaker.train.evaluate import MultiTurnRLEvaluator
from sagemaker.train.multi_turn_rl_trainer import MultiTurnRLTrainer

from aws.config import ARTIFACTS_DIR, DATA_DIR, MODEL_ID, DemoConfig, ResourceState
from aws.costs.cost_guard import MtrlCostGuard
from aws.costs.ledger import SpendLedger
from aws.deploy.deployment import DeploymentDependencies, DeploymentStage
from aws.deploy.endpoint_watchdog import watchdog_alive
from aws.guards import require_phrase
from aws.records.evidence import EvidenceStore
from aws.rl.evaluation import EvaluationStage, configure_evaluator, start_quietly
from aws.rl.pipelines import PipelineInspector
from aws.rl.training import TrainingStage
from aws.setup.prompt_datasets import PromptDatasets

_FAILED = ("Failed", "Stopped")
"""Terminal pipeline statuses that may be replaced with the retry phrase."""
_USABLE_MLFLOW = ("Created", "Updated")
"""MLflow app statuses in which the app accepts runs and traces."""


class MtrlDemoWorkflow:
    """Owns validation, datasets, the baseline evaluation, and stage wiring."""

    BASE_EVAL_CONFIRMATION = "CONFIRM_BASE_EVAL_MTRL_DEMO"
    BASE_EVAL_RETRY_CONFIRMATION = "CONFIRM_RETRY_BASE_EVAL_MTRL_DEMO"
    EVIDENCE_FIELDS = {"arn", "name", "s3_output_path", "mlflow_resource_arn", "mlflow_experiment_name"}

    def __init__(
        self,
        config: DemoConfig | None = None,
        state: ResourceState | None = None,
        session: boto3.Session | None = None,
    ) -> None:
        """Initializes the workflow and pins every SDK client to the demo region.

        Args:
            config: Shared demo configuration.
            state: Provisioned resource identifiers.
            session: Optional Boto3 session for dependency injection.
        """
        self.config = config or DemoConfig()
        self.state = state or ResourceState.load()
        self.session = session or boto3.Session(region_name=self.state.region)
        pin_sdk_region(self.session, self.state.region)
        self.sagemaker_session = Session(boto_session=self.session)
        self.s3 = self.session.client("s3")
        self.store = EvidenceStore(ARTIFACTS_DIR)
        self.datasets = PromptDatasets(self.s3, self.state.bucket, ARTIFACTS_DIR)
        self._inspector: PipelineInspector | None = None

    @property
    def training_s3_uri(self) -> str:
        """Returns the uploaded training dataset URI."""
        return f"s3://{self.state.bucket}/datasets/training_prompts.parquet"

    @property
    def evaluation_s3_uri(self) -> str:
        """Returns the uploaded evaluation dataset URI."""
        return f"s3://{self.state.bucket}/datasets/evaluation_prompts.parquet"

    @property
    def inspector(self) -> PipelineInspector:
        """Returns the shared pipeline and MLflow inspector."""
        if self._inspector is None:
            self._inspector = PipelineInspector.create(
                self.session, self.state.region, self.state.mlflow_app_arn
            )
        return self._inspector

    def ledger(self) -> SpendLedger:
        """Returns the billed-spend ledger over recorded evidence."""
        return SpendLedger(self.store, self.inspector)

    def training_stage(self) -> TrainingStage:
        """Builds the run-or-attach training stage, gated by learning signal and budget."""
        return TrainingStage.create(
            self.config, self.state, self.sagemaker_session, self.store,
            self.training_s3_uri, self._before_training,
        )

    def evaluation_stage(self) -> EvaluationStage:
        """Builds the base-versus-fine-tuned comparison stage."""
        return EvaluationStage.create(
            self.config, self.state, self.sagemaker_session, self.store,
            self.evaluation_s3_uri, self.inspector, self.budget_gate,
        )

    def deployment_stage(self, hourly_usd: Decimal | None = None) -> DeploymentStage:
        """Builds the LoRA endpoint stage with account, watchdog, and budget gates.

        Args:
            hourly_usd: Instance price per hour, used to record uptime cost.

        Returns:
            Deployment stage for the before/after inference endpoint.
        """
        deps = DeploymentDependencies(
            builder_factory=self._model_builder,
            sagemaker=self.session.client("sagemaker"),
            quota_reader=self._endpoint_quota,
            verify_account=self.verify_account,
            watchdog_alive=lambda: watchdog_alive(self.store),
            budget_gate=self.budget_gate,
        )
        return DeploymentStage(self.config, self.store, deps, hourly_usd)

    def inference(self) -> Any:
        """Builds the before/after inference comparison for the endpoint.

        Imported lazily: inference needs the agent package, which standalone
        scripts such as the endpoint watchdog do not load.
        """
        from aws.deploy.inference import InferenceComparison

        return InferenceComparison.create(self.config, self.store, self.session, self.budget_gate)

    def verify_account(self) -> dict[str, Any]:
        """Confirms the active credentials belong to the demo account.

        Returns:
            The STS caller identity.

        Raises:
            RuntimeError: If the credentials belong to another account.
        """
        identity = self.session.client("sts").get_caller_identity()
        if identity["Account"] != self.state.account_id:
            raise RuntimeError("Active AWS account does not match resource state.")
        return identity

    def validate(self) -> dict[str, Any]:
        """Validates identity, region, datasets, model support, and the live runtime and MLflow app.

        Returns:
            Validation details safe to display in a notebook.

        Raises:
            RuntimeError: If a required account or runtime condition fails.
            ValueError: If a dataset has fewer than 32 unique prompts.
        """
        identity = self.verify_account()
        if self.state.region != self.config.region:
            raise RuntimeError(f"This demo is pinned to {self.config.region}.")
        live = self.live_resources()
        train_count = self.datasets.validate_csv(DATA_DIR / "training_prompts.csv", minimum=32)
        eval_count = self.datasets.validate_csv(DATA_DIR / "evaluation_prompts.csv", minimum=32)
        supported = MultiTurnRLTrainer.list_supported_models(session=self.session)
        names = {item if isinstance(item, str) else item.get("model_id") for item in supported}
        if MODEL_ID not in names:
            raise RuntimeError(f"{MODEL_ID} is not reported as supported in {self.state.region}.")
        return {
            "account_id": identity["Account"],
            "caller_arn": identity["Arn"],
            "region": self.state.region,
            "model": MODEL_ID,
            "agent_runtime_arn": self.state.agent_runtime_arn,
            "training_prompts": train_count,
            "evaluation_prompts": eval_count,
            "mlflow_app_arn": self.state.mlflow_app_arn,
            **live,
        }

    def live_resources(self) -> dict[str, str]:
        """Asks AWS, not the local state file, whether the runtime and MLflow app are usable.

        Returns:
            Live status of the AgentCore runtime and of the MLflow app.

        Raises:
            RuntimeError: If no runtime is recorded, or either resource is not usable.
        """
        if not self.state.agent_runtime_id:
            raise RuntimeError("No AgentCore runtime is recorded: run scripts/03_deploy_agent.py first.")
        runtime = self.session.client("bedrock-agentcore-control").get_agent_runtime(
            agentRuntimeId=self.state.agent_runtime_id)["status"]
        mlflow = self.session.client("sagemaker").describe_mlflow_app(Arn=self.state.mlflow_app_arn)["Status"]
        if runtime != "READY" or mlflow not in _USABLE_MLFLOW:
            raise RuntimeError(f"AgentCore runtime is {runtime} and MLflow app is {mlflow} in AWS.")
        return {"agent_runtime_status": runtime, "mlflow_app_status": mlflow}

    def upload_datasets(self) -> dict[str, dict[str, Any]]:
        """Uploads prompt files to S3, skipping files whose bytes are unchanged.

        Returns:
            Per split, the S3 URI and whether a new upload happened.
        """
        prefix = f"s3://{self.state.bucket}/"
        return self.datasets.upload({
            "training": (DATA_DIR / "training_prompts.csv", self.training_s3_uri.removeprefix(prefix)),
            "evaluation": (DATA_DIR / "evaluation_prompts.csv", self.evaluation_s3_uri.removeprefix(prefix)),
        })

    def submit_base_evaluation(self, confirmation: str) -> str:
        """Runs a billable base-model evaluation after exact confirmation.

        Args:
            confirmation: Exact base-evaluation confirmation phrase.

        Returns:
            Terminal pipeline status.
        """
        require_phrase(confirmation, self.BASE_EVAL_CONFIRMATION, "base evaluation")
        evaluator = configure_evaluator(
            MultiTurnRLEvaluator(
                model=MODEL_ID,
                dataset=self.evaluation_s3_uri,
                agent_config=self.state.agent_runtime_arn,
                s3_output_path=f"s3://{self.state.bucket}/base-evaluation/",
                mlflow_resource_arn=self.state.mlflow_app_arn,
                role=self.state.job_role_arn,
                sagemaker_session=self.sagemaker_session,
                accept_eula=True,
            ),
            self.config,
        )
        execution = start_quietly(evaluator)
        self._save_execution("base_evaluation.json", execution)
        return self._finish_base_evaluation(execution.arn)

    def run_or_attach_base_evaluation(self, confirmation: str) -> dict[str, Any] | None:
        """Shows the recorded baseline, finishes an interrupted one, or submits one.

        A recorded baseline is never submitted twice: when its metrics are
        missing (for example after an interrupted wait), this waits for the
        recorded pipeline and downloads them.

        Args:
            confirmation: Exact phrase that authorizes a new billable baseline.

        Returns:
            Recorded execution and metrics, or None with no record and no phrase.
        """
        record = self.store.load("base_evaluation.json")
        if record and record.get("status") in _FAILED and confirmation == self.BASE_EVAL_RETRY_CONFIRMATION:
            self.store.archive("base_evaluation.json", record["arn"].rsplit("/", 1)[-1])
            record, confirmation = None, self.BASE_EVAL_CONFIRMATION
        if record is None:
            if not confirmation:
                return None
            self.submit_base_evaluation(confirmation)
        elif self.store.load("base_evaluation_metrics.json") is None:
            self._finish_base_evaluation(record["arn"])
        return self.base_evaluation_evidence()

    def _finish_base_evaluation(self, execution_arn: str) -> str:
        """Waits for a baseline pipeline, records its status, and keeps metrics only on success.

        Args:
            execution_arn: Baseline pipeline execution.

        Returns:
            ``Succeeded``.

        Raises:
            RuntimeError: If the pipeline did not succeed; the record keeps its status.
        """
        status = self.inspector.wait(execution_arn, timeout_seconds=3_600)
        record = self.store.load("base_evaluation.json") or {"arn": execution_arn}
        self.store.save("base_evaluation.json", {**record, "status": status})
        if status != "Succeeded":
            raise RuntimeError(
                f"The baseline pipeline ended {status}; no metrics were recorded. To archive it "
                f"and run a new one, pass --confirm {self.BASE_EVAL_RETRY_CONFIRMATION}."
            )
        self.store.save("base_evaluation_metrics.json", self._fetch_base_metrics())
        return status

    def base_evaluation_evidence(self) -> dict[str, Any]:
        """Loads the recorded baseline evaluation without calling AWS."""
        return {
            "execution": self.store.load("base_evaluation.json"),
            "metrics": self.store.load("base_evaluation_metrics.json"),
        }

    def _model_builder(self, model_package_arn: str) -> Any:
        """Builds the guide's ``ModelBuilder`` for the trained model package.

        Args:
            model_package_arn: Output model package of the training job.

        Returns:
            ``ModelBuilder`` with the EULA accepted, not yet built.
        """
        from sagemaker.core.resources import ModelPackage
        from sagemaker.serve import ModelBuilder

        package = ModelPackage.get(
            model_package_name=model_package_arn, session=self.session, region=self.state.region
        )
        builder = ModelBuilder(
            model=package,
            role_arn=self.state.job_role_arn,
            sagemaker_session=self.sagemaker_session,
            instance_type=self.config.endpoint_instance_type,
        )
        builder.accept_eula = True
        return builder

    def _endpoint_quota(self) -> float:
        """Returns the applied endpoint quota for the demo instance type."""
        quotas = self.session.client("service-quotas")
        return quotas.get_service_quota(
            ServiceCode="sagemaker", QuotaCode=self.config.endpoint_quota_code
        )["Quota"]["Value"]

    def budget_gate(self) -> None:
        """Raises PermissionError when projected spend exceeds the hard cap."""
        ledger = self.ledger()
        MtrlCostGuard.from_price_list(self.config).evaluate(
            ledger.measured_reference(), ledger.spent(), ledger.remaining()
        )

    def _fetch_base_metrics(self) -> dict[str, Any]:
        """Downloads metrics from the completed base-evaluation output."""
        response = self.s3.get_object(
            Bucket=self.state.bucket, Key="base-evaluation/eval-results.json"
        )
        return json.loads(response["Body"].read())

    def _before_training(self) -> None:
        """Blocks training without a learning signal, or when the budget would be exceeded."""
        self._require_learning_signal()
        self.budget_gate()

    def _require_learning_signal(self) -> None:
        """Blocks training unless the base rewards contain useful variance.

        Raises:
            RuntimeError: If metrics are stale or rewards are identical.
        """
        if not self._base_evaluation_is_current():
            raise RuntimeError("Training blocked: rerun evaluation for the current dataset.")
        metrics = self._fetch_base_metrics()
        minimum = metrics.get("eval/reward/min")
        maximum = metrics.get("eval/reward/max")
        if not isinstance(minimum, (int, float)) or not isinstance(maximum, (int, float)):
            raise RuntimeError("Base evaluation reward metrics are unavailable.")
        if minimum >= maximum:
            raise RuntimeError("Training blocked: the base evaluation has no reward variance.")

    def _base_evaluation_is_current(self) -> bool:
        """Checks that evaluation output is newer than the uploaded dataset."""
        dataset = self.s3.head_object(
            Bucket=self.state.bucket, Key="datasets/evaluation_prompts.parquet"
        )
        metrics = self.s3.head_object(
            Bucket=self.state.bucket, Key="base-evaluation/eval-results.json"
        )
        return metrics["LastModified"] >= dataset["LastModified"]

    def _save_execution(self, filename: str, execution: Any) -> None:
        """Saves an allow-listed, redacted record of an SDK execution."""
        self.store.save(filename, EvidenceStore.allowlisted(execution, self.EVIDENCE_FIELDS))


def pin_sdk_region(session: boto3.Session, region: str) -> None:
    """Points the SageMaker SDK's process-wide client singleton at one region.

    The SDK creates ``SageMakerClient`` once per process; whichever caller is
    first fixes its region, and later ``region=`` arguments are ignored. Without
    this, a default profile region (such as ``me-central-1``) can win.

    Args:
        session: Boto3 session pinned to the demo region.
        region: Demo region.
    """
    from sagemaker.core.utils.utils import SageMakerClient, SingletonMeta

    existing = SingletonMeta._instances.get(SageMakerClient)
    if existing is not None and getattr(existing, "region_name", None) == region:
        return
    SageMakerClient.reset()
    SageMakerClient(session=session, region_name=region)

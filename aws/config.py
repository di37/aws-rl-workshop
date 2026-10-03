"""Shared configuration, project paths, and persisted resource state for the MTRL demo."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, ClassVar

MODEL_ID = "openai-reasoning-gpt-oss-20b"
"""SageMaker public hub ID of the GPT-OSS-20B base model used for MTRL."""

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AGENT_DIR = PROJECT_ROOT / "agent"
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
REPORTS_DIR = PROJECT_ROOT / "reports"


@dataclass(frozen=True)
class DemoConfig:
    """Defines stable names and limits for all demo resources."""

    region: str = "us-west-2"
    project: str = "sagemaker-mtrl-demo"
    job_role_name: str = "SageMakerMTRLJobRole"
    agent_role_name: str = "SageMakerMTRLAgentCoreRole"
    ecr_repository: str = "sagemaker-mtrl-support-agent"
    mlflow_app_name: str = "sagemaker-mtrl-demo"
    agent_name: str = "mtrl_support_agent"
    budget_limit_usd: int = 25
    training_steps: int = 10
    training_group_size: int = 4
    training_batch_size: int = 32
    eval_group_size: int = 2
    pass_k_values: tuple[int, ...] = (1, 2)
    sampling_max_tokens: int = 512
    endpoint_name: str = "mtrl-support-demo"
    endpoint_instance_type: str = "ml.g6e.12xlarge"
    endpoint_quota_code: str = "L-60313EA3"
    endpoint_max_minutes: int = 60
    bedrock_base_model_id: str = "openai.gpt-oss-20b-1:0"
    imported_model_name: str = "mtrl-support-gpt-oss-20b-ft"
    inference_ticket_count: int = 6
    inference_seed: int = 2026
    bedrock_import_role_name: str = "BedrockMTRLModelImportRole"

    @property
    def tags(self) -> list[dict[str, str]]:
        """Returns common AWS tags.

        Returns:
            Tags accepted by AWS APIs that use ``Key`` and ``Value`` fields.
        """
        return [
            {"Key": "Project", "Value": self.project},
            {"Key": "ManagedBy", "Value": "aws-rl-workshop"},
        ]


@dataclass
class ResourceState:
    """Stores identifiers for resources created by the demo."""

    account_id: str
    region: str
    bucket: str
    job_role_arn: str
    agent_role_arn: str
    ecr_uri: str
    mlflow_app_arn: str
    agent_runtime_arn: str | None = None
    agent_runtime_id: str | None = None
    agent_runtime_status: str | None = None

    STATE_FILE: ClassVar[Path] = ARTIFACTS_DIR / "resources.json"

    @classmethod
    def load(cls) -> "ResourceState":
        """Loads resource state from disk.

        Returns:
            Parsed resource state.

        Raises:
            FileNotFoundError: If infrastructure has not been provisioned.
        """
        if not cls.STATE_FILE.exists():
            raise FileNotFoundError(
                "No resource state found: run scripts/01_provision.py first."
            )
        return cls(**json.loads(cls.STATE_FILE.read_text()))

    def save(self) -> None:
        """Persists resource identifiers without storing credentials."""
        self.STATE_FILE.write_text(json.dumps(asdict(self), indent=2) + "\n")

    def update_runtime(self, runtime: dict[str, Any]) -> None:
        """Records a deployed AgentCore runtime.

        Args:
            runtime: Response object returned by AgentCore Runtime APIs.
        """
        self.agent_runtime_arn = runtime["agentRuntimeArn"]
        self.agent_runtime_id = runtime["agentRuntimeId"]
        self.agent_runtime_status = runtime["status"]
        self.save()

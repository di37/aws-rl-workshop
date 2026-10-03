"""Print a concise, read-only status report for the MTRL demo resources."""

from __future__ import annotations

import boto3
from botocore.exceptions import ClientError

from aws.config import ResourceState


class ResourceStatusReporter:
    """Collects a read-only status summary for tracked resources."""

    def __init__(self, state: ResourceState) -> None:
        """Initializes the status reporter.

        Args:
            state: Persisted AWS resource identifiers.
        """
        self.state = state

    def build(self) -> dict[str, object]:
        """Builds the resource report.

        Returns:
            JSON-compatible statuses from S3, SageMaker, and AgentCore.
        """
        report: dict[str, object] = {
            "account_id": boto3.client("sts").get_caller_identity()["Account"],
            "region": self.state.region,
        }
        try:
            boto3.client("s3").head_bucket(Bucket=self.state.bucket)
            report["s3"] = {
                "bucket": self.state.bucket,
                "status": "available",
            }
        except ClientError as error:
            report["s3"] = {"status": "error", "error": str(error)}

        report["mlflow"] = boto3.client(
            "sagemaker", region_name=self.state.region
        ).describe_mlflow_app(Arn=self.state.mlflow_app_arn)
        if self.state.agent_runtime_id:
            report["agent_runtime"] = boto3.client(
                "bedrock-agentcore-control", region_name=self.state.region
            ).get_agent_runtime(agentRuntimeId=self.state.agent_runtime_id)
        else:
            report["agent_runtime"] = {"status": "not deployed"}
        return report

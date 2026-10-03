"""Idempotently provision non-training resources for the real MTRL demo."""

from __future__ import annotations

import json
import time
from typing import Any

import boto3
from botocore.exceptions import ClientError

from aws.config import DemoConfig, ResourceState

CONFIG = DemoConfig()
CONFIRMATION = "CONFIRM_PROVISION_MTRL_DEMO"


def trust_policy(*services: str) -> str:
    """Builds a service trust policy.

    Args:
        *services: AWS service principals allowed to assume the role.

    Returns:
        A serialized IAM trust policy.
    """
    return json.dumps(
        {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {"Service": list(services)},
                    "Action": "sts:AssumeRole",
                }
            ],
        }
    )


def ensure_role(
    iam: Any,
    name: str,
    trust: str,
    managed_policy_arns: list[str],
    inline_policy: dict[str, Any] | None = None,
) -> str:
    """Creates or updates one IAM execution role.

    Args:
        iam: Boto3 IAM client.
        name: IAM role name.
        trust: Serialized assume-role policy.
        managed_policy_arns: Managed policies to attach.
        inline_policy: Optional least-privilege supplemental policy.

    Returns:
        IAM role ARN.
    """
    try:
        role = iam.get_role(RoleName=name)["Role"]
        iam.update_assume_role_policy(RoleName=name, PolicyDocument=trust)
    except iam.exceptions.NoSuchEntityException:
        role = iam.create_role(
            RoleName=name,
            AssumeRolePolicyDocument=trust,
            Description=f"Execution role for {CONFIG.project}",
            Tags=CONFIG.tags,
        )["Role"]

    for policy_arn in managed_policy_arns:
        iam.attach_role_policy(RoleName=name, PolicyArn=policy_arn)
    if inline_policy:
        iam.put_role_policy(
            RoleName=name,
            PolicyName=f"{CONFIG.project}-runtime",
            PolicyDocument=json.dumps(inline_policy),
        )
    return role["Arn"]


def ensure_bucket(s3: Any, bucket: str) -> None:
    """Creates and secures the dedicated artifact bucket.

    Args:
        s3: Boto3 S3 client.
        bucket: Globally unique bucket name.
    """
    try:
        s3.head_bucket(Bucket=bucket)
    except ClientError as error:
        if error.response["Error"]["Code"] not in {"403", "404", "NoSuchBucket"}:
            raise
        s3.create_bucket(
            Bucket=bucket,
            CreateBucketConfiguration={"LocationConstraint": CONFIG.region},
        )

    s3.put_public_access_block(
        Bucket=bucket,
        PublicAccessBlockConfiguration={
            "BlockPublicAcls": True,
            "IgnorePublicAcls": True,
            "BlockPublicPolicy": True,
            "RestrictPublicBuckets": True,
        },
    )
    s3.put_bucket_encryption(
        Bucket=bucket,
        ServerSideEncryptionConfiguration={
            "Rules": [
                {
                    "ApplyServerSideEncryptionByDefault": {"SSEAlgorithm": "AES256"},
                    "BucketKeyEnabled": True,
                }
            ]
        },
    )
    s3.put_bucket_tagging(Bucket=bucket, Tagging={"TagSet": CONFIG.tags})


def ensure_ecr(ecr: Any) -> str:
    """Creates the private rollout-agent image repository.

    Args:
        ecr: Boto3 ECR client.

    Returns:
        ECR repository URI.
    """
    try:
        repository = ecr.describe_repositories(repositoryNames=[CONFIG.ecr_repository])[
            "repositories"
        ][0]
    except ecr.exceptions.RepositoryNotFoundException:
        repository = ecr.create_repository(
            repositoryName=CONFIG.ecr_repository,
            imageScanningConfiguration={"scanOnPush": True},
            encryptionConfiguration={"encryptionType": "AES256"},
            tags=CONFIG.tags,
        )["repository"]
    return repository["repositoryUri"]


def ensure_mlflow_app(sagemaker: Any, bucket: str, role_arn: str) -> str:
    """Creates the managed MLflow tracking application.

    Args:
        sagemaker: Boto3 SageMaker client.
        bucket: Artifact bucket name.
        role_arn: SageMaker job role ARN.

    Returns:
        MLflow application ARN after it becomes ready.
    """
    apps = sagemaker.list_mlflow_apps(MaxResults=100).get("Summaries", [])
    match = next((app for app in apps if app["Name"] == CONFIG.mlflow_app_name), None)
    if match:
        arn = match["Arn"]
    else:
        arn = sagemaker.create_mlflow_app(
            Name=CONFIG.mlflow_app_name,
            ArtifactStoreUri=f"s3://{bucket}/mlflow/",
            RoleArn=role_arn,
            ModelRegistrationMode="AutoModelRegistrationDisabled",
            Tags=CONFIG.tags,
        )["Arn"]

    deadline = time.monotonic() + 900
    while time.monotonic() < deadline:
        description = sagemaker.describe_mlflow_app(Arn=arn)
        status = description["Status"]
        if status in {"Created", "Updated"}:
            return arn
        if status in {"CreateFailed", "UpdateFailed", "DeleteFailed"}:
            raise RuntimeError(f"MLflow app entered {status}: {description}")
        time.sleep(10)
    raise TimeoutError("MLflow app did not become ready within 15 minutes.")


def runtime_fields(
    existing: ResourceState | None, account_id: str, region: str
) -> dict[str, str | None]:
    """Keeps a recorded AgentCore runtime only when it belongs to this account and region.

    A state file from another account (for example, a cloned repository) must
    not make a foreign runtime look deployed and READY.

    Args:
        existing: Previously saved state, if any.
        account_id: Account of the active credentials.
        region: Demo region.

    Returns:
        The runtime ARN, ID, and status to keep, or all None.
    """
    if existing is None or existing.account_id != account_id or existing.region != region:
        return {"agent_runtime_arn": None, "agent_runtime_id": None, "agent_runtime_status": None}
    return {
        "agent_runtime_arn": existing.agent_runtime_arn,
        "agent_runtime_id": existing.agent_runtime_id,
        "agent_runtime_status": existing.agent_runtime_status,
    }


def create_budget(account_id: str, email: str | None) -> None:
    """Creates an optional forecast email alert.

    Args:
        account_id: AWS account identifier.
        email: Alert recipient, or ``None`` to skip budget creation.
    """
    if not email:
        print("Budget alert skipped: set MTRL_BUDGET_EMAIL to enable email alerts.")
        return

    budgets = boto3.client("budgets", region_name="us-east-1")
    budget = {
        "BudgetName": f"{CONFIG.project}-{CONFIG.budget_limit_usd}-usd",
        "BudgetLimit": {
            "Amount": str(CONFIG.budget_limit_usd),
            "Unit": "USD",
        },
        "TimeUnit": "MONTHLY",
        "BudgetType": "COST",
        "CostTypes": {
            "IncludeTax": True,
            "IncludeSubscription": True,
            "UseBlended": False,
            "IncludeRefund": False,
            "IncludeCredit": False,
            "IncludeUpfront": True,
            "IncludeRecurring": True,
            "IncludeOtherSubscription": True,
            "IncludeSupport": True,
            "IncludeDiscount": True,
            "UseAmortized": False,
        },
    }
    notification = {
        "Notification": {
            "NotificationType": "FORECASTED",
            "ComparisonOperator": "GREATER_THAN",
            "Threshold": 80,
            "ThresholdType": "PERCENTAGE",
        },
        "Subscribers": [{"SubscriptionType": "EMAIL", "Address": email}],
    }
    try:
        budgets.create_budget(
            AccountId=account_id,
            Budget=budget,
            NotificationsWithSubscribers=[notification],
        )
    except budgets.exceptions.DuplicateRecordException:
        budgets.update_budget(AccountId=account_id, NewBudget=budget)


class InfrastructureProvisioner:
    """Coordinates creation of reusable, non-training AWS resources."""

    def __init__(
        self,
        config: DemoConfig,
        session: boto3.Session | None = None,
    ) -> None:
        """Initializes service clients.

        Args:
            config: Shared demo configuration.
            session: Optional Boto3 session for dependency injection.
        """
        self.config = config
        self.session = session or boto3.Session(region_name=config.region)
        self.iam = self.session.client("iam")
        self.s3 = self.session.client("s3")
        self.ecr = self.session.client("ecr")
        self.sagemaker = self.session.client("sagemaker")

    def provision(self, budget_email: str | None = None) -> ResourceState:
        """Creates infrastructure and persists its identifiers.

        Args:
            budget_email: Optional forecast-alert recipient.

        Returns:
            Persisted resource state.
        """
        account_id = self.session.client("sts").get_caller_identity()["Account"]
        bucket = f"{self.config.project}-{account_id}-{self.config.region}"
        try:
            existing_state = ResourceState.load()
        except FileNotFoundError:
            existing_state = None
        ensure_bucket(self.s3, bucket)
        job_role_arn = ensure_role(
            self.iam,
            self.config.job_role_name,
            trust_policy("job.sagemaker.amazonaws.com", "sagemaker.amazonaws.com"),
            ["arn:aws:iam::aws:policy/AmazonSageMakerJobFullAccess"],
            inline_policy=self._job_runtime_policy(),
        )
        agent_role_arn = ensure_role(
            self.iam,
            self.config.agent_role_name,
            trust_policy("bedrock-agentcore.amazonaws.com"),
            ["arn:aws:iam::aws:policy/AmazonSageMakerJobRuntimeAccess"],
            inline_policy=self._agent_runtime_policy(),
        )
        state = ResourceState(
            account_id=account_id,
            region=self.config.region,
            bucket=bucket,
            job_role_arn=job_role_arn,
            agent_role_arn=agent_role_arn,
            ecr_uri=ensure_ecr(self.ecr),
            mlflow_app_arn=ensure_mlflow_app(self.sagemaker, bucket, job_role_arn),
            **runtime_fields(existing_state, account_id, self.config.region),
        )
        create_budget(account_id, budget_email)
        state.save()
        return state

    @staticmethod
    def _job_runtime_policy() -> dict[str, Any]:
        """Returns SDK-required training role permissions."""
        return {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Action": [
                        "cloudwatch:PutMetricData",
                        "ecr:GetAuthorizationToken",
                        "sagemaker:AddTags",
                        "sagemaker:AddAssociation",
                        "sagemaker:CreateAction",
                        "sagemaker:CreateArtifact",
                        "sagemaker:CreateContext",
                        "sagemaker:CreateJob",
                        "sagemaker:DeleteAction",
                        "sagemaker:DeleteArtifact",
                        "sagemaker:DeleteAssociation",
                        "sagemaker:DescribeAction",
                        "sagemaker:DescribeArtifact",
                        "sagemaker:DescribeContext",
                        "sagemaker:DescribeJob",
                        "sagemaker:ListActions",
                        "sagemaker:ListArtifacts",
                        "sagemaker:ListAssociations",
                        "sagemaker:ListJobs",
                        "sagemaker:StopJob",
                        "sagemaker:UpdateAction",
                        "sagemaker:UpdateArtifact",
                    ],
                    "Resource": "*",
                },
                {
                    "Effect": "Allow",
                    "Action": "iam:PassRole",
                    "Resource": "arn:aws:iam::*:role/SageMakerMTRLJobRole",
                    "Condition": {
                        "StringEquals": {
                            "iam:PassedToService": "job.sagemaker.amazonaws.com"
                        }
                    },
                },
            ],
        }

    @staticmethod
    def _agent_runtime_policy() -> dict[str, Any]:
        """Returns supplemental AgentCore runtime permissions."""
        return {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Action": [
                        "ecr:GetAuthorizationToken",
                        "ecr:BatchCheckLayerAvailability",
                        "ecr:GetDownloadUrlForLayer",
                        "ecr:BatchGetImage",
                        "logs:CreateLogGroup",
                        "logs:CreateLogStream",
                        "logs:DescribeLogStreams",
                        "logs:PutLogEvents",
                        "xray:PutTraceSegments",
                        "xray:PutTelemetryRecords",
                        "xray:GetSamplingRules",
                        "xray:GetSamplingTargets",
                    ],
                    "Resource": "*",
                }
            ],
        }

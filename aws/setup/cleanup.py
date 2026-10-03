"""Explicitly delete AWS resources created for the MTRL demonstration."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import boto3
from botocore.exceptions import ClientError

from aws.config import DemoConfig, ResourceState
from aws.deploy.bedrock_import import IMPORT_REGION, import_bucket_name

CONFIRMATION = "DELETE_MTRL_DEMO"


class InfrastructureCleaner:
    """Deletes the demo's resources: those in the state file plus the Bedrock import."""

    NOT_FOUND_CODES = {
        "ResourceNotFoundException",
        "NoSuchEntity",
        "NoSuchBucket",
        "RepositoryNotFoundException",
        "NotFoundException",
    }

    def __init__(self, config: DemoConfig, state: ResourceState) -> None:
        """Initializes clients for resource deletion.

        Args:
            config: Shared resource configuration.
            state: Identifiers persisted during provisioning.
        """
        self.config = config
        self.state = state
        self.session = boto3.Session(region_name=state.region)
        self.s3 = self.session.client("s3")
        self.iam = self.session.client("iam")
        self.agentcore = self.session.client("bedrock-agentcore-control")
        self.sagemaker = self.session.client("sagemaker")
        self.bedrock_east = self.session.client("bedrock", region_name=IMPORT_REGION)
        self.s3_east = self.session.client("s3", region_name=IMPORT_REGION)

    def cleanup(self) -> None:
        """Deletes endpoint, Bedrock import, runtime, tracking, storage, image, IAM, and budget assets.

        Raises:
            RuntimeError: If the active AWS account does not match demo state.
        """
        identity = self.session.client("sts").get_caller_identity()
        if identity["Account"] != self.state.account_id:
            raise RuntimeError(
                "Cleanup blocked: active AWS account does not match demo state."
            )
        self._delete_endpoint()
        bedrock_errors = self._delete_bedrock_import()
        if self.state.agent_runtime_id:
            self._ignore_not_found(
                self.agentcore.delete_agent_runtime,
                agentRuntimeId=self.state.agent_runtime_id,
            )
            self._wait_for_deletion(
                lambda: self.agentcore.get_agent_runtime(
                    agentRuntimeId=self.state.agent_runtime_id
                ),
                "AgentCore runtime",
            )
        self._ignore_not_found(
            self.sagemaker.delete_mlflow_app,
            Arn=self.state.mlflow_app_arn,
        )
        self._wait_for_deletion(
            lambda: self.sagemaker.describe_mlflow_app(Arn=self.state.mlflow_app_arn),
            "MLflow app",
        )
        self._ignore_not_found(
            self.session.client("ecr").delete_repository,
            repositoryName=self.config.ecr_repository,
            force=True,
        )
        self._empty_bucket(self.s3, self.state.bucket)
        self._ignore_not_found(self.s3.delete_bucket, Bucket=self.state.bucket)
        self._delete_role(self.config.agent_role_name)
        self._delete_role(self.config.job_role_name)
        self._delete_budget()
        if bedrock_errors:
            raise RuntimeError(
                "Cleanup finished except for the Bedrock import; state file kept for a retry: "
                + "; ".join(bedrock_errors)
            )
        ResourceState.STATE_FILE.unlink(missing_ok=True)

    def _delete_endpoint(self) -> None:
        """Deletes the before/after inference endpoint first; it bills hourly."""
        from aws.workflow import MtrlDemoWorkflow

        workflow = MtrlDemoWorkflow(self.config, self.state, self.session)
        workflow.deployment_stage().delete(reason="infrastructure cleanup")

    def _delete_bedrock_import(self) -> list[str]:
        """Deletes demo imported models, the import bucket, and the tagged import role.

        Nothing is deleted while a demo import job is still running. Other
        errors are collected so the rest of the teardown still runs.

        Returns:
            Problems that left Bedrock import resources behind.
        """
        running = [
            job["jobName"]
            for job in self.bedrock_east.list_model_import_jobs(statusEquals="InProgress").get(
                "modelImportJobSummaries", [])
            if job.get("importedModelName", "").startswith(self.config.imported_model_name)
        ]
        if running:
            errors = [f"import jobs still running: {', '.join(running)}"]
        else:
            bucket = import_bucket_name(self.config, self.state.account_id)
            errors = [
                message
                for message in (self._attempt(label, step) for label, step in (
                    ("imported models", self._delete_imported_models),
                    ("import bucket", lambda: self._delete_import_bucket(bucket)),
                    ("import role", self._delete_import_role),
                ))
                if message
            ]
        for error in errors:
            print("Bedrock cleanup problem:", error)
        return errors

    @staticmethod
    def _attempt(label: str, step: Callable[[], None]) -> str | None:
        """Runs one cleanup step and reports a failure instead of raising.

        Args:
            label: What the step deletes.
            step: Deletion to run.

        Returns:
            The error message, or None on success.
        """
        try:
            step()
        except Exception as error:  # noqa: BLE001 - collected and reported to the caller
            return f"{label}: {type(error).__name__}: {error}"
        return None

    def _delete_imported_models(self) -> None:
        """Deletes every imported model created under the demo's model name."""
        models = self.bedrock_east.list_imported_models(
            nameContains=self.config.imported_model_name
        ).get("modelSummaries", [])
        for model in models:
            self._ignore_not_found(
                self.bedrock_east.delete_imported_model, modelIdentifier=model["modelName"]
            )

    def _delete_import_bucket(self, bucket: str) -> None:
        """Empties and deletes the import bucket, only if this account owns it.

        Args:
            bucket: Import bucket name.
        """
        owner = self.state.account_id
        self._empty_bucket(self.s3_east, bucket, expected_owner=owner)
        self._ignore_not_found(self.s3_east.delete_bucket, Bucket=bucket, ExpectedBucketOwner=owner)

    def _delete_import_role(self) -> None:
        """Deletes the import role only when it carries the demo's project tag."""
        name = self.config.bedrock_import_role_name
        try:
            tags = self.iam.list_role_tags(RoleName=name)["Tags"]
        except ClientError as error:
            if error.response["Error"]["Code"] in self.NOT_FOUND_CODES:
                return
            raise
        if {"Key": "Project", "Value": self.config.project} not in tags:
            print(f"Keeping role {name}: it is not tagged as a demo resource.")
            return
        self._delete_role(name)

    def _empty_bucket(self, s3: Any, bucket: str, expected_owner: str | None = None) -> None:
        """Deletes current objects, versions, and delete markers.

        Args:
            s3: S3 client for the bucket's region.
            bucket: Bucket to empty.
            expected_owner: Account that must own the bucket, when checked.
        """
        owner = {"ExpectedBucketOwner": expected_owner} if expected_owner else {}
        paginator = s3.get_paginator("list_object_versions")
        try:
            for page in paginator.paginate(Bucket=bucket, **owner):
                objects = [
                    {"Key": item["Key"], "VersionId": item["VersionId"]}
                    for key in ("Versions", "DeleteMarkers")
                    for item in page.get(key, [])
                ]
                if objects:
                    response = s3.delete_objects(
                        Bucket=bucket,
                        Delete={"Objects": objects},
                        **owner,
                    )
                    if response.get("Errors"):
                        raise RuntimeError(
                            f"Failed to delete S3 objects: {response['Errors']}"
                        )
        except ClientError as error:
            if error.response["Error"]["Code"] not in self.NOT_FOUND_CODES:
                raise

    def _delete_role(self, role_name: str) -> None:
        """Detaches policies and deletes one IAM role.

        Args:
            role_name: Name of a role owned by this demo.
        """
        try:
            attached = self.iam.list_attached_role_policies(RoleName=role_name).get(
                "AttachedPolicies", []
            )
            for policy in attached:
                self.iam.detach_role_policy(
                    RoleName=role_name,
                    PolicyArn=policy["PolicyArn"],
                )
            inline_names = self.iam.list_role_policies(RoleName=role_name).get(
                "PolicyNames", []
            )
            for policy_name in inline_names:
                self.iam.delete_role_policy(
                    RoleName=role_name,
                    PolicyName=policy_name,
                )
            self.iam.delete_role(RoleName=role_name)
        except self.iam.exceptions.NoSuchEntityException:
            return

    def _delete_budget(self) -> None:
        """Deletes the optional $25 budget if it exists."""
        self._ignore_not_found(
            boto3.client("budgets", region_name="us-east-1").delete_budget,
            AccountId=self.state.account_id,
            BudgetName=(f"{self.config.project}-{self.config.budget_limit_usd}-usd"),
        )

    def _ignore_not_found(
        self,
        operation: Callable[..., Any],
        **kwargs: object,
    ) -> None:
        """Runs a delete operation idempotently.

        Args:
            operation: Boto3 delete method.
            **kwargs: Arguments forwarded to the method.

        Raises:
            ClientError: If deletion fails for a reason other than absence.
        """
        try:
            operation(**kwargs)
        except ClientError as error:
            if error.response["Error"]["Code"] not in self.NOT_FOUND_CODES:
                raise

    def _wait_for_deletion(
        self,
        describe: Callable[[], Any],
        resource_name: str,
    ) -> None:
        """Waits until an asynchronously deleted resource is absent.

        Args:
            describe: Callback that reads the resource.
            resource_name: Name used in timeout diagnostics.

        Raises:
            TimeoutError: If deletion takes longer than ten minutes.
            ClientError: If the read fails for a reason other than absence.
        """
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            try:
                describe()
            except ClientError as error:
                if error.response["Error"]["Code"] in self.NOT_FOUND_CODES:
                    return
                raise
            time.sleep(10)
        raise TimeoutError(f"{resource_name} was not deleted within ten minutes.")

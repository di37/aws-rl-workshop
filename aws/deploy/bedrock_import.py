"""Deployment option 2 from the guide: import the fine-tuned model into Bedrock.

Amazon Bedrock Custom Model Import serves the merged fine-tuned GPT-OSS-20B
serverlessly: no GPU instance, no endpoint quota, billed per active minute and
scaled to zero when idle. GPT-OSS imports run only in us-east-1, so the merged
weights are copied there first. Two files are fixed in the copy, as the Bedrock
docs require for GPT-OSS: the chat template goes into ``tokenizer_config.json``
and ``generation_config.json`` gets all three end-of-sequence token IDs.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any

from botocore.exceptions import ClientError

from aws.config import DemoConfig, ResourceState
from aws.guards import require_phrase
from aws.records.evidence import EvidenceStore

IMPORT_REGION = "us-east-1"
"""Bedrock imports GPT-OSS architectures only in US East (N. Virginia)."""
GENERATION_EOS_TOKEN_IDS = [200002, 199999, 200012]
"""End-of-sequence IDs of the official gpt-oss-20b generation_config.json."""
_SKIPPED_FILES = {"inference.py", "__model_info__.json", "__script_info__.json", "version"}
_TERMINAL = {"Completed", "Failed"}


def import_bucket_name(config: DemoConfig, account_id: str) -> str:
    """Returns the us-east-1 bucket that holds files for Bedrock import.

    Args:
        config: Shared settings (project name).
        account_id: Demo AWS account.

    Returns:
        Bucket name, unique to this account.
    """
    return f"{config.project}-{account_id}-{IMPORT_REGION}"


def destination_prefix(model_package_arn: str) -> str:
    """Returns the import folder for one model package.

    One folder per package means a later training run can never be mistaken
    for an already-copied one, even though its shard sizes are identical.

    Args:
        model_package_arn: ARN ending in ``model-package/<group>/<version>``.

    Returns:
        S3 key prefix such as ``imported-models/<group>-v<version>/``.
    """
    group, version = model_package_arn.split(":model-package/", 1)[1].rsplit("/", 1)
    return f"imported-models/{group}-v{version}/"


def bedrock_runtime(boto_session: Any) -> Any:
    """Builds the Bedrock runtime client used for imported models.

    Retries cover the cold start (``ModelNotReadyException``) after idle.

    Args:
        boto_session: Boto3 session.

    Returns:
        ``bedrock-runtime`` client in the import region.
    """
    from botocore.config import Config

    return boto_session.client(
        "bedrock-runtime", region_name=IMPORT_REGION,
        config=Config(retries={"total_max_attempts": 10, "mode": "standard"}, read_timeout=300),
    )


def fixed_tokenizer_config(tokenizer_config: dict[str, Any], chat_template: str) -> dict[str, Any]:
    """Returns a tokenizer config with the chat template embedded.

    Args:
        tokenizer_config: Original ``tokenizer_config.json`` content.
        chat_template: Jinja chat template of the model.

    Returns:
        A new dict; the input is not modified.
    """
    return {**tokenizer_config, "chat_template": chat_template}


def fixed_generation_config(generation_config: dict[str, Any]) -> dict[str, Any]:
    """Returns a generation config with all three GPT-OSS end tokens.

    Args:
        generation_config: Original ``generation_config.json`` content.

    Returns:
        A new dict; the input is not modified.
    """
    return {**generation_config, "eos_token_id": list(GENERATION_EOS_TOKEN_IDS)}


def files_to_copy(names: Iterable[str]) -> list[str]:
    """Selects the Hugging Face model files, skipping service metadata.

    Args:
        names: File names in the merged checkpoint folder.

    Returns:
        Names to import, in their original order.
    """
    return [name for name in names if name not in _SKIPPED_FILES]


def import_role_trust(account_id: str, region: str) -> dict[str, Any]:
    """Builds the trust policy letting only this account's import jobs assume the role.

    Args:
        account_id: Demo AWS account.
        region: Region of the import jobs.

    Returns:
        IAM trust policy document.
    """
    return {
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": "bedrock.amazonaws.com"},
            "Action": "sts:AssumeRole",
            "Condition": {
                "StringEquals": {"aws:SourceAccount": account_id},
                "ArnLike": {"aws:SourceArn": f"arn:aws:bedrock:{region}:{account_id}:model-import-job/*"},
            },
        }],
    }


def import_role_policy(bucket: str) -> dict[str, Any]:
    """Builds the permission policy: read-only access to the model bucket.

    Args:
        bucket: Bucket holding the files to import.

    Returns:
        IAM policy document.
    """
    return {
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Action": ["s3:GetObject", "s3:ListBucket"],
            "Resource": [f"arn:aws:s3:::{bucket}", f"arn:aws:s3:::{bucket}/*"],
        }],
    }


class BedrockInvokeClient:
    """Lets Strands' OpenAI-format model provider call a Bedrock imported model.

    Strands' SageMaker provider already builds OpenAI chat requests with tools
    and parses OpenAI chat responses; Bedrock imported GPT-OSS models accept
    exactly that schema through ``InvokeModel``. Only the transport differs.
    """

    def __init__(self, runtime: Any, model_arn: str, exceptions: Any) -> None:
        """Initializes the adapter.

        Args:
            runtime: Boto3 ``bedrock-runtime`` client.
            model_arn: Imported model ARN.
            exceptions: Exception namespace the provider's error handling expects.
        """
        self._runtime = runtime
        self._model_arn = model_arn
        self.exceptions = exceptions

    def invoke_endpoint(self, **request: Any) -> dict[str, Any]:
        """Sends the provider's request body to the imported model.

        Args:
            **request: SageMaker-style request built by the provider.

        Returns:
            Response with the OpenAI-format JSON body under ``Body``.
        """
        response = self._runtime.invoke_model(
            modelId=self._model_arn,
            body=request["Body"],
            contentType="application/json",
            accept="application/json",
        )
        return {"Body": response["body"]}

    def invoke_endpoint_with_response_stream(self, **request: Any) -> dict[str, Any]:
        """Streaming is not used; the provider is configured with ``stream=False``.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError("Use payload_config stream=False with imported models.")


def merged_weights_uri(sagemaker_client: Any, model_package_arn: str) -> str:
    """Finds the merged fine-tuned weights folder of an MTRL model package.

    Args:
        sagemaker_client: Boto3 SageMaker client.
        model_package_arn: Output model package of the training job.

    Returns:
        S3 URI of ``checkpoints/hf_merged/`` inside the package's model data.
    """
    package = sagemaker_client.describe_model_package(ModelPackageName=model_package_arn)
    container = package["InferenceSpecification"]["Containers"][0]
    model_data = container["ModelDataSource"]["S3DataSource"]["S3Uri"]
    return model_data.rstrip("/") + "/checkpoints/hf_merged/"


def chat_with_imported_model(runtime: Any, model_arn: str, message: str, max_tokens: int) -> str:
    """Sends one plain chat message to an imported model (OpenAI chat schema).

    Args:
        runtime: Boto3 ``bedrock-runtime`` client in the import region.
        model_arn: Imported model ARN.
        message: User message.
        max_tokens: Generation limit.

    Returns:
        The reply text.
    """
    body = json.dumps({"messages": [{"role": "user", "content": message}], "max_tokens": max_tokens})
    response = runtime.invoke_model(
        modelId=model_arn, body=body, contentType="application/json", accept="application/json"
    )
    payload = json.loads(response["body"].read())
    return payload["choices"][0]["message"].get("content") or ""


def imported_model(config: DemoConfig, model_arn: str, boto_session: Any) -> Any:
    """Builds a Strands model that calls the imported fine-tuned model.

    Args:
        config: Shared settings (token limit).
        model_arn: Imported model ARN in us-east-1.
        boto_session: Boto3 session.

    Returns:
        Strands model usable with ``SupportRolloutAgent.run_episode``.
    """
    from strands.models.sagemaker import SageMakerAIModel

    model = SageMakerAIModel(
        endpoint_config={"endpoint_name": "bedrock-imported-model", "region_name": IMPORT_REGION},
        payload_config={"max_tokens": config.sampling_max_tokens, "stream": False},
        boto_session=boto_session,
    )
    model.client = BedrockInvokeClient(bedrock_runtime(boto_session), model_arn, model.client.exceptions)
    return model


class BedrockImporter:
    """Copies, fixes, and imports the fine-tuned model into Bedrock."""

    RECORD = "bedrock_import.json"
    CONFIRMATION = "CONFIRM_BEDROCK_IMPORT_MTRL_DEMO"

    def __init__(
        self,
        config: DemoConfig,
        state: ResourceState,
        store: EvidenceStore,
        session: Any,
        budget_gate: Any = None,
    ) -> None:
        """Initializes the importer.

        Args:
            config: Shared settings, including the imported model and role names.
            state: Provisioned resource identifiers (account, source bucket).
            store: Evidence store for the import record.
            session: Boto3 session with credentials for the demo account.
            budget_gate: Raises when projected spend exceeds the hard cap.
        """
        self.config = config
        self.state = state
        self.store = store
        self.session = session
        self.budget_gate = budget_gate or (lambda: None)
        self.bucket = import_bucket_name(config, state.account_id)
        self._s3_source = session.client("s3", region_name=state.region)
        self._s3 = session.client("s3", region_name=IMPORT_REGION)
        self._bedrock = session.client("bedrock", region_name=IMPORT_REGION)
        self._iam = session.client("iam")

    def destination_uri(self, model_package_arn: str) -> str:
        """Returns the S3 folder the import job reads for one model package.

        Args:
            model_package_arn: Output model package of the training job.

        Returns:
            Folder URI in the import bucket.
        """
        return f"s3://{self.bucket}/{destination_prefix(model_package_arn)}"

    def run(self, confirmation: str, model_package_arn: str, source_uri: str) -> dict[str, Any]:
        """Copies, fixes, and imports a model package, after phrase and budget checks.

        Args:
            confirmation: Exact phrase that authorizes the billable import.
            model_package_arn: Output model package of the training job.
            source_uri: S3 folder of its merged checkpoint.

        Returns:
            The import record (an existing reusable one, or a new one).
        """
        existing = self._reusable_record(model_package_arn)
        if existing:
            return existing
        self._authorize(confirmation)
        self.prepare_artifacts(source_uri, model_package_arn)
        return self._create_job(self.ensure_role(), model_package_arn)

    def prepare_artifacts(self, source_uri: str, model_package_arn: str) -> str:
        """Copies the merged checkpoint to us-east-1 with the two documented fixes.

        Files already copied for the same package (same size) are skipped, so
        this is safe to rerun; another package always gets its own folder.

        Args:
            source_uri: S3 folder of the merged checkpoint (``.../hf_merged/``).
            model_package_arn: Output model package of the training job.

        Returns:
            The destination S3 folder.
        """
        source_bucket, source_prefix = source_uri.removeprefix("s3://").split("/", 1)
        prefix = destination_prefix(model_package_arn)
        self._ensure_bucket()
        objects = {
            item["Key"].removeprefix(source_prefix): item["Size"]
            for page in self._s3_source.get_paginator("list_objects_v2").paginate(
                Bucket=source_bucket, Prefix=source_prefix)
            for item in page.get("Contents", [])
        }
        names = [n for n in files_to_copy(objects) if n not in ("tokenizer_config.json", "generation_config.json")]
        with ThreadPoolExecutor(max_workers=4) as pool:
            for name in pool.map(
                lambda n: self._copy(source_bucket, source_prefix + n, prefix + n, objects[n]), names
            ):
                print(f"  copied {name}", flush=True)

        def read(name: str) -> bytes:
            return self._s3_source.get_object(Bucket=source_bucket, Key=source_prefix + name)["Body"].read()

        tokenizer = fixed_tokenizer_config(json.loads(read("tokenizer_config.json")),
                                           read("chat_template.jinja").decode("utf-8"))
        generation = fixed_generation_config(json.loads(read("generation_config.json")))
        for name, content in (("tokenizer_config.json", tokenizer), ("generation_config.json", generation)):
            self._s3.put_object(Bucket=self.bucket, Key=prefix + name,
                                Body=json.dumps(content, indent=2).encode("utf-8"))
            print(f"  wrote fixed {name}", flush=True)
        return self.destination_uri(model_package_arn)

    def ensure_role(self) -> str:
        """Creates or updates the least-privilege import role.

        Returns:
            Role ARN.
        """
        name = self.config.bedrock_import_role_name
        trust = json.dumps(import_role_trust(self.state.account_id, IMPORT_REGION))
        tags = [{"Key": t["Key"], "Value": t["Value"]} for t in self.config.tags]
        try:
            arn = self._iam.create_role(RoleName=name, AssumeRolePolicyDocument=trust, Tags=tags,
                                        Description="Reads the MTRL demo model for Bedrock import")["Role"]["Arn"]
        except ClientError as error:
            if error.response["Error"]["Code"] != "EntityAlreadyExists":
                raise
            arn = self._iam.get_role(RoleName=name)["Role"]["Arn"]
            self._iam.update_assume_role_policy(RoleName=name, PolicyDocument=trust)
            self._iam.tag_role(RoleName=name, Tags=tags)
        self._iam.put_role_policy(RoleName=name, PolicyName="read-imported-model-files",
                                  PolicyDocument=json.dumps(import_role_policy(self.bucket)))
        return arn

    def start(self, confirmation: str, role_arn: str, model_package_arn: str) -> dict[str, Any]:
        """Starts the import job for already-copied files, or reuses the record.

        Args:
            confirmation: Exact phrase that authorizes the billable import.
            role_arn: Import role ARN.
            model_package_arn: Output model package being imported.

        Returns:
            The import record.
        """
        existing = self._reusable_record(model_package_arn)
        if existing:
            return existing
        self._authorize(confirmation)
        return self._create_job(role_arn, model_package_arn)

    def _reusable_record(self, model_package_arn: str) -> dict[str, Any] | None:
        """Returns the recorded import when it matches this package and did not fail.

        Args:
            model_package_arn: Output model package being imported.

        Returns:
            The reusable record, or None when nothing is recorded.

        Raises:
            RuntimeError: If a failed import, or one for another package, is recorded.
        """
        record = self.store.load(self.RECORD)
        if not record:
            return None
        if record.get("model_package_arn") == model_package_arn and record.get("status") != "Failed":
            return record
        raise RuntimeError(
            "A failed import or one for another model package is recorded. "
            "To import again, move artifacts/bedrock_import.json aside (keep it as evidence)."
        )

    def _authorize(self, confirmation: str) -> None:
        """Checks the confirmation phrase and the budget before billable work.

        Args:
            confirmation: Phrase supplied by the user.
        """
        require_phrase(confirmation, self.CONFIRMATION, "Bedrock import")
        self.budget_gate()

    def _create_job(self, role_arn: str, model_package_arn: str) -> dict[str, Any]:
        """Creates the import job and records it.

        Args:
            role_arn: Import role ARN.
            model_package_arn: Output model package being imported.

        Returns:
            The new import record.
        """
        job_name = f"mtrl-support-import-{datetime.now(timezone.utc):%Y%m%d%H%M%S}"
        source = self.destination_uri(model_package_arn)
        response = self._bedrock.create_model_import_job(
            jobName=job_name,
            importedModelName=self.config.imported_model_name,
            roleArn=role_arn,
            modelDataSource={"s3DataSource": {"s3Uri": source}},
        )
        record = {"job_arn": response["jobArn"], "job_name": job_name,
                  "imported_model_name": self.config.imported_model_name,
                  "model_package_arn": model_package_arn, "source_uri": source,
                  "started_at": datetime.now(timezone.utc).isoformat()}
        self.store.save(self.RECORD, record)
        return record

    def wait(self, timeout_seconds: int, poll_seconds: int = 30) -> dict[str, Any]:
        """Polls the recorded import job until it completes or fails.

        Args:
            timeout_seconds: Maximum time to wait.
            poll_seconds: Seconds between polls.

        Returns:
            The updated import record.

        Raises:
            TimeoutError: If the job is still running at the deadline.
        """
        record = self.store.load(self.RECORD) or {}
        deadline, last = time.monotonic() + timeout_seconds, None
        while True:
            job = self._bedrock.get_model_import_job(jobIdentifier=record["job_arn"])
            if job["status"] != last:
                last = job["status"]
                print(f"  {datetime.now():%H:%M:%S} import job: {last}", flush=True)
            if last in _TERMINAL:
                break
            if time.monotonic() >= deadline:
                raise TimeoutError("The model import job is still running.")
            time.sleep(poll_seconds)
        record = {**record, "status": last, "failure_message": job.get("failureMessage"),
                  "imported_model_arn": job.get("importedModelArn"),
                  "ended_at": str(job.get("endTime") or "")}
        self.store.save(self.RECORD, record)
        return record

    def describe(self) -> dict[str, Any]:
        """Describes the imported model.

        Returns:
            Model ARN, architecture, and creation time.
        """
        model = self._bedrock.get_imported_model(modelIdentifier=self.config.imported_model_name)
        return {key: model.get(key) for key in (
            "modelArn", "modelName", "modelArchitecture", "creationTime", "instructSupported",
            "customModelUnits")}

    def _ensure_bucket(self) -> None:
        """Creates the private us-east-1 bucket if it does not exist."""
        try:
            self._s3.head_bucket(Bucket=self.bucket)
            return
        except ClientError as error:
            if error.response["Error"]["Code"] not in ("404", "NoSuchBucket", "NotFound"):
                raise
        self._s3.create_bucket(Bucket=self.bucket)
        self._s3.put_public_access_block(Bucket=self.bucket, PublicAccessBlockConfiguration={
            "BlockPublicAcls": True, "IgnorePublicAcls": True,
            "BlockPublicPolicy": True, "RestrictPublicBuckets": True})
        self._s3.put_bucket_tagging(Bucket=self.bucket, Tagging={"TagSet": [
            {"Key": t["Key"], "Value": t["Value"]} for t in self.config.tags]})

    def _copy(self, source_bucket: str, source_key: str, key: str, size: int) -> str:
        """Copies one file across regions unless an identical-size copy exists.

        Args:
            source_bucket: Bucket of the training output.
            source_key: Key of the file.
            key: Destination key inside the package's import folder.
            size: Source size in bytes.

        Returns:
            The file name.
        """
        from boto3.s3.transfer import TransferConfig

        name = key.rsplit("/", 1)[-1]
        try:
            if self._s3.head_object(Bucket=self.bucket, Key=key)["ContentLength"] == size:
                return name
        except ClientError as error:
            if error.response["Error"]["Code"] not in ("404", "NoSuchKey", "NotFound"):
                raise
        self._s3.copy({"Bucket": source_bucket, "Key": source_key}, self.bucket, key,
                      SourceClient=self._s3_source,
                      Config=TransferConfig(multipart_chunksize=256 * 1024 * 1024, max_concurrency=8))
        return name

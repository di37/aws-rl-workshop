"""Guarded deployment of the trained model, following the official guide.

Guide flow: ``ModelPackage.get(...)`` → ``ModelBuilder(model=package).build()``
→ ``deploy(...)``. Depending on the package, the SDK creates either one
inference component serving the fine-tuned model (LoRA merged into the
weights), or a base-model component plus a LoRA adapter component on top.

Before anything is created, deploy() requires: the exact phrase, the expected
AWS account, a live watchdog (``endpoint_watchdog.py``), the budget gate, and
endpoint quota. Any failure or interruption during deployment deletes the
endpoint again. delete() retries until the endpoint is confirmed gone.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from botocore.exceptions import ClientError

from aws.config import DemoConfig
from aws.guards import require_phrase
from aws.records.evidence import EvidenceStore

BASE_ROLE = "base model (before training)"
ADAPTER_ROLE = "LoRA adapter (after training)"
MODEL_ROLE = "fine-tuned model (after training)"
_DELETE_ORDER = (ADAPTER_ROLE, BASE_ROLE, MODEL_ROLE)


@dataclass(frozen=True)
class DeploymentDependencies:
    """AWS entry points and safety gates used by :class:`DeploymentStage`.

    Attributes:
        builder_factory: Returns a ``ModelBuilder`` for a model package ARN.
        sagemaker: Boto3 SageMaker client pinned to the demo region.
        quota_reader: Returns the applied endpoint-instance quota.
        verify_account: Raises when credentials belong to another account.
        watchdog_alive: True when the endpoint watchdog heartbeat is fresh.
        budget_gate: Raises when the projected spend exceeds the hard cap.
        clock: Epoch-seconds clock.
        sleep: Sleep function.
        poll_seconds: Seconds between status polls.
    """

    builder_factory: Callable[[str], Any]
    sagemaker: Any
    quota_reader: Callable[[], float]
    verify_account: Callable[[], None]
    watchdog_alive: Callable[[], bool]
    budget_gate: Callable[[], None]
    clock: Callable[[], float] = time.time
    sleep: Callable[[float], None] = time.sleep
    poll_seconds: int = 30


class DeploymentStage:
    """Deploys, describes, and deletes the before/after inference endpoint."""

    RECORD = "endpoint.json"
    CONFIRMATION = "CONFIRM_DEPLOY_MTRL_DEMO"
    REDEPLOY_CONFIRMATION = "CONFIRM_REDEPLOY_MTRL_DEMO"

    def __init__(
        self,
        config: DemoConfig,
        store: EvidenceStore,
        deps: DeploymentDependencies,
        hourly_usd: Decimal | None = None,
    ) -> None:
        """Initializes the stage.

        Args:
            config: Shared settings: endpoint name, instance type, lifetime.
            store: Evidence store holding the endpoint record.
            deps: AWS entry points and safety gates.
            hourly_usd: Instance price per hour, used to record uptime cost.
        """
        self.config = config
        self.store = store
        self.deps = deps
        self.hourly_usd = hourly_usd

    def deploy(self, confirmation: str, model_package_arn: str | None) -> dict[str, Any] | None:
        """Deploys the trained adapter, or reuses or replays a recorded endpoint.

        Args:
            confirmation: Exact phrase that authorizes a billable endpoint.
            model_package_arn: Output model package of the training job.

        Returns:
            Endpoint description, the record of a deleted endpoint, or None in
            replay mode with no record.

        Raises:
            PermissionError: If the phrase is wrong or the budget gate refuses.
            RuntimeError: If a safety precondition is not met.
        """
        record = self.store.load(self.RECORD)
        expected = self.CONFIRMATION
        if record:
            if record.get("status") != "Deleted":
                return self.describe()
            if not record.get("failure_reason") or confirmation != self.REDEPLOY_CONFIRMATION:
                return record
            self.store.save(f"endpoint.{int(record.get('created_epoch', 0))}.json", record)
            expected = self.REDEPLOY_CONFIRMATION
        elif not confirmation:
            return None
        require_phrase(confirmation, expected, "deployment")
        self._check_preconditions(model_package_arn)
        builder = self.deps.builder_factory(model_package_arn)
        builder.build()
        model_name = getattr(getattr(builder, "built_model", None), "model_name", None)
        self._start_record(model_package_arn, model_name if isinstance(model_name, str) else None)
        try:
            builder.deploy(
                endpoint_name=self.config.endpoint_name,
                instance_type=self.config.endpoint_instance_type,
                initial_instance_count=1,
            )
            self._update_record(in_service_epoch=self.deps.clock())
            self._wait_until_components_ready()
        except BaseException as error:
            self.delete(reason=f"{type(error).__name__}: {error}")
            raise
        description = self.describe()
        record = self.store.load(self.RECORD) or {}
        self.store.save(
            self.RECORD,
            {**record, "status": "InService", "ready_at": _iso(self.deps.clock()),
             "components": description["components"]},
        )
        return description

    def describe(self) -> dict[str, Any]:
        """Describes the endpoint and its inference components.

        Returns:
            Endpoint name, status, instance type, and component roles.
        """
        try:
            endpoint = self.deps.sagemaker.describe_endpoint(EndpointName=self.config.endpoint_name)
            status = endpoint["EndpointStatus"]
        except ClientError as error:
            if not _is_not_found(error):
                raise
            status = "NotFound"
        return {
            "endpoint_name": self.config.endpoint_name,
            "endpoint_status": status,
            "instance_type": self.config.endpoint_instance_type,
            "components": self.components(),
        }

    def components(self) -> list[dict[str, Any]]:
        """Lists inference components with their before/after role.

        A component that references a base is a LoRA adapter; a component an
        adapter references is the base model; any other component serves the
        fine-tuned model on its own.

        Returns:
            One entry per component: name, status, role, base, and model.
        """
        summaries = self.deps.sagemaker.list_inference_components(
            EndpointNameEquals=self.config.endpoint_name
        )["InferenceComponents"]
        details = []
        for summary in summaries:
            try:
                details.append(self.deps.sagemaker.describe_inference_component(
                    InferenceComponentName=summary["InferenceComponentName"]
                ))
            except ClientError as error:
                if not _is_not_found(error):
                    raise
        bases = {d.get("Specification", {}).get("BaseInferenceComponentName") for d in details}
        return [self._describe_component(detail, bases) for detail in details]

    def component_names(self) -> dict[str, str]:
        """Returns component names keyed by what they serve.

        Returns:
            ``{"fine_tuned": ...}`` plus ``"base"`` when a base component exists.
        """
        names = {c["role"]: c["name"] for c in self.components()}
        if ADAPTER_ROLE in names:
            return {"base": names[BASE_ROLE], "fine_tuned": names[ADAPTER_ROLE]}
        return {"fine_tuned": names[MODEL_ROLE]}

    @staticmethod
    def _describe_component(detail: dict[str, Any], bases: set[str | None]) -> dict[str, Any]:
        """Summarizes one inference component.

        Args:
            detail: ``DescribeInferenceComponent`` response.
            bases: Names of components that adapters build on.

        Returns:
            Name, status, role, base component, model name, and failure reason.
        """
        spec = detail.get("Specification", {})
        base = spec.get("BaseInferenceComponentName")
        name = detail["InferenceComponentName"]
        role = ADAPTER_ROLE if base else BASE_ROLE if name in bases else MODEL_ROLE
        return {
            "name": name,
            "status": detail["InferenceComponentStatus"],
            "role": role,
            "base_component": base,
            "model_name": spec.get("ModelName"),
            "failure_reason": detail.get("FailureReason"),
        }

    def delete(self, hourly_usd: Decimal | None = None, reason: str | None = None) -> dict[str, Any]:
        """Deletes components, endpoint, config, and models until none remain.

        Transient errors are retried until the endpoint is confirmed gone.

        Args:
            hourly_usd: Instance price per hour; defaults to the stage's price.
            reason: Failure that triggered the deletion, if any.

        Returns:
            The endpoint record, marked Deleted with uptime and cost.

        Raises:
            TimeoutError: If the endpoint still exists after 30 minutes.
        """
        self.deps.verify_account()
        recorded = self.store.load(self.RECORD)
        record = recorded or {}
        config_names: set[str] = set()
        model_names: set[str] = {record["model_name"]} if record.get("model_name") else set()
        deadline = self.deps.clock() + 30 * 60
        existed = self._endpoint_exists()
        while not self._delete_pass(config_names, model_names):
            if self.deps.clock() >= deadline:
                raise TimeoutError("The endpoint still exists after 30 minutes of deletion.")
            self.deps.sleep(self.deps.poll_seconds)
        for name in sorted(config_names):
            self._attempt(lambda n=name: self.deps.sagemaker.delete_endpoint_config(EndpointConfigName=n))
        for name in sorted(model_names):
            self._attempt(lambda n=name: self.deps.sagemaker.delete_model(ModelName=n))
        if recorded is None:
            return {"endpoint_name": self.config.endpoint_name,
                    "status": "DeletedUnrecorded" if existed else "NotDeployed",
                    "note": "No deployment was recorded, so no endpoint record was written."}
        return self._record_deletion(hourly_usd if hourly_usd is not None else self.hourly_usd, reason)

    def _endpoint_exists(self) -> bool:
        """Reports whether the demo endpoint currently exists.

        Returns:
            False only when SageMaker reports the endpoint as not found.
        """
        try:
            self.deps.sagemaker.describe_endpoint(EndpointName=self.config.endpoint_name)
        except ClientError as error:
            return not _is_not_found(error)
        return True

    def _check_preconditions(self, model_package_arn: str | None) -> None:
        """Runs every safety gate before anything billable is created.

        Args:
            model_package_arn: Output model package of the training job.

        Raises:
            RuntimeError: If a precondition is not met.
            PermissionError: If the budget gate refuses.
        """
        if not model_package_arn:
            raise RuntimeError("Deployment needs the trained model package ARN.")
        self.deps.verify_account()
        if not self.deps.watchdog_alive():
            raise RuntimeError(
                "Start the endpoint watchdog first, in another terminal: python scripts/09a_watchdog.py"
            )
        self.deps.budget_gate()
        quota = self.deps.quota_reader()
        if quota < 1:
            raise RuntimeError(
                f"Endpoint quota for {self.config.endpoint_instance_type} is {quota:g}; "
                "wait for the quota increase before deploying."
            )

    def _start_record(self, model_package_arn: str, model_name: str | None) -> None:
        """Records the deployment and its lifetime deadline before creation.

        Args:
            model_package_arn: Model package being deployed.
            model_name: SageMaker model created by ``build()``, deleted on cleanup.
        """
        started = self.deps.clock()
        self.store.save(
            self.RECORD,
            {
                "endpoint_name": self.config.endpoint_name,
                "instance_type": self.config.endpoint_instance_type,
                "model_package_arn": model_package_arn,
                "model_name": model_name,
                "status": "Creating",
                "created_at": _iso(started),
                "created_epoch": started,
                "deadline_epoch": started + self.config.endpoint_max_minutes * 60,
            },
        )

    def _update_record(self, **fields: Any) -> None:
        """Merges fields into the endpoint record.

        Args:
            **fields: Values to store.
        """
        record = self.store.load(self.RECORD) or {}
        self.store.save(self.RECORD, {**record, **fields})

    def _delete_pass(self, config_names: set[str], model_names: set[str]) -> bool:
        """Issues one round of deletions and reports whether the endpoint is gone.

        Args:
            config_names: Collects endpoint config names seen so far.
            model_names: Collects model names seen so far.

        Returns:
            True when the endpoint no longer exists.
        """
        try:
            endpoint = self.deps.sagemaker.describe_endpoint(EndpointName=self.config.endpoint_name)
            config_names.add(endpoint["EndpointConfigName"])
            self._note_if_billable(endpoint["EndpointStatus"])
            components = self.components()
        except ClientError as error:
            if _is_not_found(error):
                return True
            print(f"Endpoint check failed, retrying: {error}", flush=True)
            return False
        model_names.update(c["model_name"] for c in components if c["model_name"])
        for role in _DELETE_ORDER:
            for component in (c for c in components if c["role"] == role):
                self._attempt(
                    lambda n=component["name"]: self.deps.sagemaker.delete_inference_component(
                        InferenceComponentName=n
                    )
                )
        self._attempt(lambda: self.deps.sagemaker.delete_endpoint(EndpointName=self.config.endpoint_name))
        try:
            self.deps.sagemaker.describe_endpoint(EndpointName=self.config.endpoint_name)
        except ClientError as error:
            return _is_not_found(error)
        return False

    def _note_if_billable(self, endpoint_status: str) -> None:
        """Marks the endpoint as billed once it is seen past provisioning.

        ``builder.deploy()`` waits for InService internally, so a failure inside
        it can happen after an instance already ran. Seeing any status other
        than Creating or Failed means an instance existed; billing is then
        counted conservatively from creation.

        Args:
            endpoint_status: Current ``EndpointStatus``.
        """
        record = self.store.load(self.RECORD) or {}
        if record and endpoint_status not in ("Creating", "Failed") and "in_service_epoch" not in record:
            self._update_record(in_service_epoch=self.deps.clock())

    def _record_deletion(self, hourly_usd: Decimal | None, reason: str | None) -> dict[str, Any]:
        """Marks the record Deleted once, with uptime and cost.

        Args:
            hourly_usd: Instance price per hour, if known.
            reason: Failure that triggered the deletion, if any.

        Returns:
            The endpoint record.
        """
        record = self.store.load(self.RECORD) or {"endpoint_name": self.config.endpoint_name}
        if record.get("status") == "Deleted":
            return record
        now = self.deps.clock()
        uptime = round((now - record.get("created_epoch", now)) / 60, 1)
        update: dict[str, Any] = {"status": "Deleted", "deleted_at": _iso(now), "uptime_minutes": uptime}
        if "in_service_epoch" not in record:
            update["billed_usd"] = 0.0
            update["billing_note"] = "Endpoint never reached InService; no instance was billed."
        elif hourly_usd is not None:
            update["billed_usd"] = round(float(Decimal(hourly_usd) * Decimal(str(uptime)) / 60), 2)
            update["billing_note"] = "Conservative: billed from endpoint creation to deletion."
        if reason:
            update["failure_reason"] = reason
        self.store.save(self.RECORD, {**record, **update})
        return {**record, **update}

    def _wait_until_components_ready(self) -> None:
        """Waits until every component, and any adapter's base, is InService.

        Raises:
            RuntimeError: If a component fails.
            TimeoutError: If components are not ready within 45 minutes.
        """
        deadline = self.deps.clock() + 45 * 60
        while True:
            components = self.components()
            failed = [c for c in components if c["status"] == "Failed"]
            if failed:
                raise RuntimeError(f"Inference component failed: {failed[0]['failure_reason']}")
            ready = components and all(c["status"] == "InService" for c in components)
            roles = {c["role"] for c in components}
            if ready and (ADAPTER_ROLE not in roles or BASE_ROLE in roles):
                return
            if self.deps.clock() >= deadline:
                raise TimeoutError("Inference components were not ready within 45 minutes.")
            self.deps.sleep(self.deps.poll_seconds)

    @staticmethod
    def _attempt(call: Callable[[], Any]) -> None:
        """Runs one delete call; a missing resource counts as success.

        Other errors are reported and left for the next deletion pass.

        Args:
            call: Delete call to run.
        """
        try:
            call()
        except ClientError as error:
            if not _is_not_found(error):
                print(f"Delete call will be retried: {error}", flush=True)


def should_force_delete(record: dict[str, Any] | None, now: float) -> bool:
    """Decides whether the watchdog must delete the endpoint now.

    A live record without a deadline counts as expired (fail closed).

    Args:
        record: Endpoint record, or None when nothing was deployed.
        now: Current epoch seconds.

    Returns:
        True when a live endpoint has passed, or lacks, its deadline.
    """
    if not record or record.get("status") == "Deleted":
        return False
    return now >= record.get("deadline_epoch", float("-inf"))


def _is_not_found(error: ClientError) -> bool:
    """Recognizes SageMaker's missing-resource errors."""
    code = error.response.get("Error", {}).get("Code", "")
    message = error.response.get("Error", {}).get("Message", "").lower()
    return code in {"ResourceNotFound", "ResourceNotFoundException"} or (
        code == "ValidationException" and ("could not find" in message or "not found" in message)
    )


def _iso(epoch: float) -> str:
    """Formats epoch seconds as an ISO-8601 UTC timestamp."""
    return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat()

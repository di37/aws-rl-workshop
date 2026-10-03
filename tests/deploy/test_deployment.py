"""Unit tests for the guarded LoRA endpoint deployment stage."""

from __future__ import annotations

import tempfile
import unittest
from decimal import Decimal
from pathlib import Path
from unittest.mock import MagicMock

from botocore.exceptions import ClientError

from aws.config import DemoConfig
from aws.deploy.deployment import (
    DeploymentDependencies,
    DeploymentStage,
    should_force_delete,
)
from aws.records.evidence import EvidenceStore

PACKAGE = "arn:aws:sagemaker:us-west-2:1:model-package/group/1"
BASE = "mtrl-support-demo-inference-component"
ADAPTER = "mtrl-support-demo-adapter"


def client_error(code: str, message: str, operation: str) -> ClientError:
    """Builds a botocore ClientError."""
    return ClientError({"Error": {"Code": code, "Message": message}}, operation)


def not_found(operation: str) -> ClientError:
    """Builds the error SageMaker returns for a missing resource."""
    return client_error("ValidationException", "Could not find resource", operation)


class FakeSageMaker:
    """In-memory SageMaker control plane with injectable delete failures."""

    def __init__(self) -> None:
        self.endpoint: dict | None = None
        self.components: dict[str, dict] = {}
        self.calls: list[str] = []
        self.failures: dict[str, list[ClientError]] = {}

    def create_merged_endpoint(self) -> None:
        self.endpoint = {"EndpointName": "mtrl-support-demo", "EndpointStatus": "InService",
                         "EndpointConfigName": "mtrl-support-demo"}
        self.components = {
            BASE: {"InferenceComponentStatus": "InService",
                   "Specification": {"ModelName": "merged-model"}},
        }

    def create_live_endpoint(self) -> None:
        self.endpoint = {"EndpointName": "mtrl-support-demo", "EndpointStatus": "InService",
                         "EndpointConfigName": "mtrl-support-demo"}
        self.components = {
            BASE: {"InferenceComponentStatus": "InService",
                   "Specification": {"ModelName": "base-model"}},
            ADAPTER: {"InferenceComponentStatus": "InService",
                      "Specification": {"BaseInferenceComponentName": BASE}},
        }

    def _maybe_fail(self, key: str) -> None:
        queue = self.failures.get(key, [])
        if queue:
            raise queue.pop(0)

    def describe_endpoint(self, EndpointName: str) -> dict:
        if self.endpoint is None:
            raise not_found("DescribeEndpoint")
        return self.endpoint

    def list_inference_components(self, EndpointNameEquals: str) -> dict:
        return {"InferenceComponents": [{"InferenceComponentName": n} for n in self.components]}

    def describe_inference_component(self, InferenceComponentName: str) -> dict:
        if InferenceComponentName not in self.components:
            raise not_found("DescribeInferenceComponent")
        return {"InferenceComponentName": InferenceComponentName,
                **self.components[InferenceComponentName]}

    def delete_inference_component(self, InferenceComponentName: str) -> None:
        self._maybe_fail(f"delete_ic:{InferenceComponentName}")
        if InferenceComponentName == BASE and ADAPTER in self.components:
            raise client_error("ValidationException", "Adapter still attached", "DeleteIC")
        self.calls.append(f"delete_ic:{InferenceComponentName}")
        self.components.pop(InferenceComponentName, None)

    def delete_endpoint(self, EndpointName: str) -> None:
        self._maybe_fail("delete_endpoint")
        if self.components:
            raise client_error("ValidationException", "Delete components first", "DeleteEndpoint")
        if self.endpoint is None:
            raise not_found("DeleteEndpoint")
        self.calls.append("delete_endpoint")
        self.endpoint = None

    def delete_endpoint_config(self, EndpointConfigName: str) -> None:
        self.calls.append(f"delete_config:{EndpointConfigName}")

    def delete_model(self, ModelName: str) -> None:
        self.calls.append(f"delete_model:{ModelName}")


class DeploymentStageTests(unittest.TestCase):
    """Verifies gates, deadlines, reuse, retries, and idempotent cleanup."""

    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.store = EvidenceStore(Path(self.directory.name))
        self.sagemaker = FakeSageMaker()
        self.builder = MagicMock()
        self.builder.deploy.side_effect = lambda **_: self.sagemaker.create_live_endpoint()
        self.factory = MagicMock(return_value=self.builder)
        self.now = [1_000.0]
        self.gates = MagicMock()
        self.gates.watchdog_alive.return_value = True
        self.gates.quota.return_value = 1.0
        self.stage = self._stage()

    def _stage(self) -> DeploymentStage:
        def sleep(seconds: float) -> None:
            self.now[0] += seconds

        deps = DeploymentDependencies(
            builder_factory=self.factory,
            sagemaker=self.sagemaker,
            quota_reader=self.gates.quota,
            verify_account=self.gates.verify_account,
            watchdog_alive=self.gates.watchdog_alive,
            budget_gate=self.gates.budget_gate,
            clock=lambda: self.now[0],
            sleep=sleep,
        )
        return DeploymentStage(DemoConfig(), self.store, deps, hourly_usd=Decimal("12"))

    def tearDown(self) -> None:
        self.directory.cleanup()

    def test_deploy_requires_exact_confirmation(self) -> None:
        with self.assertRaises(PermissionError):
            self.stage.deploy("deploy it", PACKAGE)

        self.factory.assert_not_called()

    def test_replay_without_record_returns_none(self) -> None:
        self.assertIsNone(self.stage.deploy("", PACKAGE))

    def test_dead_watchdog_blocks_deployment(self) -> None:
        self.gates.watchdog_alive.return_value = False

        with self.assertRaisesRegex(RuntimeError, "watchdog"):
            self.stage.deploy(DeploymentStage.CONFIRMATION, PACKAGE)

        self.factory.assert_not_called()

    def test_budget_and_account_gates_run_before_anything_is_built(self) -> None:
        self.gates.budget_gate.side_effect = PermissionError("over the $25 cap")

        with self.assertRaises(PermissionError):
            self.stage.deploy(DeploymentStage.CONFIRMATION, PACKAGE)

        self.gates.verify_account.assert_called()
        self.factory.assert_not_called()

    def test_zero_quota_blocks_before_anything_is_created(self) -> None:
        self.gates.quota.return_value = 0.0

        with self.assertRaisesRegex(RuntimeError, "quota"):
            self.stage.deploy(DeploymentStage.CONFIRMATION, PACKAGE)

        self.factory.assert_not_called()

    def test_deploy_follows_guide_and_records_deadline(self) -> None:
        description = self.stage.deploy(DeploymentStage.CONFIRMATION, PACKAGE)

        self.factory.assert_called_once_with(PACKAGE)
        self.builder.build.assert_called_once_with()
        self.builder.deploy.assert_called_once_with(
            endpoint_name="mtrl-support-demo",
            instance_type="ml.g6e.12xlarge",
            initial_instance_count=1,
        )
        record = self.store.load(DeploymentStage.RECORD)
        self.assertEqual(record["deadline_epoch"], 1_000.0 + 60 * 60)
        self.assertEqual(record["status"], "InService")
        roles = {c["role"] for c in description["components"]}
        self.assertEqual(roles, {"base model (before training)", "LoRA adapter (after training)"})

    def test_failed_deploy_cleans_up_and_reraises(self) -> None:
        def half_deploy(**_: object) -> None:
            self.sagemaker.create_live_endpoint()
            raise RuntimeError("adapter failed to load")

        self.builder.deploy.side_effect = half_deploy

        with self.assertRaisesRegex(RuntimeError, "adapter failed"):
            self.stage.deploy(DeploymentStage.CONFIRMATION, PACKAGE)

        self.assertIsNone(self.sagemaker.endpoint)
        record = self.store.load(DeploymentStage.RECORD)
        self.assertEqual(record["status"], "Deleted")
        self.assertIn("adapter failed", record["failure_reason"])
        self.assertIn("Conservative", record["billing_note"])

    def test_failure_after_instance_ran_is_billed_conservatively(self) -> None:
        def slow_failure(**_: object) -> None:
            self.sagemaker.create_live_endpoint()
            self.now[0] += 20 * 60
            raise RuntimeError("component failed to load")

        self.builder.deploy.side_effect = slow_failure

        with self.assertRaises(RuntimeError):
            self.stage.deploy(DeploymentStage.CONFIRMATION, PACKAGE)

        record = self.store.load(DeploymentStage.RECORD)
        self.assertEqual(record["billed_usd"], 4.0)

    def test_capacity_failure_before_in_service_records_zero_cost(self) -> None:
        self.builder.built_model.model_name = "built-model"
        self.builder.deploy.side_effect = RuntimeError("InsufficientInstanceCapacity")
        self.now[0] += 0

        with self.assertRaises(RuntimeError):
            self.stage.deploy(DeploymentStage.CONFIRMATION, PACKAGE)

        record = self.store.load(DeploymentStage.RECORD)
        self.assertEqual(record["billed_usd"], 0.0)
        self.assertIn("never reached InService", record["billing_note"])
        self.assertIn("delete_model:built-model", self.sagemaker.calls)

    def test_keyboard_interrupt_during_deploy_still_cleans_up(self) -> None:
        def interrupted(**_: object) -> None:
            self.sagemaker.create_live_endpoint()
            raise KeyboardInterrupt

        self.builder.deploy.side_effect = interrupted

        with self.assertRaises(KeyboardInterrupt):
            self.stage.deploy(DeploymentStage.CONFIRMATION, PACKAGE)

        self.assertIsNone(self.sagemaker.endpoint)
        self.assertIn("in_service_epoch", self.store.load(DeploymentStage.RECORD))

    def test_failed_record_needs_redeploy_phrase(self) -> None:
        self.builder.deploy.side_effect = RuntimeError("boom")
        with self.assertRaises(RuntimeError):
            self.stage.deploy(DeploymentStage.CONFIRMATION, PACKAGE)
        self.builder.deploy.side_effect = lambda **_: self.sagemaker.create_live_endpoint()

        replay = self.stage.deploy(DeploymentStage.CONFIRMATION, PACKAGE)
        retried = self.stage.deploy(DeploymentStage.REDEPLOY_CONFIRMATION, PACKAGE)

        self.assertEqual(replay["status"], "Deleted")
        self.assertEqual(retried["endpoint_status"], "InService")
        self.assertEqual(len(self.store.names("endpoint.*.json")), 1)

    def test_existing_endpoint_is_reused(self) -> None:
        self.stage.deploy(DeploymentStage.CONFIRMATION, PACKAGE)

        self.stage.deploy(DeploymentStage.CONFIRMATION, PACKAGE)

        self.builder.deploy.assert_called_once()

    def test_single_component_is_the_fine_tuned_model(self) -> None:
        self.builder.deploy.side_effect = lambda **_: self.sagemaker.create_merged_endpoint()

        description = self.stage.deploy(DeploymentStage.CONFIRMATION, PACKAGE)

        self.assertEqual([c["role"] for c in description["components"]],
                         ["fine-tuned model (after training)"])
        self.assertEqual(self.stage.component_names(), {"fine_tuned": BASE})

    def test_component_names_split_base_and_adapter(self) -> None:
        self.sagemaker.create_live_endpoint()

        self.assertEqual(self.stage.component_names(), {"base": BASE, "fine_tuned": ADAPTER})

    def test_delete_removes_adapter_first_and_records_cost(self) -> None:
        self.stage.deploy(DeploymentStage.CONFIRMATION, PACKAGE)
        self.now[0] += 30 * 60

        record = self.stage.delete()

        self.assertEqual(
            self.sagemaker.calls,
            [f"delete_ic:{ADAPTER}", f"delete_ic:{BASE}", "delete_endpoint",
             "delete_config:mtrl-support-demo", "delete_model:base-model"],
        )
        self.assertEqual(record["status"], "Deleted")
        self.assertEqual(record["uptime_minutes"], 30.0)
        self.assertEqual(record["billed_usd"], 6.0)

    def test_delete_retries_through_transient_errors(self) -> None:
        self.stage.deploy(DeploymentStage.CONFIRMATION, PACKAGE)
        throttle = client_error("ThrottlingException", "Rate exceeded", "DeleteIC")
        self.sagemaker.failures = {
            f"delete_ic:{ADAPTER}": [throttle],
            "delete_endpoint": [throttle],
        }

        record = self.stage.delete()

        self.assertIsNone(self.sagemaker.endpoint)
        self.assertEqual(record["status"], "Deleted")

    def test_second_delete_keeps_the_recorded_cost(self) -> None:
        self.stage.deploy(DeploymentStage.CONFIRMATION, PACKAGE)
        self.now[0] += 30 * 60
        first = self.stage.delete()
        self.now[0] += 24 * 3600

        second = self.stage.delete()

        self.assertEqual(second["billed_usd"], first["billed_usd"])

    def test_deleted_record_is_replayed_not_redeployed(self) -> None:
        self.stage.deploy(DeploymentStage.CONFIRMATION, PACKAGE)
        self.stage.delete()

        record = self.stage.deploy(DeploymentStage.CONFIRMATION, PACKAGE)

        self.assertEqual(record["status"], "Deleted")
        self.builder.deploy.assert_called_once()

    def test_delete_without_a_record_saves_nothing(self) -> None:
        result = self.stage.delete(reason="infrastructure cleanup")

        self.assertEqual(result["status"], "NotDeployed")
        self.assertIsNone(self.store.load(DeploymentStage.RECORD))

    def test_orphaned_endpoint_is_deleted_without_inventing_a_record(self) -> None:
        self.sagemaker.create_merged_endpoint()

        result = self.stage.delete(reason="infrastructure cleanup")

        self.assertIsNone(self.sagemaker.endpoint)
        self.assertEqual(result["status"], "DeletedUnrecorded")
        self.assertIsNone(self.store.load(DeploymentStage.RECORD))

    def test_deploy_still_works_after_a_delete_without_record(self) -> None:
        self.stage.delete(reason="infrastructure cleanup")

        self.stage.deploy(DeploymentStage.CONFIRMATION, PACKAGE)

        self.factory.assert_called_once_with(PACKAGE)


class WatchdogDecisionTests(unittest.TestCase):
    """Verifies when the watchdog force-deletes the endpoint."""

    def test_live_endpoint_past_deadline_is_deleted(self) -> None:
        self.assertTrue(should_force_delete({"status": "InService", "deadline_epoch": 10}, now=11))

    def test_live_endpoint_before_deadline_is_kept(self) -> None:
        self.assertFalse(should_force_delete({"status": "InService", "deadline_epoch": 10}, now=9))

    def test_live_record_without_deadline_fails_closed(self) -> None:
        self.assertTrue(should_force_delete({"status": "Creating"}, now=1))

    def test_deleted_or_missing_record_is_ignored(self) -> None:
        self.assertFalse(should_force_delete({"status": "Deleted", "deadline_epoch": 1}, now=99))
        self.assertFalse(should_force_delete(None, now=99))

if __name__ == "__main__":
    unittest.main()

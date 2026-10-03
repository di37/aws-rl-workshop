"""Unit tests for the endpoint watchdog safety net."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from aws.deploy.deployment import DeploymentStage
from aws.deploy.endpoint_watchdog import HEARTBEAT, EndpointWatchdog, watchdog_alive
from aws.records.evidence import EvidenceStore


class EndpointWatchdogTests(unittest.TestCase):
    """Verifies deadline enforcement, resilience, heartbeat, and exits."""

    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.store = EvidenceStore(Path(self.directory.name))
        self.now = [1_000.0]
        self.stage = MagicMock()
        self.stage.delete.side_effect = self._mark_deleted
        self.verify = MagicMock()

    def tearDown(self) -> None:
        self.directory.cleanup()

    def _mark_deleted(self, **_: object) -> dict:
        record = {**(self.store.load(DeploymentStage.RECORD) or {}), "status": "Deleted"}
        self.store.save(DeploymentStage.RECORD, record)
        return record

    def _watchdog(self, max_wait_minutes: int = 240) -> EndpointWatchdog:
        def sleep(seconds: float) -> None:
            self.now[0] += seconds

        return EndpointWatchdog(
            self.store,
            stage_factory=lambda: self.stage,
            verify_account=self.verify,
            clock=lambda: self.now[0],
            sleep=sleep,
            max_wait_minutes=max_wait_minutes,
        )

    def test_live_endpoint_is_deleted_at_its_deadline(self) -> None:
        self.store.save(DeploymentStage.RECORD, {"status": "InService", "deadline_epoch": 1_100})

        reason = self._watchdog().run()

        self.stage.delete.assert_called_once()
        self.assertGreaterEqual(self.now[0], 1_100)
        self.assertEqual(reason, "endpoint deleted")

    def test_errors_are_retried_instead_of_stopping_the_watchdog(self) -> None:
        self.store.save(DeploymentStage.RECORD, {"status": "InService", "deadline_epoch": 0})
        attempts: list[int] = []

        def flaky_delete(**kwargs: object) -> dict:
            attempts.append(1)
            if len(attempts) == 1:
                raise RuntimeError("throttled")
            return self._mark_deleted(**kwargs)

        self.stage.delete.side_effect = flaky_delete

        reason = self._watchdog().run()

        self.assertEqual(self.stage.delete.call_count, 2)
        self.assertEqual(reason, "endpoint deleted")

    def test_exits_when_no_deployment_starts(self) -> None:
        self.assertEqual(self._watchdog(max_wait_minutes=0).run(), "no deployment started")
        self.stage.delete.assert_not_called()

    def test_account_mismatch_stops_before_watching(self) -> None:
        self.verify.side_effect = RuntimeError("wrong account")

        with self.assertRaisesRegex(RuntimeError, "wrong account"):
            self._watchdog().run()

    def test_heartbeat_marks_the_watchdog_alive(self) -> None:
        self.store.save(DeploymentStage.RECORD, {"status": "Deleted"})

        self._watchdog().run()

        self.assertTrue(watchdog_alive(self.store, clock=lambda: self.now[0] + 60))
        self.assertFalse(watchdog_alive(self.store, clock=lambda: self.now[0] + 600))

    def test_missing_heartbeat_means_not_alive(self) -> None:
        self.assertIsNone(self.store.load(HEARTBEAT))
        self.assertFalse(watchdog_alive(self.store))


if __name__ == "__main__":
    unittest.main()

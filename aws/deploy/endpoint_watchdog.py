"""Safety net that deletes the demo endpoint once its lifetime deadline passes.

Start it in another terminal before deploying; deployment refuses to start
without its heartbeat:

    python scripts/09a_watchdog.py

It verifies the AWS account, writes a heartbeat every poll, deletes a live
endpoint at the recorded deadline (or immediately if the deadline is missing),
retries through errors, and exits once the endpoint record says Deleted.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from typing import Any

from aws.deploy.deployment import DeploymentStage, should_force_delete
from aws.records.evidence import EvidenceStore

HEARTBEAT = "endpoint_watchdog.heartbeat.json"
HEARTBEAT_MAX_AGE_SECONDS = 120


class EndpointWatchdog:
    """Enforces the endpoint lifetime deadline independently of the notebook."""

    def __init__(
        self,
        store: EvidenceStore,
        stage_factory: Callable[[], Any],
        verify_account: Callable[[], None],
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
        poll_seconds: int = 30,
        max_wait_minutes: int = 240,
    ) -> None:
        """Initializes the watchdog.

        Args:
            store: Evidence store holding the endpoint record and heartbeat.
            stage_factory: Builds the deployment stage used for deletion.
            verify_account: Raises when credentials belong to another account.
            clock: Epoch-seconds clock.
            sleep: Sleep function.
            poll_seconds: Seconds between checks.
            max_wait_minutes: Exit if no deployment starts within this time.
        """
        self.store = store
        self._stage_factory = stage_factory
        self._verify_account = verify_account
        self._clock = clock
        self._sleep = sleep
        self._poll_seconds = poll_seconds
        self._max_wait_seconds = max_wait_minutes * 60

    def run(self) -> str:
        """Watches until the endpoint is deleted or no deployment starts.

        Returns:
            Why the watchdog stopped.

        Raises:
            RuntimeError: If the AWS account does not match the demo account.
        """
        self._verify_account()
        started = self._clock()
        stage = None
        while True:
            try:
                self._beat()
                record = self.store.load(DeploymentStage.RECORD)
                if record and record.get("status") == "Deleted":
                    return "endpoint deleted"
                if should_force_delete(record, self._clock()):
                    print("Deadline passed: deleting the endpoint now.", flush=True)
                    stage = stage or self._stage_factory()
                    stage.delete(reason="watchdog: lifetime deadline reached")
                    continue
                if record is None and self._clock() - started >= self._max_wait_seconds:
                    return "no deployment started"
            except Exception as error:  # noqa: BLE001 - logged and retried, never fatal
                print(f"Watchdog error, retrying: {type(error).__name__}: {error}", flush=True)
            self._sleep(self._poll_seconds)

    def _beat(self) -> None:
        """Writes the heartbeat that deploy() requires."""
        self.store.save(HEARTBEAT, {"pid": os.getpid(), "updated_epoch": self._clock()})


def watchdog_alive(store: EvidenceStore, clock: Callable[[], float] = time.time) -> bool:
    """Reports whether a watchdog heartbeat is fresh.

    Args:
        store: Evidence store holding the heartbeat.
        clock: Epoch-seconds clock.

    Returns:
        True when the heartbeat is younger than the allowed age.
    """
    heartbeat = store.load(HEARTBEAT)
    if not heartbeat:
        return False
    return clock() - heartbeat.get("updated_epoch", 0) <= HEARTBEAT_MAX_AGE_SECONDS

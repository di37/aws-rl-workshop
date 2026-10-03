"""Configure and deploy the rollout agent through AgentCore CodeBuild."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import boto3

from aws.config import AGENT_DIR, DemoConfig, ResourceState

CONFIRMATION = "CONFIRM_DEPLOY_AGENT_MTRL_DEMO"
REDEPLOY_CONFIRMATION = "CONFIRM_REDEPLOY_AGENT_MTRL_DEMO"
"""Rebuilding a READY runtime changes the agent that recorded jobs used, so it needs its own phrase."""


def required_phrase(state: ResourceState) -> str:
    """Returns the phrase that authorizes building the agent runtime.

    Args:
        state: Provisioned resource identifiers.

    Returns:
        The redeploy phrase when a runtime is already READY, else the deploy phrase.
    """
    return REDEPLOY_CONFIRMATION if state.agent_runtime_status == "READY" else CONFIRMATION


class AgentRuntimeDeployer:
    """Deploys the existing agent code and records its runtime identity."""

    def __init__(self, config: DemoConfig, state: ResourceState) -> None:
        """Initializes the AgentCore deployment coordinator.

        Args:
            config: Shared resource configuration.
            state: Provisioned AWS resource identifiers.
        """
        self.config = config
        self.state = state
        self.control = boto3.client(
            "bedrock-agentcore-control", region_name=config.region
        )

    def deploy(self) -> dict[str, Any]:
        """Configures, builds, and deploys the AgentCore runtime.

        Returns:
            Ready AgentCore runtime metadata.
        """
        self._run_cli(
            "configure",
            "--entrypoint",
            "app.py",
            "--name",
            self.config.agent_name,
            "--execution-role",
            self.state.agent_role_arn,
            "--ecr",
            self.state.ecr_uri,
            "--requirements-file",
            "requirements.txt",
            "--disable-memory",
            "--deployment-type",
            "container",
            "--protocol",
            "HTTP",
            "--region",
            self.config.region,
            "--non-interactive",
        )
        self._run_cli(
            "launch",
            "--agent",
            self.config.agent_name,
            "--auto-update-on-conflict",
        )
        runtime = self._wait_for_runtime()
        self.state.update_runtime(runtime)
        return runtime

    def sync_state(self) -> dict[str, Any]:
        """Discovers the deployed runtime and refreshes local state.

        Returns:
            Ready AgentCore runtime metadata.
        """
        runtime = self._wait_for_runtime()
        self.state.update_runtime(runtime)
        return runtime

    def _run_cli(self, *arguments: str) -> None:
        """Runs the AgentCore CLI installed next to the running Python interpreter.

        Args:
            *arguments: CLI arguments after the executable name.
        """
        executable = Path(sys.executable).with_name("agentcore")
        environment = os.environ.copy()
        environment["AWS_REGION"] = self.config.region
        environment["AWS_DEFAULT_REGION"] = self.config.region
        environment["AGENTCORE_SUPPRESS_RECOMMENDATION"] = "1"
        subprocess.run(
            [str(executable), *arguments],
            cwd=AGENT_DIR,
            env=environment,
            check=True,
        )

    def _wait_for_runtime(self) -> dict[str, Any]:
        """Waits for the named runtime to become ready.

        Returns:
            Ready runtime metadata.

        Raises:
            RuntimeError: If AgentCore reports a failed deployment.
            TimeoutError: If readiness takes longer than 20 minutes.
        """
        deadline = time.monotonic() + 1_200
        while time.monotonic() < deadline:
            runtimes = self.control.list_agent_runtimes().get("agentRuntimes", [])
            matches = [
                runtime
                for runtime in runtimes
                if runtime.get("agentRuntimeName") == self.config.agent_name
            ]
            if matches:
                runtime = max(
                    matches,
                    key=lambda item: item.get("lastUpdatedAt") or item.get("createdAt"),
                )
                status = runtime.get("status")
                if status == "READY":
                    return runtime
                if status in {"CREATE_FAILED", "UPDATE_FAILED"}:
                    raise RuntimeError(f"Agent runtime entered {status}: {runtime}")
            time.sleep(10)
        raise TimeoutError("AgentCore runtime did not become READY within 20 minutes.")

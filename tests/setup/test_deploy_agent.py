"""Unit tests for the agent deployment phrases."""

from __future__ import annotations

import unittest

from aws.config import ResourceState
from aws.setup.deploy_agent import CONFIRMATION, REDEPLOY_CONFIRMATION, required_phrase


def state(status: str | None) -> ResourceState:
    """Builds resource state with an optional runtime status."""
    return ResourceState(account_id="1", region="us-west-2", bucket="b", job_role_arn="j",
                         agent_role_arn="a", ecr_uri="e", mlflow_app_arn="m",
                         agent_runtime_arn="arn:runtime" if status else None,
                         agent_runtime_id="r" if status else None, agent_runtime_status=status)


class RequiredPhraseTests(unittest.TestCase):
    """Verifies a ready runtime, which recorded jobs used, needs a distinct phrase to rebuild."""

    def test_first_deploy_uses_the_deploy_phrase(self) -> None:
        self.assertEqual(required_phrase(state(None)), CONFIRMATION)

    def test_ready_runtime_needs_the_redeploy_phrase(self) -> None:
        self.assertEqual(required_phrase(state("READY")), REDEPLOY_CONFIRMATION)
        self.assertNotEqual(CONFIRMATION, REDEPLOY_CONFIRMATION)


if __name__ == "__main__":
    unittest.main()

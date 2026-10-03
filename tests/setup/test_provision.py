"""Unit tests for carrying provisioned state across reruns."""

from __future__ import annotations

import unittest

from aws.config import ResourceState
from aws.setup.provision import runtime_fields

RUNTIME = "arn:aws:bedrock-agentcore:us-west-2:111111111111:runtime/agent-1"


def recorded_state(account_id: str = "111111111111", region: str = "us-west-2") -> ResourceState:
    """Builds a resource state with a deployed runtime."""
    return ResourceState(
        account_id=account_id, region=region, bucket="b", job_role_arn="j", agent_role_arn="a",
        ecr_uri="e", mlflow_app_arn="m", agent_runtime_arn=RUNTIME, agent_runtime_id="agent-1",
        agent_runtime_status="READY",
    )


class RuntimeCarryOverTests(unittest.TestCase):
    """Verifies a recorded runtime is kept only for the same account and region."""

    def test_same_account_and_region_keeps_the_runtime(self) -> None:
        fields = runtime_fields(recorded_state(), "111111111111", "us-west-2")

        self.assertEqual(fields["agent_runtime_arn"], RUNTIME)
        self.assertEqual(fields["agent_runtime_status"], "READY")

    def test_another_account_starts_without_a_runtime(self) -> None:
        fields = runtime_fields(recorded_state(), "222222222222", "us-west-2")

        self.assertEqual(set(fields.values()), {None})

    def test_another_region_or_no_state_starts_without_a_runtime(self) -> None:
        self.assertEqual(set(runtime_fields(recorded_state(region="us-east-1"), "111111111111",
                                            "us-west-2").values()), {None})
        self.assertEqual(set(runtime_fields(None, "111111111111", "us-west-2").values()), {None})


if __name__ == "__main__":
    unittest.main()

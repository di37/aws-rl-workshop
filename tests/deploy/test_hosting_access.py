"""Unit tests for the scoped endpoint image-pull permission."""

from __future__ import annotations

import json
import unittest
from unittest.mock import MagicMock

from botocore.exceptions import ClientError

from aws.deploy.hosting_access import POLICY_NAME, ensure_hosting_access, hosting_policy


class HostingAccessTests(unittest.TestCase):
    """Verifies the policy scope and idempotent application."""

    def test_policy_only_allows_pulling_the_inference_image(self) -> None:
        statement = hosting_policy("us-west-2")["Statement"][0]

        self.assertEqual(
            sorted(statement["Action"]),
            ["ecr:BatchCheckLayerAvailability", "ecr:BatchGetImage", "ecr:GetDownloadUrlForLayer"],
        )
        self.assertEqual(
            statement["Resource"], "arn:aws:ecr:us-west-2:763104351884:repository/djl-inference"
        )

    def test_missing_policy_is_added(self) -> None:
        iam = MagicMock()
        iam.get_role_policy.side_effect = ClientError(
            {"Error": {"Code": "NoSuchEntity", "Message": "missing"}}, "GetRolePolicy"
        )

        self.assertTrue(ensure_hosting_access(iam, "Role", "us-west-2"))
        call = iam.put_role_policy.call_args.kwargs
        self.assertEqual((call["RoleName"], call["PolicyName"]), ("Role", POLICY_NAME))
        self.assertEqual(json.loads(call["PolicyDocument"]), hosting_policy("us-west-2"))

    def test_identical_policy_is_left_alone(self) -> None:
        iam = MagicMock()
        iam.get_role_policy.return_value = {"PolicyDocument": hosting_policy("us-west-2")}

        self.assertFalse(ensure_hosting_access(iam, "Role", "us-west-2"))
        iam.put_role_policy.assert_not_called()

    def test_other_iam_errors_propagate(self) -> None:
        iam = MagicMock()
        iam.get_role_policy.side_effect = ClientError(
            {"Error": {"Code": "AccessDenied", "Message": "no"}}, "GetRolePolicy"
        )

        with self.assertRaises(ClientError):
            ensure_hosting_access(iam, "Role", "us-west-2")


if __name__ == "__main__":
    unittest.main()

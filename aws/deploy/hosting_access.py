"""Scoped permission the endpoint needs but MTRL training never did.

Training and evaluation run on service-managed infrastructure, so the demo job
role was never allowed to pull container images. A real-time endpoint pulls
the AWS ``djl-inference`` image with that role; this grants exactly that.
"""

from __future__ import annotations

import json
from typing import Any

from botocore.exceptions import ClientError

POLICY_NAME = "sagemaker-mtrl-demo-hosting"
DLC_ACCOUNT = "763104351884"
"""AWS account that publishes the Deep Learning Container images."""


def hosting_policy(region: str) -> dict[str, Any]:
    """Builds the least-privilege image-pull policy.

    Args:
        region: Region of the endpoint and its container image.

    Returns:
        IAM policy document allowing pulls from the ``djl-inference`` repo only.
    """
    return {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "PullInferenceContainer",
                "Effect": "Allow",
                "Action": [
                    "ecr:BatchCheckLayerAvailability",
                    "ecr:BatchGetImage",
                    "ecr:GetDownloadUrlForLayer",
                ],
                "Resource": f"arn:aws:ecr:{region}:{DLC_ACCOUNT}:repository/djl-inference",
            }
        ],
    }


def ensure_hosting_access(iam: Any, role_name: str, region: str) -> bool:
    """Adds the image-pull policy to the role unless it is already present.

    Args:
        iam: Boto3 IAM client.
        role_name: Endpoint execution role.
        region: Region of the endpoint.

    Returns:
        True when the policy was added or updated.
    """
    desired = hosting_policy(region)
    try:
        current = iam.get_role_policy(RoleName=role_name, PolicyName=POLICY_NAME)
        if current["PolicyDocument"] == desired:
            return False
    except ClientError as error:
        if error.response["Error"]["Code"] != "NoSuchEntity":
            raise
    iam.put_role_policy(
        RoleName=role_name, PolicyName=POLICY_NAME, PolicyDocument=json.dumps(desired)
    )
    return True

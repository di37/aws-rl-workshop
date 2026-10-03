"""Provision the demo's AWS resources once (idempotent; needs --confirm).

Creates or updates the job and agent IAM roles, the S3 bucket, the ECR
repository for the agent image, the serverless MLflow app, and a $25 AWS
Budget (optional email alert at 80% forecast). Then grants the job role
permission to pull the AWS inference container, which only deployment option 1
needs. Writes the identifiers to artifacts/resources.json.
"""

# region Imports & setup
from __future__ import annotations

import os

import _bootstrap as bootstrap
import boto3

from aws.deploy.hosting_access import ensure_hosting_access
from aws.guards import require_phrase
from aws.setup.provision import CONFIRMATION, InfrastructureProvisioner

# endregion


# region Entry point
def main() -> None:
    """Provisions after the exact confirmation phrase."""
    args = bootstrap.parse_args(__doc__.splitlines()[0], CONFIRMATION, extra=lambda parser: parser.add_argument(
        "--budget-email", default=os.environ.get("MTRL_BUDGET_EMAIL"),
        help="optional address for the budget's 80%% forecast alert"), shows_record=False)
    bootstrap.banner("Provision (01)", "IAM roles, S3, ECR, MLflow app, budget, endpoint image access")
    require_phrase(args.confirm, CONFIRMATION, "provisioning")
    state = InfrastructureProvisioner(bootstrap.CONFIG).provision(args.budget_email)
    changed = ensure_hosting_access(boto3.client("iam"), bootstrap.CONFIG.job_role_name, bootstrap.CONFIG.region)
    bootstrap.show([{"resource": key, "value": value} for key, value in vars(state).items()])
    print(f"Endpoint image-pull policy {'added' if changed else 'already present'}.")


if __name__ == "__main__":
    bootstrap.run(main)
# endregion

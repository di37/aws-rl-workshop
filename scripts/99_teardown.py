"""Delete every demo resource in AWS (needs --confirm DELETE_MTRL_DEMO).

In order: the endpoint (if any); the Bedrock imported models, the us-east-1
copy bucket, and the import role (only if tagged by this demo); the AgentCore
runtime; the MLflow app (with its traces); the ECR repository; the S3 bucket
(datasets and job outputs); both IAM roles; and the budget. The evidence in
artifacts/ and the reports stay, so steps 12-14 still work afterwards.
"""

# region Imports & setup
from __future__ import annotations

import _bootstrap as bootstrap

from aws.guards import require_phrase
from aws.setup.cleanup import CONFIRMATION, InfrastructureCleaner

# endregion


# region Entry point
def main() -> None:
    """Deletes the tracked resources after the exact confirmation phrase."""
    args = bootstrap.parse_args(__doc__.splitlines()[0], CONFIRMATION, shows_record=False)
    bootstrap.banner("Teardown (99)", "delete every demo resource; keep the local evidence")
    require_phrase(args.confirm, CONFIRMATION, "teardown")
    InfrastructureCleaner(bootstrap.CONFIG, bootstrap.require_state()).cleanup()
    print("Deleted tracked demo resources. Evidence in artifacts/ and reports/ is kept.")


if __name__ == "__main__":
    bootstrap.run(main)
# endregion

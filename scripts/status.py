"""Show the status of the demo's AWS resources (read-only; run any time).

Prints the account, region, S3 bucket, MLflow app, and AgentCore runtime as
AWS reports them. Useful before and after teardown.
"""

# region Imports & setup
from __future__ import annotations

import json

import _bootstrap as bootstrap

from aws.setup.status import ResourceStatusReporter

# endregion


# region Entry point
def main() -> None:
    """Prints the current resource status as JSON."""
    bootstrap.parse_args(__doc__.splitlines()[0])
    bootstrap.banner("Resource status", "S3, MLflow app, AgentCore runtime (read-only)")
    report = ResourceStatusReporter(bootstrap.require_state()).build()
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    bootstrap.run(main)
# endregion

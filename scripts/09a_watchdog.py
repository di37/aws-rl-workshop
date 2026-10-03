"""Safety watchdog for deployment option 1: start it before 09a_deploy_endpoint.py.

Run it in a second terminal and leave it running. It verifies the AWS account,
writes a heartbeat (deployment refuses to start without one), deletes a live
endpoint at its recorded deadline (60 minutes after creation), retries through
errors, and exits once the endpoint record says Deleted.
"""

# region Imports & setup
from __future__ import annotations

from typing import Any

import _bootstrap as bootstrap

from aws.costs.cost_guard import MtrlCostGuard
from aws.deploy.endpoint_watchdog import EndpointWatchdog

# endregion


# region Entry point
def main() -> None:
    """Runs the watchdog until the endpoint is deleted or the wait limit passes."""
    args = bootstrap.parse_args(__doc__.splitlines()[0], extra=lambda parser: (
        parser.add_argument("--max-wait-minutes", type=int, default=240),
        parser.add_argument("--poll-seconds", type=int, default=30)))
    bootstrap.banner("Endpoint watchdog (09a)", "deletes the endpoint at its deadline; keep this running")
    workflow = bootstrap.load_workflow()

    def stage() -> Any:
        hourly = MtrlCostGuard.from_price_list(workflow.config).hosting_hourly_usd
        return workflow.deployment_stage(hourly_usd=hourly)

    watchdog = EndpointWatchdog(workflow.store, stage_factory=stage, verify_account=workflow.verify_account,
                                poll_seconds=args.poll_seconds, max_wait_minutes=args.max_wait_minutes)
    print("Watchdog stopped:", watchdog.run(), flush=True)


if __name__ == "__main__":
    bootstrap.run(main)
# endregion

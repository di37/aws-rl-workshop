"""Deployment option 1: SageMaker real-time endpoint, as in the official guide (needs --confirm).

ModelPackage.get -> ModelBuilder(model=package).build() -> deploy() on one
ml.g6e.12xlarge (about $13 per hour). Start the watchdog first, in another
terminal: python scripts/09a_watchdog.py. It deletes the endpoint after 60
minutes at the latest, and deployment refuses to start without it. Delete
with --delete as soon as you are done. Run of record: no ml.g6e.12xlarge
capacity (InsufficientInstanceCapacity); the endpoint never reached
InService and nothing was billed.
"""

# region Imports & setup
from __future__ import annotations

import _bootstrap as bootstrap

from aws.costs.cost_guard import MtrlCostGuard
from aws.deploy.deployment import DeploymentStage

FIELDS = ("endpoint_name", "status", "endpoint_status", "instance_type", "uptime_minutes", "billed_usd",
          "billing_note", "failure_reason")
# endregion


# region Entry point
def main() -> None:
    """Deploys (when confirmed), shows, or deletes the endpoint."""
    args = bootstrap.parse_args(__doc__.splitlines()[0], DeploymentStage.CONFIRMATION, extra=lambda parser: (
        parser.add_argument("--delete", action="store_true", help="delete the endpoint and record its cost")))
    bootstrap.banner("Deploy endpoint (09a)", "ModelBuilder -> ml.g6e.12xlarge endpoint (option 1)")
    workflow = bootstrap.load_workflow()
    guard = MtrlCostGuard.from_price_list(bootstrap.CONFIG)
    deployment = workflow.deployment_stage(hourly_usd=guard.hosting_hourly_usd)
    recorded = workflow.store.load(DeploymentStage.RECORD) or {}
    if args.delete:
        endpoint = recorded if recorded.get("status") == "Deleted" else deployment.delete()
    else:
        job = workflow.training_stage().run_or_attach("")
        package = job.output_model_package_arn if job is not None and job.job_status == "Completed" else None
        endpoint = deployment.deploy(args.confirm, package)
    if endpoint is None:
        print(f"No endpoint recorded. Start the watchdog, then pass --confirm {DeploymentStage.CONFIRMATION}.")
        return
    bootstrap.show_fields(endpoint, FIELDS)
    if endpoint.get("components"):
        bootstrap.show(endpoint["components"])
    if endpoint.get("status") == "Deleted" and endpoint.get("failure_reason"):
        print(f"A failed attempt is recorded. To try again: --confirm {DeploymentStage.REDEPLOY_CONFIRMATION}")


if __name__ == "__main__":
    bootstrap.run(main)
# endregion

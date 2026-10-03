"""Verify the environment and the fixed datasets; with --aws, also the live AWS setup (read-only).

Checks that installed packages match the exact pins of the run of record and
that the prompt files are byte-identical to the run of record, with a disjoint
held-out set. With --aws it also asks AWS for the account, region, supported
base model, and the live status of the AgentCore runtime and the MLflow app.
Creates nothing and bills nothing. Exits non-zero if any check fails.
"""

# region Imports & setup
from __future__ import annotations

import sys

import _bootstrap as bootstrap

from aws.reporting.invariants import (
    RUN_OF_RECORD_DATASETS,
    check_dataset_fingerprints,
    check_datasets,
    check_environment,
)
from aws.reporting.repro_artifacts import exact_requirements_file

# endregion


# region Checks
def aws_check() -> tuple[str, bool, str]:
    """Validates identity, region, model support, and the live runtime and MLflow app (read-only)."""
    label = "AWS account, region, model, live AgentCore runtime and MLflow app"
    try:
        details = bootstrap.load_workflow().validate()
    except Exception as error:  # noqa: BLE001 - reported as a FAIL line with the reason
        return label, False, f"{type(error).__name__}: {error}"
    return label, True, (f"account {details['account_id']}, {details['region']}, model {details['model']}, "
                         f"runtime {details['agent_runtime_status']}, MLflow app {details['mlflow_app_status']}")
# endregion


# region Entry point
def main() -> None:
    """Runs every check and exits non-zero on failure."""
    args = bootstrap.parse_args(__doc__.splitlines()[0], extra=lambda parser: parser.add_argument(
        "--aws", action="store_true", help="also check the live AWS setup (read-only)"))
    bootstrap.banner("Verify setup (00)", "exact pins, datasets" + (", live AWS resources" if args.aws else ""))
    results = [
        ("environment matches the exact pins", *check_environment(exact_requirements_file(bootstrap.PROJECT_ROOT))),
        ("datasets: sizes, uniqueness, held-out disjoint", *check_datasets(bootstrap.DATA_DIR)),
        ("datasets unchanged since the run of record",
         *check_dataset_fingerprints(bootstrap.DATA_DIR, RUN_OF_RECORD_DATASETS)),
    ]
    if args.aws:
        results.append(aws_check())
    if bootstrap.report(results):
        sys.exit(1)


if __name__ == "__main__":
    bootstrap.run(main)
# endregion

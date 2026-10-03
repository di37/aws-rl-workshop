"""Write the reproducibility record to reports/repro/ (offline).

Five files: environment_versions.csv (installed vs pinned versions),
run_commands.csv (the ordered steps), study_metadata.json (datasets with
SHA-256, task, model, settings, seeds, every AWS job of the run of record,
headline results, cost totals), compute_accounting.csv, and
artifact_inventory.csv. Runs nothing on AWS; reads the evidence and the live
Python environment only.
"""

# region Imports & setup
from __future__ import annotations

import _bootstrap as bootstrap
from environment import MAX_TURNS, SUCCESS_PATH

from aws.costs.cost_snapshot import RECORD as COST_RECORD
from aws.records.evidence import EvidenceStore
from aws.reporting.report_tables import cost_table
from aws.reporting.repro_artifacts import (
    artifact_inventory,
    environment_versions,
    pinned_requirements,
    requirements_file,
    study_metadata,
    write_repro_record,
)

# endregion


# region Entry point
def main() -> None:
    """Builds and writes the five reproducibility files."""
    bootstrap.parse_args(__doc__.splitlines()[0])
    bootstrap.banner("Reproducibility record (13)", "environment, commands, metadata, costs, inventory")
    store = EvidenceStore(bootstrap.ARTIFACTS_DIR)
    snapshot = store.load(COST_RECORD)
    if snapshot is None:
        raise FileNotFoundError(f"artifacts/{COST_RECORD} is missing: run scripts/11_snapshot_costs.py first.")
    environment = environment_versions(pinned_requirements(requirements_file(bootstrap.PROJECT_ROOT)))
    metadata = study_metadata(bootstrap.PROJECT_ROOT, store, bootstrap.CONFIG, snapshot,
                              task={"success_path": list(SUCCESS_PATH), "max_turns": MAX_TURNS})
    paths = write_repro_record(bootstrap.REPRO_DIR, environment=environment, metadata=metadata,
                               cost_lines=cost_table(snapshot), inventory=artifact_inventory(bootstrap.PROJECT_ROOT))
    for path in paths:
        print(f"  saved {path.relative_to(bootstrap.PROJECT_ROOT)}")
    mismatched = [row["package"] for row in environment if not row["matches_pin"]]
    print(f"\nPackages checked: {len(environment)}; not matching their pin: {mismatched or 'none'}")


if __name__ == "__main__":
    bootstrap.run(main)
# endregion

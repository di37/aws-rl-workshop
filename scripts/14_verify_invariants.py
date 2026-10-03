"""Verify the study's reproducibility invariants (offline); exits non-zero on failure.

Runs nothing on AWS. Re-checks, against the files on disk: the fixed datasets
(sizes, held-out disjointness, run-of-record fingerprints); the training run
(approved size, every step's metrics); the learning signal; the comparison
(succeeded, 32 x 2 per model; recorded traces reproduce its counts); the
Bedrock import (same model package); the live-inference protocol (first
held-out tickets, documented seed); no endpoint left running; spend within the
$25 cap; no signed links in evidence or notebooks; pinned environment; clean
notebooks; and the presence of every report table and figure.
"""

# region Imports & setup
from __future__ import annotations

import sys

import _bootstrap as bootstrap

from aws.reporting.invariants import run_all

# endregion


# region Entry point
def main() -> None:
    """Prints a PASS/FAIL line per invariant; exits non-zero if any fails."""
    bootstrap.parse_args(__doc__.splitlines()[0])
    bootstrap.banner("Invariants (14)", "PASS/FAIL reproducibility checks over data, evidence, and outputs")
    if bootstrap.report(run_all(bootstrap.PROJECT_ROOT)):
        sys.exit(1)


if __name__ == "__main__":
    bootstrap.run(main)
# endregion

"""Check notebooks before or after execution: every cell parses, no hidden tracebacks, no signed links.

    python scripts/dev/check_notebook.py NOTEBOOK [NOTEBOOK ...]

Uses the same checks as scripts/14_verify_invariants.py. Exits 1 on any problem.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aws.reporting.invariants import (  # noqa: E402
    check_no_signed_links,
    check_notebooks,
)


def main() -> None:
    """Prints one PASS/FAIL line per notebook and check."""
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    failures = 0
    for path in (Path(arg) for arg in sys.argv[1:]):
        for label, (ok, detail) in (("cells", check_notebooks([path])),
                                    ("signed links", check_no_signed_links([path]))):
            print(f"[{'PASS' if ok else 'FAIL'}] {path.name} {label}: {detail}")
            failures += not ok
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()

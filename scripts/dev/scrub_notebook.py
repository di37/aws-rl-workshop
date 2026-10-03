"""Remove temporary sign-in links from executed notebooks; show the home folder as ``~``.

    python scripts/dev/scrub_notebook.py NOTEBOOK [NOTEBOOK ...]

Run it after every notebook execution, before sharing.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aws.records.evidence import scrub_notebook  # noqa: E402


def main() -> None:
    """Scrubs each notebook in place and reports the links removed."""
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    for argument in sys.argv[1:]:
        print(f"{argument}: {scrub_notebook(Path(argument))} signed links removed")


if __name__ == "__main__":
    main()

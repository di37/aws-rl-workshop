"""Build the reproducibility sheet, a PDF, from the evidence (offline).

Runs every invariant first and refuses to build unless all pass. Then fills
aws/reporting/templates/repro_sheet.tex with values read from artifacts/ and
reports/ (no number is typed by hand), writes reports/repro/repro_sheet.tex,
compiles it with tectonic, and saves REPRO_SageMaker_MTRL.pdf at the project
root. Without tectonic (brew install tectonic), compile the .tex on Overleaf
together with reports/figures/.
"""

# region Imports & setup
from __future__ import annotations

import shutil
import subprocess

import _bootstrap as bootstrap
from environment import MAX_PROGRESS, MAX_TURNS, SUCCESS_PATH
from rollout_driver import MAX_POLICY_CALLS

from aws.reporting.invariants import run_all
from aws.reporting.repro_sheet import SHEET_NAME, build_sheet

# endregion


# region Entry point
def main() -> None:
    """Checks the invariants, renders the LaTeX, and compiles the PDF."""
    bootstrap.parse_args(__doc__.splitlines()[0])
    bootstrap.banner("Reproducibility sheet (15)", "invariants -> LaTeX -> PDF, from the evidence")
    task = {"success_path": list(SUCCESS_PATH), "max_turns": MAX_TURNS, "max_progress": MAX_PROGRESS,
            "max_policy_calls": MAX_POLICY_CALLS}
    tex = bootstrap.REPRO_DIR / "repro_sheet.tex"
    tex.write_text(build_sheet(bootstrap.PROJECT_ROOT, run_all(bootstrap.PROJECT_ROOT), task))
    print(f"  saved {tex.relative_to(bootstrap.PROJECT_ROOT)}")
    compiler = shutil.which("tectonic")
    if compiler is None:
        raise FileNotFoundError("tectonic is not installed (brew install tectonic); compile "
                                "reports/repro/repro_sheet.tex on Overleaf with reports/figures/ instead.")
    result = subprocess.run([compiler, "-X", "compile", tex.name], cwd=tex.parent, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"LaTeX compilation failed:\n{result.stderr[-2000:]}")
    target = bootstrap.PROJECT_ROOT / f"{SHEET_NAME}.pdf"
    shutil.copyfile(tex.with_suffix(".pdf"), target)
    print(f"  saved {target.name}")


if __name__ == "__main__":
    bootstrap.run(main)
# endregion

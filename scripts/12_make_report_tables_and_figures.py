"""Rebuild every report table and figure from the recorded evidence (offline).

Reads only artifacts/*.json and writes reports/tables/*.csv and
reports/figures/*.png. No AWS access is needed, so anyone can regenerate the
reported numbers from a clean checkout of the repository.
"""

# region Imports & setup
from __future__ import annotations

from pathlib import Path
from typing import Any

import _bootstrap as bootstrap
import pandas as pd

from aws.records.evidence import EvidenceStore
from aws.reporting.report_figures import (
    evaluation_before_after,
    live_inference_outcomes,
    per_ticket_scatter,
    training_reward_curve,
)
from aws.reporting.report_tables import FIGURE_FILES, TABLE_FILES, build_tables

# endregion


# region Tables & figures
def write_figures(tables: dict[str, list[dict[str, Any]]]) -> list[Path]:
    """Renders every report figure from the tables."""
    return [
        training_reward_curve(tables["training_steps.csv"], bootstrap.FIGURES_DIR / FIGURE_FILES[0]),
        evaluation_before_after(tables["evaluation_comparison.csv"], bootstrap.FIGURES_DIR / FIGURE_FILES[1]),
        per_ticket_scatter(tables["per_ticket_rewards.csv"], bootstrap.FIGURES_DIR / FIGURE_FILES[2]),
        live_inference_outcomes(tables["live_inference.csv"], bootstrap.FIGURES_DIR / FIGURE_FILES[3]),
    ]
# endregion


# region Entry point
def main() -> None:
    """Writes all tables and figures and lists them."""
    bootstrap.parse_args(__doc__.splitlines()[0])
    bootstrap.banner("Report tables and figures (12)", "artifacts/*.json -> reports/tables, reports/figures")
    tables = build_tables(EvidenceStore(bootstrap.ARTIFACTS_DIR))
    if tuple(tables) != TABLE_FILES:
        raise RuntimeError("The built tables differ from report_tables.TABLE_FILES.")
    bootstrap.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    for name, rows in tables.items():
        pd.DataFrame(rows).to_csv(bootstrap.TABLES_DIR / name, index=False)
        print(f"  saved reports/tables/{name} ({len(rows)} rows)")
    for path in write_figures(tables):
        print(f"  saved reports/figures/{path.name}")
    print()
    bootstrap.show([{k: row[k] for k in ("label", "base", "fine_tuned", "delta")}
                    for row in tables["evaluation_comparison.csv"]])


if __name__ == "__main__":
    bootstrap.run(main)
# endregion

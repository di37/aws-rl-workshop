# reports/

Everything here is rebuilt offline from the evidence in `artifacts/`, with no AWS access. An invariant check confirms the tables equal a fresh rebuild and that every figure is present.

| Folder | Built by | Contents |
|---|---|---|
| `tables/` | `scripts/12` | CSVs: `evaluation_comparison`, `baseline_evaluations`, `training_steps`, `per_ticket_rewards`, `live_inference`, `run_of_record`, `cost_accounting` |
| `figures/` | `scripts/12` | PNGs: the training reward curve, before/after evaluation, per-ticket before/after, live inference |
| `repro/` | `scripts/13`, `scripts/15` | The reproducibility record: `environment_versions.csv`, `run_commands.csv`, `study_metadata.json` (datasets, task, model, the job's own config, seeds, headline results, costs), `compute_accounting.csv`, `artifact_inventory.csv`; plus the sheet's source and PDF, `repro_sheet.tex` and `repro_sheet.pdf` |
| `logs/` | every script | One log per script run, with signed links removed and the home folder shown as `~`. The last 5 per script are kept, on your machine only (git ignores them). |

To rebuild and check everything:

```bash
python scripts/12_make_report_tables_and_figures.py
python scripts/13_build_repro_artifacts.py
python scripts/14_verify_invariants.py
python scripts/15_build_repro_sheet.py   # needs tectonic
```

Only the generation timestamp and the installed-environment rows depend on the machine.

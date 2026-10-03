# aws/

The library behind the study. The numbered scripts and the notebook are thin wrappers over it, so both run exactly the same code. Start reading at `workflow.py`.

| Module | Role |
|---|---|
| `workflow.py` | `MtrlDemoWorkflow`, the one facade that wires everything: setup checks, datasets, baseline, and the training, evaluation, deployment and inference stages |
| `config.py` | Settings, project paths and the persisted resource state |
| `guards.py` | The confirmation check every billable step goes through |

| Package | Modules | Role |
|---|---|---|
| `setup/` | `provision`, `request_quotas`, `deploy_agent`, `prompt_datasets`, `status`, `cleanup` | IAM roles, S3, ECR, MLflow and the budget; quotas; the agent on AgentCore; dataset upload; status; teardown |
| `rl/` | `training`, `evaluation`, `pipelines`, `trajectories` | `MultiTurnRLTrainer` and `MultiTurnRLEvaluator` stages (run or re-attach), pipeline status, and the conversations SageMaker recorded |
| `deploy/` | `deployment`, `endpoint_watchdog`, `hosting_access`, `bedrock_import`, `inference` | Option 1, the SageMaker endpoint with its watchdog; option 2, Bedrock Custom Model Import; live before/after inference |
| `costs/` | `cost_guard`, `ledger`, `cost_snapshot` | The $25 hard-cap budget guard, the ledger of billed spend, and the frozen cost record |
| `records/` | `evidence`, `provenance` | Redacted, atomically written evidence files, notebook scrubbing, and what actually ran |
| `reporting/` | `report_tables`, `report_figures`, `repro_artifacts`, `repro_sheet`, `repro_sheet_materials`, `latex`, `invariants`, `provenance_checks`, `templates/` | Report tables and figures, the reproducibility record and sheet, and the 19 PASS/FAIL invariants, all offline |

## Rules the library keeps

- **Nothing billable without its phrase.** Every step that costs money needs its exact confirmation phrase; without one, it shows the recorded result.
- **Run or re-attach.** A recorded job is re-attached, never submitted twice. Only a failed or stopped job can be replaced, with a separate retry phrase.
- **Budget first.** The budget guard runs before training, the comparison, the endpoint, the Bedrock import and live inference.
- **Evidence is safe to share.** Evidence is redacted (signed links and tokens removed) and written atomically to `artifacts/`.
- **Reporting is offline.** It reads only `artifacts/`, so it works without AWS access.

Tests for every package are in `tests/`, in matching folders.

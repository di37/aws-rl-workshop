# scripts/

The pipeline, as numbered scripts that run in order. Each is a thin wrapper over the `aws/` library, and logs to `reports/logs/`. The full commands for each path, free offline check or full run in your own account, are in the main README.

| Script | Does | Cost |
|---|---|---|
| `00_verify_setup.py` | Checks the environment and the fixed datasets; `--aws` adds the live account, model, runtime and MLflow checks | free |
| `01_provision.py` | Creates the AWS resources once: roles, S3, ECR, MLflow and the budget | `--confirm` |
| `02_request_quotas.py` | Reports the SageMaker quotas needed; `--submit` requests missing ones (approval can take hours) | free |
| `03_deploy_agent.py` | Builds and deploys the support agent to AgentCore | `--confirm` |
| `04_upload_datasets.py` | Uploads the datasets to S3 as Parquet, skipping unchanged files | free |
| `05_base_evaluation.py` | Baseline: the untrained model through the agent | ~ $0.04 |
| `06_train.py` | Training with `MultiTurnRLTrainer` | ~ $2.83 |
| `07_compare_evaluation.py` | Base vs fine-tuned on the held-out tickets | ~ $0.07 |
| `08_recorded_trajectories.py` | Downloads the conversations SageMaker recorded | read-only |
| `09a_deploy_endpoint.py` | Option 1: SageMaker real-time endpoint; start `09a_watchdog.py` first | ~ $13/hour |
| `09b_bedrock_import.py` | Option 2: import the fine-tuned model into Bedrock | ~ $0.84 |
| `10_live_inference.py` | Live before vs after on held-out tickets | per minute |
| `11_snapshot_costs.py`, `11b_snapshot_provenance.py` | Freeze the costs, and record what actually ran | read-only |
| `12_make_report_tables_and_figures.py` | Report tables and figures from the evidence | offline |
| `13_build_repro_artifacts.py` | The reproducibility record in `reports/repro/` | offline |
| `14_verify_invariants.py` | The 19 PASS/FAIL checks; exits 1 on failure | offline |
| `15_build_repro_sheet.py` | The reproducibility sheet PDF (needs tectonic) | offline |
| `99_teardown.py` | Deletes every demo resource (`--confirm DELETE_MTRL_DEMO`) | free |

**How they behave**

- **Shows the record unless confirmed.** A billable step runs only with its exact `--confirm` phrase; without it, the script shows the recorded result.
- **Never submits a job twice.** A recorded job is re-attached. Replacing a failed job needs a separate retry phrase.
- **Budget guard first.** The $25 hard cap is checked before training, the comparison, the endpoint, the Bedrock import and live inference.
- **Shared startup.** `_bootstrap.py` is imported first by every numbered script; it sets up paths, logging and argument parsing.

**Also here**

- `status.py`: what exists in AWS right now (read-only, any time).
- `09a_watchdog.py`: the safety net that deletes the endpoint after 60 minutes at the latest.
- `dev/`: `build_notebook.py` (generates the presentation notebook), `check_notebook.py` and `scrub_notebook.py` (check and clean executed notebooks).

# artifacts/

The evidence of the run of record (1–2 October 2026, us-west-2): what each AWS job was, what it reported, what it cost and what actually ran. The scripts write these files (redacted and written atomically by `aws/records/evidence.py`), and the offline steps `12`–`15` rebuild every table, figure and the reproducibility sheet from them alone. Don't edit them by hand, except `extra_costs.json`.

| Stage | Files |
|---|---|
| Setup | `resources.json` (account, region, bucket, roles, ECR), `dataset_provenance.json` (the S3 datasets the jobs read), `agent_provenance.json` and `agent_deployed_source.zip` (the agent runtime, image and source that served every rollout) |
| Baseline | `base_evaluation.json`, `base_evaluation_metrics.json`, `base_evaluation_pipeline_definition.json`; the first attempt, before the agent fix, is kept as `*.pre_fix.json` |
| Training | `training_job.json` (job, its own config and billed tokens), `training_metrics.json` (per-step reward, turns and tokens); the rejected first attempt is `training_job.openai-reasoning-gpt-oss-20b-mtrl-20261002040240.json`, and the original 32 prompts are `training_prompts.32.csv` |
| Comparison | `comparison_evaluation.json`, `comparison_evaluation_metrics.json` (base and fine-tuned), `comparison_pipeline_definition.json`, `evaluation_trajectories.json` (the recorded conversations) |
| Deployment | `endpoint.json` (option 1: no GPU capacity, never in service, $0), `bedrock_import.json` and `bedrock_import_provenance.json` (option 2: the import and the files it read) |
| Live inference | `inference_before_after.json` (6 held-out tickets, seed 2026, before vs after), `inference_chat.json` (one chat reply) |
| Costs | `cost_accounting.json` (billed tokens × AWS Price List rates, plus totals), `extra_costs.json` (amounts computed or estimated by hand; replace them with AWS Cost Explorer's numbers) |
| Notebook logs | `live-inference-run.executed.ipynb`, `deploy-attempt1-capacity-failure.executed.ipynb` (its "$11.33 billed" line is outdated: the endpoint never ran) |

Kept on your machine only (git ignores them): the Parquet copies of the datasets, the endpoint watchdog's heartbeat, `superseded/` and the old Cursor notebook copies.

**Starting a fresh run in your own account?** Move this folder aside first (`mv artifacts artifacts-run-of-record && mkdir artifacts`), or the scripts will re-attach to these recorded jobs instead of submitting yours.

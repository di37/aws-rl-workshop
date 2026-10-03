# SageMaker Multi-Turn RL: training a customer-support agent on GPT-OSS-20B

A study of whether multi-turn reinforcement learning on Amazon SageMaker AI teaches an LLM agent to follow a troubleshooting procedure. It follows the official guide, [Multi-turn reinforcement learning in SageMaker AI](https://docs.aws.amazon.com/sagemaker/latest/dg/model-customize-mtrl.html), on real AWS resources:

- **The agent:** GPT-OSS-20B inside a Strands agent on Amazon Bedrock AgentCore.
- **The task:** a 4-turn internet-outage ticket where only one action per situation makes progress. Each correct step earns 0.25; a solved ticket earns 1.0.
- **Measurement:** a baseline, then training (`MultiTurnRLTrainer`), then base vs fine-tuned on 32 held-out tickets (`MultiTurnRLEvaluator`).
- **Deployment:** a SageMaker endpoint was attempted but no GPU capacity was available; Bedrock Custom Model Import succeeded.

## Results

| Held-out tickets, 32 x 2 tries per model | Base | Fine-tuned |
|---|---|---|
| Fully solved (pass@1) | 0.578 | **0.781** |
| Solved in 1 of 2 tries (pass@2) | 0.813 | **0.938** |
| Live inference on Bedrock, 6 tickets | 0 of 6 | **5 of 6** |

Training took 10 steps of 32 tickets x 4 tries (1,280 conversations) in 19.2 minutes for $2.83. The whole study cost $5.78 against a $25 hard cap.

**Everything else is in the [Reproducibility Sheet](REPRO_SageMaker_MTRL_community-day.pdf):** job identifiers, data fingerprints, seeds, pinned versions, every result table, the cost breakdown, disclosures and the 19 invariant checks, all read from the evidence in `artifacts/`. It's in the AWS Community Day talk's theme; `scripts/15` rebuilds the same content as `REPRO_SageMaker_MTRL.pdf`. Diagrams of each step are in [HOW_IT_WORKS.md](HOW_IT_WORKS.md).

## Setup

```bash
conda env create -f environment-macos.yaml      # or environment-linux.yaml
conda activate aws-mtrl
# or: pip install -r requirements-macos.txt     (exact: requirements-macos.lock.txt)
```

AWS steps need credentials for an account with SageMaker AI, Bedrock and AgentCore access in **us-west-2** (Bedrock imports of GPT-OSS run in us-east-1). Every script pins the region itself.

## Run

The pipeline is the numbered scripts in `scripts/`. A billable step runs only with its exact `--confirm` phrase; without it, a script shows the recorded result. A recorded job is never submitted twice, and a budget guard enforces the $25 cap.

**Rebuild every reported number, free and offline** (no AWS account needed):

```bash
python scripts/00_verify_setup.py                    # exact pins, datasets, fingerprints
python scripts/12_make_report_tables_and_figures.py  # artifacts/*.json -> reports/tables, reports/figures
python scripts/13_build_repro_artifacts.py           # reports/repro: environment, commands, metadata, costs
python scripts/14_verify_invariants.py               # 19/19 PASS; exits 1 on failure
python scripts/15_build_repro_sheet.py               # the PDF sheet, plain style (needs tectonic)
```

**Repeat the study in your own account** (about $6–8). First move this run's evidence aside with `mv artifacts artifacts-run-of-record && mv reports reports-run-of-record && mkdir artifacts`, then:

```bash
python scripts/00_verify_setup.py
python scripts/01_provision.py --confirm CONFIRM_PROVISION_MTRL_DEMO [--budget-email you@example.com]
python scripts/02_request_quotas.py --submit         # free; approval can take hours, so start here
python scripts/03_deploy_agent.py --confirm CONFIRM_DEPLOY_AGENT_MTRL_DEMO
python scripts/04_upload_datasets.py
python scripts/00_verify_setup.py --aws
python scripts/05_base_evaluation.py --confirm CONFIRM_BASE_EVAL_MTRL_DEMO          # ~ $0.04
python scripts/06_train.py --confirm CONFIRM_TRAIN_MTRL_DEMO                        # ~ $2.83, ~20 min
python scripts/07_compare_evaluation.py --confirm CONFIRM_COMPARISON_EVAL_MTRL_DEMO # ~ $0.07
python scripts/08_recorded_trajectories.py
python scripts/09a_deploy_endpoint.py --confirm CONFIRM_DEPLOY_MTRL_DEMO            # optional, ~ $13 per hour
python scripts/09b_bedrock_import.py --confirm CONFIRM_BEDROCK_IMPORT_MTRL_DEMO     # ~ $0.84 copy; ~ 25 min
python scripts/10_live_inference.py --confirm CONFIRM_LIVE_INFERENCE_MTRL_DEMO      # imported model, per minute
python scripts/11_snapshot_costs.py
python scripts/11b_snapshot_provenance.py
python scripts/12_make_report_tables_and_figures.py
python scripts/13_build_repro_artifacts.py
python scripts/14_verify_invariants.py
python scripts/15_build_repro_sheet.py
python scripts/99_teardown.py --confirm DELETE_MTRL_DEMO                            # when done
```

- **Endpoint (09a).** Start `python scripts/09a_watchdog.py` in another terminal first; it deletes the endpoint after 60 minutes at the latest, and deployment refuses to start without it.
- **Rebuilding the agent.** Redeploying a runtime that is already READY needs `CONFIRM_REDEPLOY_AGENT_MTRL_DEMO`, because recorded jobs used it.
- **Costs AWS doesn't report per job** (the cross-region copy, imported-model minutes, Bedrock storage, AgentCore, CodeBuild, S3 and logs) go into `artifacts/extra_costs.json` by hand. Replace them with AWS Cost Explorer's numbers a day later.
- **Your numbers will differ** within sampling noise, since service-side sampling can't be seeded. Compare whether the fine-tuned model beats the base model on pass@1.

## Project layout

```text
scripts/      the pipeline, in order: 00 … 15, then 99 (status.py shows what exists in AWS)
aws/          the library the scripts and the notebook share; start reading at aws/workflow.py
agent/        the agent AgentCore runs: environment, rollout driver, app
notebooks/    what to show an audience (see below)
data/         64 training tickets and 32 held-out tickets, no overlap
artifacts/    evidence of the run of record: job records, costs, provenance
reports/      tables, figures and the reproducibility record, rebuilt from artifacts/
tests/        unit tests, mirroring aws/
```

## Notebooks

- **`notebooks/sagemaker_mtrl_real_demo.ipynb`:** the presentation notebook, sections 1–14 in script order. It replays the recorded run unless a confirmation phrase is set through `MTRL_*` environment variables, and it needs the AWS setup from steps 01–04.
- **`notebooks/sagemaker_mtrl_real_demo.executed.ipynb`:** the replay with all outputs. Present from this one.
- **`notebooks/sagemaker_mtrl_real_demo.training-run.ipynb`:** the log of the live training run, kept as recorded. Read it; don't re-run it.
- **`artifacts/live-inference-run.executed.ipynb`** and **`artifacts/deploy-attempt1-capacity-failure.executed.ipynb`:** the logs of the live inference and the first endpoint attempt. The latter's "$11.33 billed" line is outdated: the endpoint never started, so it cost $0.

## Tests

```bash
python -m pytest                       # no AWS access needed
ruff check aws agent scripts tests
```

## Teardown

```bash
python scripts/status.py                                   # what exists now (read-only)
python scripts/99_teardown.py --confirm DELETE_MTRL_DEMO
```

This deletes the endpoint (if any), the Bedrock imported models with their us-east-1 copy bucket and import role, the AgentCore runtime, the MLflow app, the ECR repository, the S3 bucket, both IAM roles and the budget. Until then, Bedrock storage costs about $1.95 a month for the imported model.

Left behind, to delete in the AWS console:

- **From the AgentCore starter toolkit:** the CodeBuild project `bedrock-agentcore-mtrl_support_agent-builder`, its `AmazonBedrockAgentCoreSDKCodeBuild-*` role, the `bedrock-agentcore-codebuild-sources-*` bucket and the CloudWatch log groups.
- **The model package record** in the SageMaker Model Registry (its weights go with the bucket).

The evidence in `artifacts/` and `reports/` is kept, so the offline steps still work afterwards.

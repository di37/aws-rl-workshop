"""Train with MultiTurnRLTrainer, as in the official guide (needs --confirm).

Approved size: 10 steps, each on 32 tickets x 4 rollouts, LoRA on
GPT-OSS-20B, with the AgentCore agent as the environment. Training is blocked
unless the baseline shows reward variance. A recorded job is re-attached,
never submitted twice. When the job completes, its summary (including the
job's own training config) and per-step metrics are saved to artifacts/.
"""

# region Imports & setup
from __future__ import annotations

import _bootstrap as bootstrap

from aws.reporting.report_tables import training_curve
from aws.rl.training import TrainingStage

TIMEOUT_SECONDS = 4 * 3600
# endregion


# region Entry point
def main() -> None:
    """Attaches to or submits the training job, waits, and records its metrics."""
    args = bootstrap.parse_args(__doc__.splitlines()[0], TrainingStage.CONFIRMATION)
    bootstrap.banner("Training (06)", "MultiTurnRLTrainer: 10 steps x 32 tickets x 4 rollouts")
    workflow = bootstrap.load_workflow()
    training = workflow.training_stage()
    job = training.run_or_attach(args.confirm)
    if job is None:
        print(f"No training job recorded. Pass --confirm {TrainingStage.CONFIRMATION} to submit one.")
        return
    print(f"[{bootstrap.now()}] job {job.job_name}: {job.job_status}", flush=True)
    job = training.wait(job, timeout_seconds=TIMEOUT_SECONDS)
    summary = training.summary(job)
    bootstrap.show_fields(summary, ("job_name", "job_status", "failure_reason", "duration_minutes",
                                    "output_model_package_arn", "billable_token_usage", "progress_info"))
    if job.job_status != "Completed":
        raise RuntimeError(f"Training ended with status {job.job_status}. To archive this job and "
                           f"submit a new one: --confirm {TrainingStage.RETRY_CONFIRMATION}")
    steps = (workflow.store.load(TrainingStage.METRICS_RECORD) or {}).get("steps") or training.record_metrics(job)
    bootstrap.show(training_curve({"steps": steps}))


if __name__ == "__main__":
    bootstrap.run(main)
# endregion

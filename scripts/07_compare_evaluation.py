"""Evaluate base vs fine-tuned on the held-out tickets (needs --confirm).

One MultiTurnRLEvaluator(model=<training job>, evaluate_base_model=True)
pipeline, as in the official guide: both models, the same 32 held-out tickets,
2 rollouts each, through the same agent. Each model's metrics come from its
own MLflow run. A recorded pipeline is re-attached, never submitted twice.
"""

# region Imports & setup
from __future__ import annotations

import _bootstrap as bootstrap

from aws.reporting.report_tables import evaluation_comparison
from aws.rl.evaluation import EvaluationStage

TIMEOUT_SECONDS = 2 * 3600
# endregion


# region Entry point
def main() -> None:
    """Attaches to or submits the comparison pipeline and prints both models' metrics."""
    args = bootstrap.parse_args(__doc__.splitlines()[0], EvaluationStage.CONFIRMATION)
    bootstrap.banner("Comparison evaluation (07)", "base vs fine-tuned, 32 held-out tickets x 2 rollouts")
    workflow = bootstrap.load_workflow()
    job = workflow.training_stage().run_or_attach("")
    trained = job if job is not None and job.job_status == "Completed" else None
    evaluation = workflow.evaluation_stage()
    execution_arn = evaluation.run_or_attach(args.confirm, trained)
    if execution_arn is None:
        print(f"No comparison recorded. Pass --confirm {EvaluationStage.CONFIRMATION} "
              "after training completes.")
        return
    print("Pipeline execution:", execution_arn)
    results = workflow.store.load(EvaluationStage.METRICS)
    if results is None:
        status = evaluation.wait(execution_arn, timeout_seconds=TIMEOUT_SECONDS)
        if status != "Succeeded":
            raise RuntimeError(f"Comparison pipeline ended with status {status}.")
        results = evaluation.results(execution_arn)
    bootstrap.show(evaluation_comparison(results))
    print("64 rollouts per model: differences under about 0.1 are within sampling noise.")


if __name__ == "__main__":
    bootstrap.run(main)
# endregion

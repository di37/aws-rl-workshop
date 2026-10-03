"""Download the conversations SageMaker recorded during the comparison (read-only).

The comparison evaluation stored every rollout of both models as an MLflow
trace: ticket, reasoning, each tool call, and the reward. This reads the raw
trace files from S3, pairs both models' conversations by held-out ticket, and
saves artifacts/evaluation_trajectories.json. An existing record is kept
unless --refresh is given.
"""

# region Imports & setup
from __future__ import annotations

import _bootstrap as bootstrap
import pandas as pd

from aws.records.evidence import EvidenceStore
from aws.reporting.invariants import check_trajectories_match_metrics
from aws.rl.evaluation import EvaluationStage
from aws.rl.trajectories import TrajectoryReader, comparison_record

# endregion


# region Entry point
def main() -> None:
    """Reads, pairs, and saves the recorded conversations, then cross-checks them."""
    args = bootstrap.parse_args(__doc__.splitlines()[0], extra=lambda parser: parser.add_argument(
        "--refresh", action="store_true", help="download again even when a record exists"))
    bootstrap.banner("Recorded conversations (08)", "MLflow traces of the comparison evaluation")
    store = EvidenceStore(bootstrap.ARTIFACTS_DIR)
    recorded = store.load(TrajectoryReader.RECORD)
    if recorded is None or args.refresh:
        results = store.load(EvaluationStage.METRICS)
        if results is None:
            raise FileNotFoundError("No comparison metrics recorded: run scripts/07_compare_evaluation.py first.")
        workflow = bootstrap.load_workflow()
        reader = TrajectoryReader.create(workflow.session, workflow.state.mlflow_app_arn)
        order = pd.read_csv(bootstrap.DATA_DIR / "evaluation_prompts.csv")["prompt"].tolist()
        recorded = comparison_record(reader, results, order)
        store.save(TrajectoryReader.RECORD, recorded)
    summary = recorded["summary"]
    print(f"{len(recorded['pairs'])} held-out tickets: fine-tuned higher on {summary['improved']}, "
          f"same on {summary['same']}, lower on {summary['worse']} (mean reward per ticket)")
    ok, detail = check_trajectories_match_metrics(store)
    print(f"Cross-check with evaluation metrics: {'PASS' if ok else 'FAIL'} ({detail})")


if __name__ == "__main__":
    bootstrap.run(main)
# endregion

"""Baseline: evaluate the untrained model through the agent (needs --confirm).

One MultiTurnRLEvaluator run of base GPT-OSS-20B on the 32 held-out tickets,
2 rollouts each, through the deployed agent. Its reward variance gates
training (no variance means no learning signal), and its billed tokens
calibrate the budget guard. A recorded baseline is re-attached, never
submitted twice.
"""

# region Imports & setup
from __future__ import annotations

import _bootstrap as bootstrap

from aws.reporting.report_tables import KEY_METRICS
from aws.workflow import MtrlDemoWorkflow

# endregion


# region Entry point
def main() -> None:
    """Shows, finishes, or (when confirmed) submits the baseline evaluation."""
    args = bootstrap.parse_args(__doc__.splitlines()[0], MtrlDemoWorkflow.BASE_EVAL_CONFIRMATION)
    bootstrap.banner("Baseline evaluation (05)", "base model, 32 held-out tickets x 2 rollouts")
    evidence = bootstrap.load_workflow().run_or_attach_base_evaluation(args.confirm)
    if evidence is None:
        print(f"No baseline recorded. Pass --confirm {MtrlDemoWorkflow.BASE_EVAL_CONFIRMATION} to run one.")
        return
    print("Pipeline execution:", evidence["execution"]["arn"])
    metrics = evidence["metrics"] or {}
    bootstrap.show([{"metric": label, "base model": metrics.get(key)} for key, label in KEY_METRICS.items()])


if __name__ == "__main__":
    bootstrap.run(main)
# endregion

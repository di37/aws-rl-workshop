"""Live inference before vs after training on held-out tickets (needs --confirm).

Runs the agent used in training (SupportRolloutAgent.run_episode: same prompt,
tool, and environment) on the first 6 held-out tickets with both models and
one shared action-order seed (2026):
- before: base GPT-OSS-20B on Amazon Bedrock, the open weights training started from
- after: the fine-tuned model on the SageMaker endpoint if it is InService,
  otherwise the Bedrock imported model (option 2)
Also asks the fine-tuned model one free-form question. The first live run is
the study's record; a later run never overwrites it and is saved beside it,
named by UTC time. The budget gate runs first. Without --confirm, shows the
recorded run.
"""

# region Imports & setup
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import _bootstrap as bootstrap
import pandas as pd

from aws.deploy.bedrock_import import (
    BedrockImporter,
    bedrock_runtime,
    chat_with_imported_model,
    imported_model,
)
from aws.deploy.inference import (
    AFTER_ENDPOINT_LABEL,
    AFTER_IMPORT_LABEL,
    BEFORE_LABEL,
    CHAT_QUESTION,
    InferenceComparison,
    bedrock_base_model,
    endpoint_model,
)
from aws.guards import require_phrase
from aws.records.evidence import EvidenceStore
from aws.reporting.report_tables import live_inference

# endregion


# region Models
def fine_tuned_target(workflow: Any, comparison: InferenceComparison) -> tuple[str, Any, Callable[[], str]]:
    """Picks the running fine-tuned deployment: the endpoint first, else the Bedrock import.

    Args:
        workflow: Region-pinned workflow facade.
        comparison: Comparison that records the transcripts and the chat reply.

    Returns:
        Label, Strands model, and a function that records one chat reply and returns its file.

    Raises:
        RuntimeError: If neither deployment is ready.
    """
    config, session = bootstrap.CONFIG, workflow.session
    if (workflow.store.load("endpoint.json") or {}).get("status") == "InService":
        component = workflow.deployment_stage().component_names()["fine_tuned"]

        def endpoint_chat() -> str:
            reply = comparison.chat(component, CHAT_QUESTION, record=False)
            return comparison.record_chat(component, CHAT_QUESTION, reply)

        return AFTER_ENDPOINT_LABEL, endpoint_model(config, component, session), endpoint_chat
    imported = workflow.store.load(BedrockImporter.RECORD) or {}
    if imported.get("status") != "Completed":
        raise RuntimeError("No fine-tuned deployment is ready: run scripts/09a or scripts/09b first.")
    arn = imported["imported_model_arn"]

    def imported_chat() -> str:
        reply = chat_with_imported_model(bedrock_runtime(session), arn, CHAT_QUESTION, config.sampling_max_tokens)
        return comparison.record_chat(AFTER_IMPORT_LABEL, CHAT_QUESTION, reply)

    return AFTER_IMPORT_LABEL, imported_model(config, arn, session), imported_chat


def run_live() -> tuple[dict[str, Any], str]:
    """Replays the held-out tickets with both models, after the budget gate.

    Returns:
        The results (with the evidence file written) and the chat evidence file.
    """
    workflow = bootstrap.load_workflow()
    comparison = workflow.inference()
    label, model, chat = fine_tuned_target(workflow, comparison)
    tickets = pd.read_csv(bootstrap.DATA_DIR / "evaluation_prompts.csv")["prompt"].tolist()
    models = {"before": (BEFORE_LABEL, bedrock_base_model(bootstrap.CONFIG, workflow.session)),
              "after": (label, model)}
    results = comparison.compare(models, tickets[: bootstrap.CONFIG.inference_ticket_count],
                                 seed=bootstrap.CONFIG.inference_seed)
    return results, chat()
# endregion


# region Entry point
def main() -> None:
    """Runs live inference when confirmed, then shows the recorded results."""
    args = bootstrap.parse_args(__doc__.splitlines()[0], InferenceComparison.LIVE_CONFIRMATION)
    bootstrap.banner("Live inference (10)", "same agent, same tickets, same seed: before vs after training")
    store = EvidenceStore(bootstrap.ARTIFACTS_DIR)
    chat_file = InferenceComparison.CHAT_RECORD
    if args.confirm:
        require_phrase(args.confirm, InferenceComparison.LIVE_CONFIRMATION, "live inference")
        results, chat_file = run_live()
        print(f"Saved as artifacts/{results['saved_as']} (the run of record is never overwritten).")
    else:
        results = store.load(InferenceComparison.RECORD)
    if results is None:
        print(f"No live inference recorded. Pass --confirm {InferenceComparison.LIVE_CONFIRMATION} to run it.")
        return
    rows = live_inference(results)
    for side in ("before", "after"):
        solved = sum(1 for row in rows if row[f"{side}_reward"] == 1.0)
        print(f"{results['labels'][side]}: {solved} of {len(rows)} tickets fully solved")
    bootstrap.show([{key: row[key] for key in ("ticket", "before_reward", "after_reward", "after_actions")}
                    for row in rows])
    chat = store.load(chat_file) or {}
    if chat.get("reply"):
        print(f"\nQ: {chat['message']}\nFine-tuned model: {chat['reply'].strip()[:600]}")


if __name__ == "__main__":
    bootstrap.run(main)
# endregion

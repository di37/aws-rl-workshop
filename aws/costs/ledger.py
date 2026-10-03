"""Collects what AWS actually billed for this demo from recorded evidence."""

from __future__ import annotations

from typing import Any

from aws.records.evidence import EvidenceStore
from aws.rl.pipelines import PipelineInspector

_FIRST_ATTEMPT = "pre_fix"
"""Suffix of the archived first baseline, run before the agent fix."""
_COMPARISON_LABELS = (("base", "Comparison: base model"), ("fine_tuned", "Comparison: fine-tuned model"))


class SpendLedger:
    """Turns recorded jobs into billed-usage records for the budget guard."""

    def __init__(self, store: EvidenceStore, inspector: PipelineInspector) -> None:
        """Initializes the ledger.

        Args:
            store: Evidence store with job and endpoint records.
            inspector: Reads billed token usage of evaluation jobs.
        """
        self.store = store
        self.inspector = inspector

    def measured_reference(self) -> dict[str, int]:
        """Returns billed usage of the latest base evaluation and its rollouts.

        Returns:
            ``prefill`` and ``sample`` token counts over ``rollouts`` rollouts.

        Raises:
            RuntimeError: If no base evaluation has been recorded.
        """
        usage = self._evaluation_usage("base_evaluation.json")
        metrics = self.store.load("base_evaluation_metrics.json")
        if usage is None or metrics is None:
            raise RuntimeError("Run the base evaluation before projecting costs.")
        rollouts = int(
            metrics["eval/reward/num_prompts"] * metrics["eval/reward/rollouts_per_prompt"]
        )
        return {
            "prefill": int(usage.get("PrefillTokenCount", 0)),
            "sample": int(usage.get("SampleTokenCount", 0)),
            "rollouts": rollouts,
        }

    def spent(self) -> list[dict[str, Any]]:
        """Lists every billable stage that has already run.

        Returns:
            Records with ``label`` plus either ``component`` and ``usage``
            (token-priced) or ``usd`` (endpoint uptime).
        """
        records: list[dict[str, Any]] = []
        baselines = [(_baseline_label(name), name) for name in self.store.names("base_evaluation.*.json")]
        for label, name in [*baselines, ("Base evaluation", "base_evaluation.json")]:
            usage = self._evaluation_usage(name)
            if usage is not None:
                records.append({"label": label, "component": "Evaluation", "usage": usage})
        for name in self.store.names("training_job.*.json"):
            earlier = self.store.load(name) or {}
            if earlier.get("billable_token_usage"):
                records.append(
                    {"label": f"Training (earlier attempt {earlier.get('job_name')})",
                     "component": "Finetuning", "usage": earlier["billable_token_usage"]}
                )
        training = self.store.load("training_job.json") or {}
        if training.get("billable_token_usage"):
            records.append(
                {"label": "Training", "component": "Finetuning",
                 "usage": training["billable_token_usage"]}
            )
        comparison = self.store.load("comparison_evaluation_metrics.json") or {}
        for key, label in _COMPARISON_LABELS:
            if key in comparison:
                records.append(
                    {"label": label, "component": "Evaluation",
                     "usage": comparison[key].get("billable_token_usage") or {}}
                )
        for name in self.store.names("endpoint.*.json"):
            earlier = self.store.load(name) or {}
            if "billed_usd" in earlier:
                records.append({"label": "Endpoint uptime (earlier attempt)", "usd": earlier["billed_usd"]})
        for item in (self.store.load("extra_costs.json") or {}).get("items", []):
            usd = item.get("usd")
            if isinstance(usd, bool) or not isinstance(usd, (int, float)) or usd < 0:
                raise ValueError(f"Invalid extra cost for {item.get('label')!r}: {usd!r}")
            records.append({"label": item["label"], "usd": usd})
        endpoint = self.store.load("endpoint.json") or {}
        if endpoint.get("status") == "Deleted" and "billed_usd" in endpoint:
            records.append({"label": "Endpoint uptime", "usd": endpoint["billed_usd"]})
        return records

    def remaining(self) -> dict[str, bool]:
        """Reports which billable stages have not run yet.

        A live endpoint still counts as planned at its full lifetime until it
        is deleted and its real uptime cost is recorded.

        Returns:
            Flags for ``training``, ``comparison``, and ``endpoint``.
        """
        training = self.store.load("training_job.json") or {}
        endpoint = self.store.load("endpoint.json") or {}
        return {
            "training": training.get("job_status") != "Completed",
            "comparison": self.store.load("comparison_evaluation_metrics.json") is None,
            "endpoint": endpoint.get("status") != "Deleted",
        }

    def _evaluation_usage(self, record_name: str) -> dict[str, int] | None:
        """Reads billed tokens of the base-model step of a recorded pipeline.

        Args:
            record_name: Evidence file holding the pipeline execution ARN.

        Returns:
            ``BillableTokenUsage``, or None when the record does not exist.
        """
        record = self.store.load(record_name)
        if not record:
            return None
        job = self.inspector.step_jobs(record["arn"]).get("EvaluateBaseModel")
        if job is None:
            return None
        return self.inspector.eval_job_details(job).get("billable_token_usage")


def _baseline_label(name: str) -> str:
    """Labels an archived baseline record by the suffix in its file name.

    Args:
        name: File name such as ``base_evaluation.pre_fix.json``.

    Returns:
        A cost-line label for that earlier attempt.
    """
    suffix = name.removeprefix("base_evaluation.").removesuffix(".json")
    return "Base evaluation (first attempt)" if suffix == _FIRST_ATTEMPT else f"Base evaluation (earlier attempt {suffix})"

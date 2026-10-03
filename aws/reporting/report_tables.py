"""Turns recorded evidence into the report's tables, without calling AWS.

Every function reads evidence already saved in ``artifacts/`` and returns plain
rows, so ``scripts/12_make_report_tables_and_figures.py`` can rebuild all
tables offline and the invariants can check them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from aws.records.evidence import EvidenceStore
from aws.rl.trajectories import mean_reward, ticket_change

KEY_METRICS = {
    "eval/reward/pass_at_1": "Ticket fully solved (pass@1)",
    "eval/reward/pass_at_2": "Solved in 1 of 2 tries (pass@2)",
    "eval/reward/mean": "Mean reward (with partial credit)",
    "eval/reward/succeeded_rollouts": "Rollouts fully solved",
    "eval/reward/failed_rollouts": "Rollouts not fully solved",
    "eval/turns/mean": "Model calls per rollout",
}
"""Headline evaluation metrics in report order, with plain-language labels."""

TABLE_FILES = (
    "evaluation_comparison.csv",
    "baseline_evaluations.csv",
    "training_steps.csv",
    "per_ticket_rewards.csv",
    "live_inference.csv",
    "run_of_record.csv",
    "cost_accounting.csv",
)
FIGURE_FILES = (
    "training_reward_curve.png",
    "evaluation_before_after.png",
    "per_ticket_before_after.png",
    "live_inference_before_after.png",
)
PRODUCED_BY = {
    "base_evaluation_metrics.json": "05_base_evaluation.py",
    "training_metrics.json": "06_train.py",
    "comparison_evaluation_metrics.json": "07_compare_evaluation.py",
    "evaluation_trajectories.json": "08_recorded_trajectories.py",
    "inference_before_after.json": "10_live_inference.py",
    "cost_accounting.json": "11_snapshot_costs.py",
}
"""Evidence file to the script that produces it, for clear errors when one is missing."""
SOLVED = 1.0
"""Reward of a fully solved ticket (the evaluation's success threshold)."""
_CUSTOMER_PREFIX = "Customer says: "


def customer_text(ticket: str) -> str:
    """Returns the customer's words from a full ticket prompt."""
    return ticket.split(_CUSTOMER_PREFIX)[-1].strip()


def require(store: EvidenceStore, name: str) -> Any:
    """Loads one evidence file, naming the step that produces it when it is missing.

    Raises:
        FileNotFoundError: If the evidence file does not exist.
    """
    record = store.load(name)
    if record is None:
        raise FileNotFoundError(f"artifacts/{name} is missing: run scripts/{PRODUCED_BY[name]} first.")
    return record


def build_tables(store: EvidenceStore) -> dict[str, list[dict[str, Any]]]:
    """Builds every report table from the evidence, keyed by file name (``TABLE_FILES`` order).

    Args:
        store: Evidence store of the run.

    Returns:
        Table file name to rows.
    """
    return {
        "evaluation_comparison.csv": evaluation_comparison(require(store, "comparison_evaluation_metrics.json")),
        "baseline_evaluations.csv": baseline_history(store.load("base_evaluation_metrics.pre_fix.json"),
                                                     require(store, "base_evaluation_metrics.json")),
        "training_steps.csv": training_curve(require(store, "training_metrics.json")),
        "per_ticket_rewards.csv": per_ticket_rewards(require(store, "evaluation_trajectories.json")),
        "live_inference.csv": live_inference(require(store, "inference_before_after.json")),
        "run_of_record.csv": run_of_record(store),
        "cost_accounting.csv": cost_table(require(store, "cost_accounting.json")),
    }


def evaluation_comparison(results: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Builds base versus fine-tuned rows for the headline metrics.

    Args:
        results: Contents of ``comparison_evaluation_metrics.json``.

    Returns:
        One row per key metric with both values and their difference.
    """
    base = results.get("base", {}).get("metrics", {})
    tuned = results.get("fine_tuned", {}).get("metrics", {})
    return [
        {"metric": key, "label": label, "base": base.get(key), "fine_tuned": tuned.get(key),
         "delta": _delta(base.get(key), tuned.get(key))}
        for key, label in KEY_METRICS.items()
    ]


def baseline_history(first: Mapping[str, Any] | None, current: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Shows the first baseline attempt (before the agent fix) next to the baseline of record.

    Args:
        first: Metrics of the first attempt, or None when there was none.
        current: Metrics of the baseline of record.

    Returns:
        One row per key metric.
    """
    return [
        {"metric": key, "label": label, "first_attempt": (first or {}).get(key), "baseline": current.get(key)}
        for key, label in KEY_METRICS.items()
    ]


def training_curve(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Builds per-step training rows from ``training_metrics.json``.

    Args:
        record: Recorded training metrics with a ``steps`` list.

    Returns:
        Rows of step, mean reward, mean turns, tokens, and trajectories, by step.
    """
    rows = [
        {"step": int(step["step"]),
         "mean_reward": step.get("rollout/reward/mean"),
         "mean_turns": step.get("rollout/turns/mean"),
         "total_tokens": _as_int(step.get("training/total_tokens")),
         "trajectories": _as_int(step.get("training/num_trajectories"))}
        for step in record.get("steps", [])
    ]
    return sorted(rows, key=lambda row: row["step"])


def per_ticket_rewards(recorded: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Builds per-ticket before/after rewards from the recorded evaluation traces.

    Args:
        recorded: Contents of ``evaluation_trajectories.json``.

    Returns:
        Rows of ticket text, each attempt's reward, means, and the change.
    """
    return [
        {"ticket": customer_text(pair["ticket"]),
         "before_rewards": _attempts(pair["before"]),
         "after_rewards": _attempts(pair["after"]),
         "before_mean": _rounded(mean_reward(pair["before"])),
         "after_mean": _rounded(mean_reward(pair["after"])),
         "change": ticket_change(pair)}
        for pair in recorded.get("pairs", [])
    ]


def live_inference(results: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Builds per-ticket rows of the live before/after inference run.

    Args:
        results: Contents of ``inference_before_after.json``.

    Returns:
        Rows of ticket text and, per model, reward, actions, and any error.
    """
    rows = []
    for entry in results.get("tickets", []):
        row: dict[str, Any] = {"ticket": customer_text(entry["ticket"])}
        for side in ("before", "after"):
            outcome = entry.get(side) or {}
            row[f"{side}_reward"] = outcome.get("reward")
            row[f"{side}_actions"] = " > ".join((outcome.get("metrics") or {}).get("actions_taken", []))
            row[f"{side}_error"] = outcome.get("error")
        rows.append(row)
    return rows


def run_of_record(store: EvidenceStore) -> list[dict[str, Any]]:
    """Lists every recorded stage of the run with its identifier and outcome.

    Args:
        store: Evidence store of the run.

    Returns:
        Rows of stage, identifier, status, and detail, in pipeline order.
    """
    rows = [
        _baseline_row("Base evaluation (first attempt)", store, "base_evaluation.pre_fix.json",
                      "base_evaluation_metrics.pre_fix.json"),
        _baseline_row("Base evaluation", store, "base_evaluation.json", "base_evaluation_metrics.json"),
    ]
    for name in store.names("training_job.*.json"):
        rows.append(_training_row("Training (rejected attempt)", store.load(name)))
    rows.append(_training_row("Training", store.load("training_job.json")))
    rows.append(_comparison_row(store))
    for name in store.names("endpoint.*.json"):
        rows.append(_endpoint_row(store.load(name), "SageMaker endpoint (earlier attempt)"))
    rows.append(_endpoint_row(store.load("endpoint.json")))
    rows.append(_import_row(store.load("bedrock_import.json")))
    rows.append(_live_row(store.load("inference_before_after.json")))
    return [row for row in rows if row is not None]


def headline_results(store: EvidenceStore) -> dict[str, Any]:
    """Collects the study's headline numbers from the evidence.

    Args:
        store: Evidence store of the run.

    Returns:
        Baseline, comparison, training-curve, and live-inference results.
    """
    comparison = store.load("comparison_evaluation_metrics.json") or {}
    curve = training_curve(store.load("training_metrics.json") or {})
    live = (store.load("inference_before_after.json") or {}).get("tickets", [])

    def metric(key: str) -> dict[str, Any]:
        return {side: comparison.get(side, {}).get("metrics", {}).get(key) for side in ("base", "fine_tuned")}

    return {
        "baseline_pass_at_1": (store.load("base_evaluation_metrics.json") or {}).get("eval/reward/pass_at_1"),
        "comparison_pass_at_1": metric("eval/reward/pass_at_1"),
        "comparison_mean_reward": metric("eval/reward/mean"),
        "recorded_ticket_changes": (store.load("evaluation_trajectories.json") or {}).get("summary"),
        "training_mean_reward": ({"first_step": curve[0]["mean_reward"], "last_step": curve[-1]["mean_reward"]}
                                 if curve else None),
        "live_inference_solved": {"before": _solved(live, "before"), "after": _solved(live, "after"),
                                  "tickets": len(live)},
    }


def cost_table(snapshot: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Builds the cost rows from ``cost_accounting.json``.

    Args:
        snapshot: Cost snapshot from :func:`cost_snapshot.build_cost_snapshot`.

    Returns:
        Rows of item, USD, and source.
    """
    return [{"item": item["item"], "usd": item["usd"], "source": item["source"]}
            for item in snapshot.get("items", [])]


def _baseline_row(stage: str, store: EvidenceStore, record_name: str, metrics_name: str) -> dict | None:
    """Describes one recorded base evaluation, or None when absent."""
    record, metrics = store.load(record_name), store.load(metrics_name) or {}
    if not record:
        return None
    return {"stage": stage, "identifier": _execution_id(record.get("arn")),
            "status": record.get("status") or ("Succeeded" if metrics else "no metrics recorded"),
            "detail": f"pass@1 {metrics.get('eval/reward/pass_at_1')}, mean reward {metrics.get('eval/reward/mean')}"}


def _training_row(stage: str, record: Mapping[str, Any] | None) -> dict | None:
    """Describes one recorded training job, or None when absent."""
    if not record:
        return None
    progress = record.get("progress_info") or {}
    detail = (record.get("failure_reason") or "")[:160] or (
        f"{progress.get('CurrentStep')}/{progress.get('MaxSteps')} steps in {record.get('duration_minutes')} min; "
        f"model package {record.get('output_model_package_arn')}")
    return {"stage": stage, "identifier": record.get("job_name"), "status": record.get("job_status"),
            "detail": detail}


def _comparison_row(store: EvidenceStore) -> dict | None:
    """Describes the recorded comparison pipeline, or None when absent."""
    record = store.load("comparison_evaluation.json")
    if not record:
        return None
    rows = {row["metric"]: row for row in evaluation_comparison(
        store.load("comparison_evaluation_metrics.json") or {})}
    pass_1 = rows["eval/reward/pass_at_1"]
    return {"stage": "Comparison evaluation", "identifier": _execution_id(record.get("arn")),
            "status": record.get("status"),
            "detail": f"pass@1 base {pass_1['base']} -> fine-tuned {pass_1['fine_tuned']}"}


def _endpoint_row(record: Mapping[str, Any] | None, stage: str = "SageMaker endpoint") -> dict | None:
    """Describes one recorded endpoint attempt, or None when absent."""
    if not record:
        return None
    reason = (record.get("failure_reason") or "").split("Failure Reason: ")[-1].split(". ")[0].rstrip(".")
    billed = f"billed ${record.get('billed_usd')}"
    return {"stage": stage, "identifier": record.get("endpoint_name"),
            "status": record.get("status"), "detail": f"{reason}; {billed}" if reason else billed}


def _import_row(record: Mapping[str, Any] | None) -> dict | None:
    """Describes the recorded Bedrock import, or None when absent."""
    if not record:
        return None
    return {"stage": "Bedrock import", "identifier": record.get("job_name"), "status": record.get("status"),
            "detail": f"imported model {record.get('imported_model_arn')}"}


def _live_row(record: Mapping[str, Any] | None) -> dict | None:
    """Describes the recorded live inference run, or None when absent."""
    if not record:
        return None
    tickets = record.get("tickets", [])
    return {"stage": "Live inference", "identifier": f"seed {record.get('seed')}", "status": "Recorded",
            "detail": f"fully solved: before {_solved(tickets, 'before')}/{len(tickets)}, "
                      f"after {_solved(tickets, 'after')}/{len(tickets)}"}


def _solved(tickets: Sequence[Mapping[str, Any]], side: str) -> int:
    """Counts tickets fully solved by one side of a live comparison."""
    return sum(1 for entry in tickets if (entry.get(side) or {}).get("reward") == SOLVED)


def _attempts(conversations: Sequence[Mapping[str, Any]]) -> str:
    """Formats each attempt's reward, marking unknown rewards."""
    return "; ".join("n/a" if c.get("reward") is None else str(c["reward"]) for c in conversations)


def _delta(before: Any, after: Any) -> float | None:
    """Returns after minus before when both are numbers."""
    if isinstance(before, (int, float)) and isinstance(after, (int, float)):
        return round(after - before, 4)
    return None


def _rounded(value: float | None) -> float | None:
    """Rounds a mean to four decimals, keeping None."""
    return None if value is None else round(value, 4)


def _as_int(value: Any) -> int | None:
    """Converts a numeric metric to int, keeping None."""
    return None if value is None else int(value)


def _execution_id(arn: str | None) -> str | None:
    """Returns the last segment of a pipeline execution ARN."""
    return arn.rsplit("/", 1)[-1] if arn else None

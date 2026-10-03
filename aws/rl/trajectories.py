"""Real before/after conversations recorded by SageMaker during evaluation.

The comparison evaluation ran the base and the fine-tuned model through the
same agent on the same held-out tickets, and SageMaker stored every rollout as
an MLflow trace (prompt, model reasoning, tool calls, reward). These traces are
genuine SageMaker inference results for both models and need no endpoint.

The raw ``traces.json`` artifacts are read directly from S3 because SageMaker
writes UUID span IDs that the local MLflow client cannot parse.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

_INITIAL_ENVIRONMENT = "\n\nInitial environment"


class TrajectoryReader:
    """Lists and parses the evaluation traces of one MLflow run."""

    RECORD = "evaluation_trajectories.json"

    def __init__(
        self,
        list_traces: Callable[[str, str], list[str]],
        read_json: Callable[[str], dict[str, Any]],
        experiment_of: Callable[[str], str],
    ) -> None:
        """Initializes the reader with injectable MLflow and S3 access.

        Args:
            list_traces: Returns the ``traces.json`` S3 URIs for (experiment, run).
            read_json: Downloads and parses one ``traces.json`` file.
            experiment_of: Returns the MLflow experiment ID of a run.
        """
        self._list_traces = list_traces
        self._read_json = read_json
        self._experiment_of = experiment_of

    @classmethod
    def create(cls, session: Any, mlflow_arn: str) -> TrajectoryReader:
        """Builds a reader wired to the managed MLflow app and S3.

        Args:
            session: Boto3 session pinned to the demo region.
            mlflow_arn: Managed MLflow app ARN used as tracking URI.

        Returns:
            A ready reader.
        """
        import mlflow
        from mlflow.tracing.client import TracingClient
        from mlflow.tracking import MlflowClient

        mlflow.set_tracking_uri(mlflow_arn)
        s3 = session.client("s3")

        def list_traces(experiment_id: str, run_id: str) -> list[str]:
            uris, skipped, token = [], 0, None
            while True:
                page = TracingClient().search_traces(
                    locations=[experiment_id], run_id=run_id, max_results=100,
                    include_spans=False, page_token=token,
                )
                page_uris, page_skipped = trace_uris(page)
                uris += page_uris
                skipped += page_skipped
                token = getattr(page, "token", None)
                if not token:
                    if skipped:
                        print(f"Skipped {skipped} traces without an S3 artifact location.")
                    return uris

        def read_json(uri: str) -> dict[str, Any]:
            bucket, key = uri.removeprefix("s3://").split("/", 1)
            return json.loads(s3.get_object(Bucket=bucket, Key=key)["Body"].read())

        def experiment_of(run_id: str) -> str:
            return MlflowClient().get_run(run_id).info.experiment_id

        return cls(list_traces, read_json, experiment_of)

    def read_run(self, run_id: str) -> list[dict[str, Any]]:
        """Reads every recorded conversation of one evaluation run.

        Args:
            run_id: MLflow run ID of one evaluation step.

        Returns:
            Parsed conversations.
        """
        uris = self._list_traces(self._experiment_of(run_id), run_id)
        return [self.parse(self._read_json(uri)) for uri in uris]

    @staticmethod
    def parse(trace: dict[str, Any]) -> dict[str, Any]:
        """Turns one raw trace into a readable conversation.

        Args:
            trace: Parsed ``traces.json`` content.

        Returns:
            Ticket, final reward, the model's first reasoning, and each action
            with whether it helped and what the environment answered.
        """
        spans = sorted(trace.get("spans", []), key=lambda span: span.get("start_time") or 0)
        conversation: dict[str, Any] = {"ticket": "", "reward": 0.0, "first_thought": "", "steps": []}
        for span in spans:
            attributes = span.get("attributes", {})
            name = span.get("name", "")
            if name == "trajectory":
                prompt = _json(attributes.get("mlflow.spanInputs")).get("prompt", "")
                conversation["ticket"] = prompt.split(_INITIAL_ENVIRONMENT)[0].strip()
                reward = attributes.get("reward")
                conversation["reward"] = float(reward) if reward not in (None, "") else None
            elif name == "turn-1":
                conversation["first_thought"] = _json(attributes.get("mlflow.spanOutputs")).get("thinking", "")
            elif name == "take_support_action":
                arguments = _json(_json(attributes.get("mlflow.spanInputs")).get("arguments"))
                result = _json(_json(attributes.get("mlflow.spanOutputs")).get("result"))
                conversation["steps"].append({
                    "action": arguments.get("action", ""),
                    "accepted": bool(result.get("accepted")),
                    "observation": result.get("observation", ""),
                })
        return conversation


def pair_by_ticket(
    before: Sequence[dict[str, Any]], after: Sequence[dict[str, Any]], order: Sequence[str]
) -> list[dict[str, Any]]:
    """Groups both models' conversations by ticket, in held-out dataset order.

    Args:
        before: Conversations of the base model.
        after: Conversations of the fine-tuned model.
        order: Tickets in held-out dataset order.

    Returns:
        One entry per ticket that both models attempted.
    """
    def grouped(conversations: Sequence[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
        groups: dict[str, list[dict[str, Any]]] = {}
        for conversation in conversations:
            groups.setdefault(conversation["ticket"].strip(), []).append(conversation)
        return groups

    base, tuned = grouped(before), grouped(after)
    pairs = [
        {"ticket": ticket, "before": base[ticket], "after": tuned[ticket]}
        for ticket in (t.strip() for t in order)
        if ticket in base and ticket in tuned
    ]
    if len(pairs) < len(order):
        print(f"Paired {len(pairs)} of {len(order)} tickets; the rest lack traces for one model.")
    return pairs


def comparison_record(
    reader: TrajectoryReader, results: Mapping[str, Any], order: Sequence[str]
) -> dict[str, Any]:
    """Reads both models' evaluation runs and pairs their conversations by ticket.

    Args:
        reader: Trace reader for the evaluation's MLflow app.
        results: Comparison metrics with each model's MLflow ``run_id``.
        order: Held-out tickets in dataset order.

    Returns:
        ``{"pairs": [...], "summary": {...}}``, as saved in ``RECORD``.
    """
    before = reader.read_run(results["base"]["run_id"])
    after = reader.read_run(results["fine_tuned"]["run_id"])
    pairs = pair_by_ticket(before, after, order)
    return {"pairs": pairs, "summary": compare_tickets(pairs)}


def compare_tickets(pairs: Sequence[dict[str, Any]]) -> dict[str, int]:
    """Counts tickets where the fine-tuned model scored higher, equal, or lower.

    Args:
        pairs: Output of :func:`pair_by_ticket`.

    Returns:
        ``{"improved": n, "same": n, "worse": n}`` by mean reward per ticket.
    """
    counts = {"improved": 0, "same": 0, "worse": 0}
    for pair in pairs:
        change = ticket_change(pair)
        if change is not None:
            counts[change] += 1
    return counts


def ticket_change(pair: dict[str, Any]) -> str | None:
    """Classifies one ticket by the change in its mean reward after training.

    Args:
        pair: One ticket with ``before`` and ``after`` conversations.

    Returns:
        ``"improved"``, ``"same"``, or ``"worse"``; None when a side has no reward.
    """
    before = mean_reward(pair["before"])
    after = mean_reward(pair["after"])
    if before is None or after is None:
        return None
    return "improved" if after > before else "worse" if after < before else "same"


def trace_uris(infos: Iterable[Any]) -> tuple[list[str], int]:
    """Turns MLflow trace infos into ``traces.json`` S3 URIs.

    Args:
        infos: Trace objects or trace infos carrying ``mlflow.artifactLocation``.

    Returns:
        The S3 URIs, and how many traces had no S3 location.
    """
    uris, skipped = [], 0
    for item in infos:
        location = getattr(getattr(item, "info", item), "tags", {}).get("mlflow.artifactLocation", "")
        if location.startswith("s3://"):
            uris.append(location.rstrip("/") + "/traces.json")
        else:
            skipped += 1
    return uris, skipped


def mean_reward(conversations: Sequence[dict[str, Any]]) -> float | None:
    """Averages known rewards; None when there are none."""
    known = [c["reward"] for c in conversations if c.get("reward") is not None]
    return sum(known) / len(known) if known else None


def _json(value: Any) -> dict[str, Any]:
    """Parses a JSON string attribute, tolerating dicts and missing values."""
    if isinstance(value, dict):
        return value
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}

"""Before/after inference with the same support agent.

Both models are driven by ``SupportRolloutAgent.run_episode``, the exact
function AgentCore runs during training; only the model provider differs:

- before training: base GPT-OSS-20B on Amazon Bedrock (same open weights)
- after training: the fine-tuned model on the deployed SageMaker endpoint, or
  the Bedrock imported model when no endpoint is running
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

from environment import SupportEnvironment

from aws.config import DemoConfig
from aws.records.evidence import EvidenceStore

BEFORE_LABEL = "Before training: base GPT-OSS-20B (Amazon Bedrock)"
AFTER_ENDPOINT_LABEL = "After training: fine-tuned model (SageMaker endpoint)"
AFTER_IMPORT_LABEL = "After training: fine-tuned model (Bedrock Custom Model Import)"
CHAT_QUESTION = "My internet is not working. What is the first troubleshooting step?"


class InferenceComparison:
    """Runs the same tickets through a before and an after model."""

    RECORD = "inference_before_after.json"
    CHAT_RECORD = "inference_chat.json"
    LIVE_CONFIRMATION = "CONFIRM_LIVE_INFERENCE_MTRL_DEMO"

    def __init__(
        self,
        store: EvidenceStore,
        episode_runner: Callable[[Any, str, int], dict[str, Any]],
        chat: Callable[[str, str], str],
        budget_gate: Callable[[], None] = lambda: None,
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        """Initializes the comparison with injectable model access.

        Args:
            store: Evidence store for the saved transcripts.
            episode_runner: Runs one agent episode for (model, ticket, seed).
            chat: Sends one chat message to an endpoint component, returns text.
            budget_gate: Raises PermissionError when the projected spend exceeds the cap.
            clock: Current UTC time, used to name later runs.
        """
        self.store = store
        self._episode = episode_runner
        self._chat = chat
        self._budget_gate = budget_gate
        self._clock = clock

    @classmethod
    def create(
        cls,
        config: DemoConfig,
        store: EvidenceStore,
        boto_session: Any,
        budget_gate: Callable[[], None] = lambda: None,
    ) -> InferenceComparison:
        """Builds the comparison wired to the real agent and endpoint.

        Args:
            config: Shared settings: endpoint name, region, token limit.
            store: Evidence store.
            boto_session: Boto3 session pinned to the demo region.
            budget_gate: Raises PermissionError when the projected spend exceeds the cap.

        Returns:
            A comparison that calls live models.
        """
        from app import SupportRolloutAgent

        def episode(model: Any, ticket: str, seed: int) -> dict[str, Any]:
            return SupportRolloutAgent.run_episode(model, ticket, seed, callback_handler=None)

        runtime = boto_session.client("sagemaker-runtime")

        def chat(component: str, message: str) -> str:
            body = {
                "model": "/opt/ml/model",
                "messages": [{"role": "user", "content": message}],
                "max_tokens": config.sampling_max_tokens,
                "stream": False,
            }
            response = runtime.invoke_endpoint(
                EndpointName=config.endpoint_name,
                InferenceComponentName=component,
                ContentType="application/json",
                Body=json.dumps(body),
            )
            payload = json.loads(response["Body"].read())
            return payload["choices"][0]["message"].get("content") or ""

        return cls(store, episode, chat, budget_gate)

    def compare(
        self, models: Mapping[str, tuple[str, Any]], tickets: Sequence[str], seed: int
    ) -> dict[str, Any]:
        """Runs every ticket through both models and saves the transcripts.

        The first run is the study's record. Later runs never overwrite it;
        each is saved beside it with a UTC timestamp in the file name.

        Args:
            models: ``{"before": (label, model), "after": (label, model)}``.
            tickets: Customer tickets to replay.
            seed: Action-order seed shared by both models for a fair comparison.

        Returns:
            Labels, and per ticket and model: reward, actions, and a
            turn-by-turn trace, or an ``error`` message when that model failed;
            plus ``saved_as``, the evidence file written.

        Raises:
            PermissionError: If the budget gate refuses the run.
        """
        self._budget_gate()
        results: dict[str, Any] = {
            "seed": seed,
            "labels": {key: label for key, (label, _) in models.items()},
            "tickets": [],
        }
        for ticket in tickets:
            entry: dict[str, Any] = {"ticket": ticket}
            for key, (_, model) in models.items():
                entry[key] = self._run_one(model, ticket, seed)
            results["tickets"].append(entry)
        return {**results, "saved_as": self._save_once(self.RECORD, results)}

    def chat(self, component: str, message: str, record: bool = True) -> str:
        """Sends one plain chat message to an endpoint component (the guide's invoke).

        Args:
            component: Inference component name.
            message: User message.
            record: Whether to save the exchange as evidence (see :meth:`record_chat`).

        Returns:
            The model's reply text.
        """
        reply = self._chat(component, message)
        if record:
            self.record_chat(component, message, reply)
        return reply

    def record_chat(self, component: str, message: str, reply: str) -> str:
        """Saves one chat exchange under the same write-once rule as ``compare``.

        Args:
            component: Model or inference component that replied.
            message: User message.
            reply: Model reply.

        Returns:
            The evidence file written.
        """
        return self._save_once(self.CHAT_RECORD, {"component": component, "message": message, "reply": reply})

    def _save_once(self, record: str, payload: Mapping[str, Any]) -> str:
        """Saves the first run as the record; later runs beside it, named by UTC time.

        Args:
            record: Record file name, such as ``inference_before_after.json``.
            payload: Evidence to save.

        Returns:
            The file name written.
        """
        name = record
        if self.store.load(record) is not None:
            name = record.replace(".json", f".{self._clock():%Y%m%dT%H%M%SZ}.json")
        self.store.save(name, payload)
        return name

    def _run_one(self, model: Any, ticket: str, seed: int) -> dict[str, Any]:
        """Runs one episode, capturing a failure instead of aborting the demo.

        Args:
            model: Strands model serving the policy.
            ticket: Customer ticket.
            seed: Action-order seed.

        Returns:
            Episode result with its trace, or an ``error`` entry.
        """
        try:
            result = self._episode(model, ticket, seed)
        except Exception as error:  # noqa: BLE001 - shown to the audience, not swallowed
            return {"error": f"{type(error).__name__}: {error}"}
        actions = result.get("metrics", {}).get("actions_taken", [])
        return {**result, "trace": trace(actions, seed)}


def endpoint_model(config: DemoConfig, component: str, boto_session: Any) -> Any:
    """Builds a Strands model for one component of the deployed endpoint.

    Args:
        config: Shared settings.
        component: Inference component name.
        boto_session: Boto3 session pinned to the demo region.

    Returns:
        ``SageMakerAIModel`` targeting the component.
    """
    from strands.models.sagemaker import SageMakerAIModel

    return SageMakerAIModel(
        endpoint_config={
            "endpoint_name": config.endpoint_name,
            "inference_component_name": component,
            "region_name": config.region,
        },
        payload_config={"max_tokens": config.sampling_max_tokens, "stream": False},
        boto_session=boto_session,
    )


def bedrock_base_model(config: DemoConfig, boto_session: Any) -> Any:
    """Builds a Strands model for base GPT-OSS-20B on Amazon Bedrock.

    Args:
        config: Shared settings, including the Bedrock model ID.
        boto_session: Boto3 session pinned to the demo region.

    Returns:
        ``BedrockModel`` serving the untrained base model.
    """
    from strands.models import BedrockModel

    return BedrockModel(
        boto_session=boto_session,
        model_id=config.bedrock_base_model_id,
        max_tokens=config.sampling_max_tokens,
    )


def trace(actions: Sequence[str], seed: int) -> list[dict[str, Any]]:
    """Replays actions in a fresh environment to show each turn's effect.

    Args:
        actions: Actions in the order the agent applied them.
        seed: Action-order seed used by the episode.

    Returns:
        One entry per turn: action, whether it helped, observation, reward so far.
    """
    environment = SupportEnvironment(action_order_seed=seed)
    steps = []
    for turn, action in enumerate(actions, start=1):
        result = environment.apply(action)
        steps.append(
            {
                "turn": turn,
                "action": action,
                "accepted": result.accepted,
                "observation": result.observation,
                "reward": environment.reward,
            }
        )
    return steps


def side_by_side(entry: Mapping[str, Any], labels: Mapping[str, str]) -> list[dict[str, str]]:
    """Builds a turn-by-turn table comparing both models on one ticket.

    Args:
        entry: One ticket entry from :meth:`InferenceComparison.compare`.
        labels: ``{"before": title, "after": title}`` column titles.

    Returns:
        Table rows: one per turn, then a final reward row.
    """
    columns = [(labels[key], entry[key]) for key in ("before", "after")]
    turns = max((len(result.get("trace", [])) for _, result in columns), default=0)
    rows = []
    for index in range(max(turns, 1)):
        row = {"Turn": str(index + 1)}
        for title, result in columns:
            row[title] = _describe_turn(result, index)
        rows.append(row)
    rows.append({"Turn": "Reward", **{title: _reward(result) for title, result in columns}})
    return rows


def _describe_turn(result: Mapping[str, Any], index: int) -> str:
    """Formats one turn of one model's trace."""
    if "error" in result:
        return f"error: {result['error']}" if index == 0 else ""
    steps = result.get("trace", [])
    if index >= len(steps):
        return ""
    step = steps[index]
    return f"{step['action']}  [{'correct' if step['accepted'] else 'wasted turn'}]"


def _reward(result: Mapping[str, Any]) -> str:
    """Formats a model's final reward, or a dash after an error."""
    return "-" if "error" in result else str(result.get("reward"))

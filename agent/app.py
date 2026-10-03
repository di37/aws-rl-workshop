"""Bedrock AgentCore entrypoint for SageMaker multi-turn RL rollouts."""

from __future__ import annotations

import json
import logging
import os
import secrets
from typing import Any
from urllib.parse import urlsplit

from bedrock_agentcore.runtime import BedrockAgentCoreApp
from pydantic import BaseModel, ConfigDict, Field
from sagemaker.core.token_generator import generate_token
from sagemaker.train.rft import sagemaker_rft_handler
from sagemaker.train.rft.adapters.strands import wrap_model
from strands import Agent, tool
from strands.models.openai import OpenAIModel

from environment import SupportEnvironment
from rollout_driver import RolloutDriver, RolloutOutcome


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = BedrockAgentCoreApp()

SYSTEM_PROMPT = """You are a customer-support troubleshooting agent.
Your objective is to restore the customer's internet connection.

Rules:
- Use the take_support_action tool exactly once for each decision.
- Choose only an action listed in available_actions.
- Use each tool result as the next observation.
- Continue until terminal=true or no turns remain.
- Do not invent tool results or claim success unless the tool reports it.
- If you cannot call the tool, reply with only the chosen action name.
- Keep text responses brief because success is measured by the environment.
"""


class RolloutInstance(BaseModel):
    """Validates the customer-provided portion of a rollout request."""

    model_config = ConfigDict(extra="allow")

    prompt: str = Field(min_length=1, max_length=4_000)


class PolicyModelConfig(BaseModel):
    """Stores the normalized OpenAI-compatible rollout model configuration."""

    base_url: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    api_key: str = Field(min_length=1)
    sampling_params: dict[str, Any] = Field(default_factory=dict)


class SupportRolloutAgent:
    """Coordinates policy inference, tool execution, and reward reporting."""

    @staticmethod
    def health() -> dict[str, str]:
        """Returns a side-effect-free deployment health response.

        Returns:
            Health status used before an RFT job exists.
        """
        return {"status": "healthy", "service": "mtrl-support-agent"}

    def run(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Executes one complete support rollout.

        Args:
            payload: SageMaker RFT rollout request.

        Returns:
            Reward, behavioral metrics, and a short model summary.

        Raises:
            ValueError: If the request lacks an RFT runtime endpoint.
        """
        prompt = self._extract_prompt(payload)
        metadata = payload.get("metadata") or {}
        policy_model = self._create_policy_model(payload, metadata)
        result = self.run_episode(policy_model, prompt, secrets.randbits(32))
        logger.info("Rollout result: %s", result)
        return result

    @classmethod
    def run_episode(
        cls,
        model: Any,
        prompt: str,
        action_order_seed: int,
        **agent_options: Any,
    ) -> dict[str, Any]:
        """Runs one support episode with any Strands model.

        AgentCore rollouts pass the SageMaker Job Runtime model; the notebook
        passes SageMaker endpoint models, so before/after inference uses this
        exact agent.

        Args:
            model: Strands model that serves the policy.
            prompt: Customer ticket.
            action_order_seed: Seed for the order in which actions are listed.
            **agent_options: Extra ``strands.Agent`` options, such as
                ``callback_handler=None`` to silence streaming output.

        Returns:
            Reward, behavioral metrics, and a short model summary.
        """
        environment = SupportEnvironment(action_order_seed=action_order_seed)
        agent = Agent(
            model=model,
            tools=[cls.create_support_tool(environment)],
            system_prompt=SYSTEM_PROMPT,
            **agent_options,
        )
        outcome = RolloutDriver(agent, environment).run(
            cls.opening_message(prompt, environment)
        )
        return cls.build_result(environment, outcome)

    @staticmethod
    def opening_message(prompt: str, environment: SupportEnvironment) -> str:
        """Builds the first user message of an episode.

        Args:
            prompt: Customer ticket.
            environment: Fresh environment whose state is shown to the policy.

        Returns:
            Ticket plus the initial observation as JSON.
        """
        return (
            f"{prompt}\n\nInitial environment:\n"
            f"{json.dumps(environment.current_observation())}\n"
            "Begin troubleshooting using the tool."
        )

    @staticmethod
    def _extract_prompt(payload: dict[str, Any]) -> str:
        """Extracts and validates the training prompt.

        Args:
            payload: SageMaker RFT rollout request.

        Returns:
            A non-empty prompt string.
        """
        instance = payload.get("instance")
        if isinstance(instance, dict):
            return RolloutInstance.model_validate(instance).prompt
        return RolloutInstance(prompt=payload.get("prompt")).prompt

    @staticmethod
    def _create_policy_model(payload: dict[str, Any], metadata: dict[str, Any]) -> Any:
        """Builds the tracked OpenAI-compatible policy client.

        Args:
            payload: SageMaker RFT rollout request.
            metadata: SageMaker-generated trajectory metadata.

        Returns:
            A Strands model wrapped with RFT tracking headers.

        Raises:
            ValueError: If no Job Runtime endpoint is available.
        """
        config = SupportRolloutAgent._resolve_policy_config(payload, metadata)
        model = OpenAIModel(
            model_id=config.model_id,
            client_args={
                "api_key": config.api_key,
                "base_url": config.base_url,
            },
            params=config.sampling_params,
        )
        return wrap_model(model)

    @staticmethod
    def _resolve_policy_config(
        payload: dict[str, Any],
        metadata: dict[str, Any],
    ) -> PolicyModelConfig:
        """Normalizes supported SageMaker rollout payload contracts.

        Args:
            payload: SageMaker RFT rollout request.
            metadata: SageMaker-generated trajectory metadata.

        Returns:
            Validated policy model connection settings.

        Raises:
            ValueError: If the endpoint or sampling parameters are unsafe.
        """
        region = metadata.get("region") or os.environ.get("AWS_REGION", "us-west-2")
        endpoint = (
            metadata.get("endpoint")
            or payload.get("modelEndpoint")
            or payload.get("model_endpoint")
            or os.environ.get("RFT_RUNTIME_ENDPOINT")
            or f"https://job-runtime.sagemaker.{region}.api.aws"
        )
        sampling_params = (
            payload.get("inferenceParams") or payload.get("inference_params") or {}
        )
        if not isinstance(sampling_params, dict):
            raise ValueError("Rollout inference parameters must be an object.")
        return PolicyModelConfig(
            base_url=SupportRolloutAgent._validate_runtime_endpoint(endpoint, region),
            model_id=payload.get("modelName") or payload.get("model_name") or "default",
            api_key=generate_token(region=region),
            sampling_params=SupportRolloutAgent._normalize_sampling_params(
                sampling_params
            ),
        )

    @staticmethod
    def _validate_runtime_endpoint(endpoint: object, region: str) -> str:
        """Validates and normalizes the SageMaker Job Runtime URL.

        Args:
            endpoint: Candidate endpoint supplied by SageMaker.
            region: AWS region associated with the rollout.

        Returns:
            An HTTPS OpenAI-compatible Job Runtime base URL.

        Raises:
            ValueError: If the URL could expose a bearer token to another host.
        """
        if not isinstance(endpoint, str):
            raise ValueError("The SageMaker Job Runtime endpoint must be a URL.")
        parsed = urlsplit(endpoint)
        expected_host = f"job-runtime.sagemaker.{region}.api.aws"
        if (
            parsed.scheme != "https"
            or parsed.hostname != expected_host
            or parsed.port is not None
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or parsed.path.rstrip("/") not in ("", "/v1")
        ):
            raise ValueError("Refusing an untrusted SageMaker Job Runtime endpoint.")
        return f"https://{expected_host}/v1"

    @staticmethod
    def _normalize_sampling_params(params: dict[str, Any]) -> dict[str, Any]:
        """Converts SageMaker camel-case inference keys for OpenAI clients.

        Args:
            params: Sampling parameters from a rollout request.

        Returns:
            A copy using OpenAI-compatible snake-case parameter names.
        """
        aliases = {"maxTokens": "max_tokens", "topP": "top_p"}
        return {aliases.get(key, key): value for key, value in params.items()}

    @staticmethod
    def create_support_tool(environment: SupportEnvironment) -> Any:
        """Creates a rollout-scoped tool bound to one environment.

        Args:
            environment: Isolated support state for this trajectory.

        Returns:
            A Strands tool callable.
        """

        @tool
        def take_support_action(action: str) -> dict[str, object]:
            """Execute one action from the current available-actions list.

            Args:
                action: Exact name of one action from available_actions.
            """
            return environment.apply(action).as_dict()

        return take_support_action

    @staticmethod
    def build_result(
        environment: SupportEnvironment, outcome: RolloutOutcome
    ) -> dict[str, Any]:
        """Builds the JSON response consumed by the RFT decorator.

        Args:
            environment: Final state of the support trajectory.
            outcome: Interaction statistics from the rollout driver.

        Returns:
            Reward and diagnostics for SageMaker and MLflow.
        """
        return {
            "reward": environment.reward,
            "metrics": {
                "task_completed": environment.completed,
                "progress_steps": environment.progress,
                "turn_count": environment.turns,
                "actions_taken": list(environment.actions_taken),
                "text_fallback_actions": outcome.text_fallback_actions,
                "policy_calls": outcome.policy_calls,
                "policy_error": outcome.policy_error or "",
                "action_order_seed": environment.action_order_seed,
            },
            "summary": outcome.final_text[-1_000:],
        }


rollout_agent = SupportRolloutAgent()


@sagemaker_rft_handler
def run_rft_rollout(payload: dict[str, Any]) -> dict[str, Any]:
    """Runs one RFT trajectory through the support agent.

    Args:
        payload: SageMaker RFT rollout request.

    Returns:
        Reward and trajectory diagnostics.
    """
    return rollout_agent.run(payload)


@app.entrypoint
def handle_rollout(payload: dict[str, Any]) -> dict[str, Any]:
    """Routes health checks separately from tracked RFT rollouts.

    Args:
        payload: AgentCore invocation payload.

    Returns:
        Health status or an RFT rollout result.
    """
    if payload.get("smoke_test") is True:
        return rollout_agent.health()
    return run_rft_rollout(payload)


if __name__ == "__main__":
    app.run()

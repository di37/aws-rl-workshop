"""Turn-level driver that keeps rollouts alive when the policy replies in text.

GPT-OSS models served through the Job Runtime often make the first structured
tool call and then name the next action in plain text. Strands treats a
text-only reply as the end of the conversation, which ended every baseline
rollout after two turns. This driver recovers the intended action from the
text, applies it to the environment, and feeds the observation back.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from strands.types.exceptions import MaxTokensReachedException

from environment import MAX_TURNS, SupportEnvironment


logger = logging.getLogger(__name__)

MAX_POLICY_CALLS = MAX_TURNS * 2
"""Upper bound on agent invocations, allowing one reminder per turn."""

LIMIT_STOP_REASON = "limit_turns"
"""Strands stop reason emitted when the per-call cycle cap is reached."""

REMINDER_MESSAGE = (
    "No valid action was received. Call take_support_action, or reply with "
    "only one action name from available_actions.\nCurrent environment:\n{state}"
)
CONTINUE_MESSAGE = (
    "Current environment:\n{state}\n"
    "Continue troubleshooting with the take_support_action tool."
)
OBSERVATION_MESSAGE = "Environment result for action '{action}':\n{result}\n"

_DECORATION = " \t\r\n`'\".,:;!*"
_ACTION_FIELD = re.compile(r'"action"\s*:\s*"([A-Za-z0-9_]+)"')


@dataclass(frozen=True)
class RolloutOutcome:
    """Summarizes how the policy interacted with the environment.

    Attributes:
        final_text: Text of the last policy response.
        text_fallback_actions: Actions recovered from plain-text replies.
        policy_calls: Number of agent invocations made by the driver.
        policy_error: Name of a recoverable model error that ended the
            rollout early, or None.
    """

    final_text: str
    text_fallback_actions: int
    policy_calls: int
    policy_error: str | None = None


class TextActionParser:
    """Extracts a support action from free-form policy text.

    Only unambiguous replies are accepted, in this order: the whole reply is an
    action name, an explicit ``"action": "<name>"`` field, or prose that names
    exactly one available action. Anything else returns None so the driver asks
    again instead of guessing.
    """

    @classmethod
    def parse(cls, text: str, available_actions: Sequence[str]) -> str | None:
        """Finds the single action the policy chose.

        Args:
            text: Policy response text, including raw tool-call markup.
            available_actions: Actions valid for the current state.

        Returns:
            The selected action, or None when the reply is empty or ambiguous.
        """
        if not text or not available_actions:
            return None
        bare = text.strip(_DECORATION)
        if bare in available_actions:
            return bare
        explicit = [
            name for name in _ACTION_FIELD.findall(text) if name in available_actions
        ]
        if explicit:
            return explicit[-1]
        mentioned = cls._mentioned_actions(text, available_actions)
        return mentioned.pop() if len(mentioned) == 1 else None

    @staticmethod
    def _mentioned_actions(text: str, available_actions: Sequence[str]) -> set[str]:
        """Collects available actions named as whole identifiers.

        Args:
            text: Policy response text.
            available_actions: Actions valid for the current state.

        Returns:
            Distinct actions mentioned in the text.
        """
        names = "|".join(re.escape(action) for action in available_actions)
        pattern = rf"(?<![A-Za-z0-9_])({names})(?![A-Za-z0-9_])"
        return set(re.findall(pattern, text))


class RolloutDriver:
    """Drives the policy turn by turn until the environment terminates."""

    def __init__(
        self,
        agent: Callable[..., Any],
        environment: SupportEnvironment,
        max_policy_calls: int = MAX_POLICY_CALLS,
    ) -> None:
        """Initializes the driver for one isolated trajectory.

        Args:
            agent: Callable Strands agent that keeps conversation history and
                accepts a ``limits`` keyword.
            environment: Environment shared with the agent's tool.
            max_policy_calls: Upper bound on agent invocations.
        """
        self._agent = agent
        self._environment = environment
        self._max_policy_calls = max_policy_calls

    def run(self, opening_message: str) -> RolloutOutcome:
        """Runs the trajectory, recovering text-only actions along the way.

        Args:
            opening_message: First user message containing the task.

        Returns:
            Interaction statistics for the completed trajectory.

        Raises:
            Exception: Any agent error other than a max-tokens truncation,
                so infrastructure failures surface as failed rollouts.
        """
        message = opening_message
        final_text = ""
        fallback_actions = 0
        policy_calls = 0

        while policy_calls < self._max_policy_calls:
            policy_calls += 1
            turns_before = self._environment.turns
            try:
                response = self._invoke(message)
            except MaxTokensReachedException:
                logger.warning("Policy output truncated; ending rollout early.")
                return RolloutOutcome(
                    final_text=final_text,
                    text_fallback_actions=fallback_actions,
                    policy_calls=policy_calls,
                    policy_error=MaxTokensReachedException.__name__,
                )
            final_text = self.response_text(response)
            if self._environment.terminal or self._hit_cycle_cap(response):
                break
            if self._environment.turns != turns_before:
                message = self._state_message(CONTINUE_MESSAGE)
                continue
            action = TextActionParser.parse(
                final_text, self._environment.available_actions
            )
            if action is None:
                message = self._state_message(REMINDER_MESSAGE)
                continue
            fallback_actions += 1
            logger.info("Recovered text-only action: %s", action)
            result = self._environment.apply(action)
            if result.terminal:
                break
            message = OBSERVATION_MESSAGE.format(
                action=action, result=json.dumps(result.as_dict())
            ) + self._state_message(CONTINUE_MESSAGE)

        return RolloutOutcome(
            final_text=final_text,
            text_fallback_actions=fallback_actions,
            policy_calls=policy_calls,
        )

    def _invoke(self, message: str) -> Any:
        """Calls the agent with its inner loop capped by the remaining turns.

        Each remaining environment turn needs one tool cycle, plus one cycle
        for the closing text reply.

        Args:
            message: User message for this invocation.

        Returns:
            Strands agent response.
        """
        remaining = MAX_TURNS - self._environment.turns
        return self._agent(message, limits={"turns": remaining + 1})

    @staticmethod
    def _hit_cycle_cap(response: Any) -> bool:
        """Indicates whether Strands stopped because of the cycle cap.

        Args:
            response: Strands agent response.

        Returns:
            True when the inner loop was cut off.
        """
        return getattr(response, "stop_reason", None) == LIMIT_STOP_REASON

    def _state_message(self, template: str) -> str:
        """Formats a message template with the current environment state.

        Args:
            template: Message template containing a ``{state}`` field.

        Returns:
            The formatted user message.
        """
        state = json.dumps(self._environment.current_observation())
        return template.format(state=state)

    @staticmethod
    def response_text(response: Any) -> str:
        """Joins the text blocks of a Strands agent response.

        Args:
            response: Strands agent response.

        Returns:
            Concatenated text, or an empty string when none is present.
        """
        message = getattr(response, "message", None)
        content = message.get("content") if isinstance(message, dict) else None
        return "".join(
            block["text"]
            for block in content or []
            if isinstance(block, dict) and "text" in block
        )

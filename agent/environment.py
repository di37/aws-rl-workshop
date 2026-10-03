"""Deterministic customer-support environment used by the real RFT agent."""

from __future__ import annotations

import random
from dataclasses import dataclass, field


MAX_TURNS = 4
INITIAL_STATE = "new_ticket"
TERMINAL_STATE = "resolved"
SUCCESS_PATH: tuple[str, ...] = (
    INITIAL_STATE,
    "no_outage",
    "red_wan_light",
    "config_mismatch",
    TERMINAL_STATE,
)
MAX_PROGRESS = len(SUCCESS_PATH) - 1

ACTIONS: dict[str, tuple[str, ...]] = {
    "new_ticket": ("check_outage", "restart_router", "close_ticket"),
    "no_outage": ("inspect_router_lights", "restart_router", "close_ticket"),
    "red_wan_light": ("check_account_config", "replace_router", "close_ticket"),
    "config_mismatch": ("apply_config_fix", "escalate", "close_ticket"),
}

TRANSITIONS: dict[tuple[str, str], tuple[str, str]] = {
    ("new_ticket", "check_outage"): ("no_outage", "No area outage is detected."),
    ("no_outage", "inspect_router_lights"): (
        "red_wan_light",
        "The router has a red WAN light.",
    ),
    ("red_wan_light", "check_account_config"): (
        "config_mismatch",
        "The account has a configuration mismatch.",
    ),
    ("config_mismatch", "apply_config_fix"): (
        TERMINAL_STATE,
        "The connection is restored.",
    ),
}

OBSERVATIONS: dict[str, str] = {
    "new_ticket": "The customer reports that their internet is not working.",
    "no_outage": "No area outage is detected.",
    "red_wan_light": "The router has a red WAN light.",
    "config_mismatch": "The account has a configuration mismatch.",
    TERMINAL_STATE: "The connection is restored.",
}


@dataclass(frozen=True)
class ActionResult:
    """Represents one environment transition."""

    accepted: bool
    state: str
    observation: str
    available_actions: tuple[str, ...]
    terminal: bool
    turn: int

    def as_dict(self) -> dict[str, object]:
        """Serializes the transition for a model tool response.

        Returns:
            JSON-compatible transition data.
        """
        return {
            "accepted": self.accepted,
            "state": self.state,
            "observation": self.observation,
            "available_actions": list(self.available_actions),
            "terminal": self.terminal,
            "turn": self.turn,
        }


@dataclass
class SupportEnvironment:
    """Models the deterministic four-turn customer-support task.

    Attributes:
        action_order_seed: Seed that permutes how actions are presented in each
            state, so the correct action is not always listed first. None keeps
            the canonical order.
        state: Current diagnostic state.
        turns: Number of actions applied so far.
        actions_taken: Actions in the order they were applied.
    """

    action_order_seed: int | None = None
    state: str = INITIAL_STATE
    turns: int = 0
    actions_taken: list[str] = field(default_factory=list)

    @property
    def completed(self) -> bool:
        """Indicates whether the connection was restored."""
        return self.state == TERMINAL_STATE

    @property
    def terminal(self) -> bool:
        """Indicates whether the episode has ended."""
        return self.completed or self.turns >= MAX_TURNS

    @property
    def progress(self) -> int:
        """Returns how many correct diagnostic steps have been completed."""
        return SUCCESS_PATH.index(self.state)

    @property
    def reward(self) -> float:
        """Returns partial credit for each correct step along the success path.

        Wrong actions never advance the state, so they earn nothing, and only a
        fully restored connection reaches 1.0.
        """
        return self.progress / MAX_PROGRESS

    @property
    def available_actions(self) -> tuple[str, ...]:
        """Returns actions valid for the current state in presentation order."""
        actions = ACTIONS.get(self.state, ())
        if self.action_order_seed is None:
            return actions
        rng = random.Random(f"{self.action_order_seed}:{self.state}")
        return tuple(rng.sample(actions, k=len(actions)))

    def current_observation(self) -> dict[str, object]:
        """Builds the observation describing the current state.

        Returns:
            JSON-compatible environment observation.
        """
        return {
            "state": self.state,
            "observation": OBSERVATIONS[self.state],
            "available_actions": list(self.available_actions),
            "turns_remaining": MAX_TURNS - self.turns,
        }

    def apply(self, action: str) -> ActionResult:
        """Applies one policy action to the environment.

        Args:
            action: Action selected by the policy model.

        Returns:
            The resulting observation and terminal status.
        """
        if self.completed:
            return self._result(False, "The task is already complete.")
        if self.turns >= MAX_TURNS:
            return self._result(False, "The turn limit has been reached.")

        self.turns += 1
        self.actions_taken.append(action)
        transition = TRANSITIONS.get((self.state, action))
        if transition is None:
            return self._result(
                False,
                f"Action '{action}' did not resolve the current state.",
            )

        self.state, observation = transition
        return self._result(True, observation)

    def _result(self, accepted: bool, observation: str) -> ActionResult:
        """Constructs an action result without duplicating state metadata.

        Args:
            accepted: Whether the action followed a valid transition.
            observation: Human-readable environment response.

        Returns:
            An immutable action result.
        """
        return ActionResult(
            accepted=accepted,
            state=self.state,
            observation=observation,
            available_actions=self.available_actions,
            terminal=self.terminal,
            turn=self.turns,
        )

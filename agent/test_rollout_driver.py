import unittest
from types import SimpleNamespace
from typing import Any, Callable

from strands.types.exceptions import MaxTokensReachedException

from environment import MAX_TURNS, SupportEnvironment
from rollout_driver import MAX_POLICY_CALLS, RolloutDriver, TextActionParser

SOLUTION = (
    "check_outage",
    "inspect_router_lights",
    "check_account_config",
    "apply_config_fix",
)


def text_response(text: str, stop_reason: str = "end_turn") -> SimpleNamespace:
    """Builds a minimal stand-in for a Strands agent response."""
    return SimpleNamespace(
        message={"content": [{"text": text}]}, stop_reason=stop_reason
    )


class ScriptedAgent:
    """Replays scripted policy turns and records how it was invoked."""

    def __init__(self, turns: list[Callable[[], Any]]) -> None:
        self._turns = list(turns)
        self.messages: list[str] = []
        self.limits: list[dict[str, int] | None] = []

    def __call__(
        self, message: str, limits: dict[str, int] | None = None
    ) -> SimpleNamespace:
        self.messages.append(message)
        self.limits.append(limits)
        turn = self._turns.pop(0) if self._turns else (lambda: "")
        reply = turn()
        return reply if isinstance(reply, SimpleNamespace) else text_response(reply)


class TextActionParserTests(unittest.TestCase):
    AVAILABLE = ("inspect_router_lights", "restart_router", "close_ticket")

    def test_plain_action_name_is_parsed(self) -> None:
        self.assertEqual(
            TextActionParser.parse("inspect_router_lights", self.AVAILABLE),
            "inspect_router_lights",
        )

    def test_decorated_action_name_is_parsed(self) -> None:
        self.assertEqual(
            TextActionParser.parse(" `restart_router`. ", self.AVAILABLE),
            "restart_router",
        )

    def test_action_inside_raw_tool_call_text_is_parsed(self) -> None:
        text = 'to=functions.take_support_action {"action": "inspect_router_lights"}'

        self.assertEqual(
            TextActionParser.parse(text, self.AVAILABLE), "inspect_router_lights"
        )

    def test_explicit_action_field_beats_other_mentions(self) -> None:
        text = 'Not close_ticket. {"action": "inspect_router_lights"}'

        self.assertEqual(
            TextActionParser.parse(text, self.AVAILABLE), "inspect_router_lights"
        )

    def test_ambiguous_mentions_return_none(self) -> None:
        text = "I will inspect_router_lights rather than close_ticket."

        self.assertIsNone(TextActionParser.parse(text, self.AVAILABLE))

    def test_echoed_action_list_returns_none(self) -> None:
        text = "available_actions: inspect_router_lights, restart_router, close_ticket"

        self.assertIsNone(TextActionParser.parse(text, self.AVAILABLE))

    def test_single_mention_in_prose_is_parsed(self) -> None:
        text = "Next I will inspect_router_lights to see the WAN status."

        self.assertEqual(
            TextActionParser.parse(text, self.AVAILABLE), "inspect_router_lights"
        )

    def test_unavailable_actions_are_ignored(self) -> None:
        self.assertIsNone(
            TextActionParser.parse("I already ran check_outage.", self.AVAILABLE)
        )

    def test_action_embedded_in_longer_identifier_is_ignored(self) -> None:
        self.assertIsNone(TextActionParser.parse("close_ticket_later", self.AVAILABLE))

    def test_empty_text_returns_none(self) -> None:
        self.assertIsNone(TextActionParser.parse("", self.AVAILABLE))


class RolloutDriverTests(unittest.TestCase):
    def test_text_actions_continue_the_episode_to_completion(self) -> None:
        environment = SupportEnvironment()
        agent = ScriptedAgent([(lambda action=a: action) for a in SOLUTION])

        outcome = RolloutDriver(agent, environment).run("Help the customer.")

        self.assertTrue(environment.completed)
        self.assertEqual(environment.reward, 1.0)
        self.assertEqual(outcome.text_fallback_actions, 4)
        self.assertEqual(outcome.policy_calls, 4)

    def test_observation_is_fed_back_after_text_action(self) -> None:
        environment = SupportEnvironment()
        agent = ScriptedAgent([lambda: "check_outage", lambda: "close_ticket"])

        RolloutDriver(agent, environment, max_policy_calls=2).run("Help.")

        self.assertIn("No area outage is detected.", agent.messages[1])
        self.assertIn("inspect_router_lights", agent.messages[1])

    def test_recap_of_tool_action_is_not_reapplied(self) -> None:
        environment = SupportEnvironment()

        def failed_tool_call_then_recap() -> str:
            environment.apply("restart_router")
            return "restart_router"

        agent = ScriptedAgent([failed_tool_call_then_recap, lambda: "check_outage"])

        outcome = RolloutDriver(agent, environment, max_policy_calls=2).run("Help.")

        self.assertEqual(environment.actions_taken, ["restart_router", "check_outage"])
        self.assertEqual(outcome.text_fallback_actions, 1)
        self.assertNotIn("No valid action", agent.messages[1])
        self.assertIn("check_outage", agent.messages[1])

    def test_tool_only_rollout_is_not_double_applied(self) -> None:
        environment = SupportEnvironment()

        def solve_with_tools() -> str:
            for action in SOLUTION:
                environment.apply(action)
            return "Done. I used apply_config_fix."

        agent = ScriptedAgent([solve_with_tools])

        outcome = RolloutDriver(agent, environment).run("Help.")

        self.assertEqual(environment.turns, 4)
        self.assertEqual(outcome.text_fallback_actions, 0)
        self.assertEqual(outcome.policy_calls, 1)

    def test_unparseable_reply_triggers_reminder_without_using_a_turn(self) -> None:
        environment = SupportEnvironment()
        agent = ScriptedAgent([lambda: "Let me think.", lambda: "check_outage"])

        RolloutDriver(agent, environment, max_policy_calls=2).run("Help.")

        self.assertIn("No valid action", agent.messages[1])
        self.assertEqual(environment.turns, 1)
        self.assertEqual(environment.progress, 1)

    def test_policy_calls_are_bounded(self) -> None:
        environment = SupportEnvironment()
        agent = ScriptedAgent([])

        outcome = RolloutDriver(agent, environment).run("Help.")

        self.assertEqual(outcome.policy_calls, MAX_POLICY_CALLS)
        self.assertEqual(environment.turns, 0)

    def test_inner_agent_loop_is_capped_by_remaining_turns(self) -> None:
        environment = SupportEnvironment()
        agent = ScriptedAgent([lambda: "check_outage", lambda: "close_ticket"])

        RolloutDriver(agent, environment, max_policy_calls=2).run("Help.")

        self.assertEqual(agent.limits, [{"turns": MAX_TURNS + 1}, {"turns": MAX_TURNS}])

    def test_turn_limit_stop_ends_the_rollout(self) -> None:
        environment = SupportEnvironment()
        agent = ScriptedAgent(
            [lambda: text_response("check_outage", stop_reason="limit_turns")]
        )

        outcome = RolloutDriver(agent, environment).run("Help.")

        self.assertEqual(outcome.policy_calls, 1)
        self.assertEqual(environment.turns, 0)

    def test_max_tokens_error_keeps_partial_credit(self) -> None:
        environment = SupportEnvironment()

        def truncated() -> str:
            raise MaxTokensReachedException("truncated")

        agent = ScriptedAgent([lambda: "check_outage", truncated])

        outcome = RolloutDriver(agent, environment).run("Help.")

        self.assertEqual(environment.reward, 0.25)
        self.assertEqual(outcome.policy_error, "MaxTokensReachedException")
        self.assertEqual(outcome.policy_calls, 2)

    def test_other_errors_propagate(self) -> None:
        def broken() -> str:
            raise RuntimeError("auth failure")

        with self.assertRaises(RuntimeError):
            RolloutDriver(ScriptedAgent([broken]), SupportEnvironment()).run("Help.")

    def test_final_text_is_returned(self) -> None:
        environment = SupportEnvironment()
        agent = ScriptedAgent([lambda: "close_ticket"] * 4)

        outcome = RolloutDriver(agent, environment).run("Help.")

        self.assertEqual(outcome.final_text, "close_ticket")
        self.assertIsNone(outcome.policy_error)
        self.assertTrue(environment.terminal)


class ResponseTextTests(unittest.TestCase):
    def test_text_blocks_are_joined_and_other_blocks_skipped(self) -> None:
        response = SimpleNamespace(
            message={"content": [{"text": "a"}, {"toolUse": {}}, {"text": "b"}]}
        )

        self.assertEqual(RolloutDriver.response_text(response), "ab")

    def test_missing_message_yields_empty_text(self) -> None:
        self.assertEqual(RolloutDriver.response_text(object()), "")


if __name__ == "__main__":
    unittest.main()

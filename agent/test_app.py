import unittest
from types import SimpleNamespace
from unittest.mock import patch

from pydantic import ValidationError

from app import SupportRolloutAgent
from environment import SupportEnvironment
from rollout_driver import RolloutOutcome


class SupportRolloutAgentTests(unittest.TestCase):
    def test_health_check_has_no_external_dependencies(self) -> None:
        self.assertEqual(
            SupportRolloutAgent.health(),
            {"status": "healthy", "service": "mtrl-support-agent"},
        )

    def test_prompt_is_extracted_from_rft_instance(self) -> None:
        prompt = SupportRolloutAgent._extract_prompt(
            {"instance": {"prompt": "Resolve this ticket."}}
        )

        self.assertEqual(prompt, "Resolve this ticket.")

    def test_malformed_prompt_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            SupportRolloutAgent._extract_prompt(
                {"instance": {"prompt": ["unexpected", "content"]}}
            )

    @patch("app.generate_token", return_value="generated-token")
    def test_regional_runtime_url_is_derived(self, generate_token_mock) -> None:
        config = SupportRolloutAgent._resolve_policy_config(
            {
                "modelName": "policy-model",
                "inferenceParams": {"maxTokens": 512, "topP": 0.9},
            },
            {"region": "us-west-2"},
        )

        self.assertEqual(
            config.base_url,
            "https://job-runtime.sagemaker.us-west-2.api.aws/v1",
        )
        self.assertEqual(config.api_key, "generated-token")
        self.assertEqual(config.sampling_params, {"max_tokens": 512, "top_p": 0.9})
        generate_token_mock.assert_called_once_with(region="us-west-2")

    def test_untrusted_runtime_endpoint_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "untrusted"):
            SupportRolloutAgent._resolve_policy_config(
                {"modelEndpoint": "https://attacker.example/v1"},
                {"region": "us-west-2"},
            )

    def test_existing_v1_path_is_not_duplicated(self) -> None:
        endpoint = SupportRolloutAgent._validate_runtime_endpoint(
            "https://job-runtime.sagemaker.us-west-2.api.aws/v1",
            "us-west-2",
        )

        self.assertEqual(
            endpoint,
            "https://job-runtime.sagemaker.us-west-2.api.aws/v1",
        )

    def test_result_reports_partial_reward_and_fallback_metrics(self) -> None:
        environment = SupportEnvironment(action_order_seed=11)
        environment.apply("check_outage")
        environment.apply("inspect_router_lights")
        outcome = RolloutOutcome(
            final_text="done", text_fallback_actions=1, policy_calls=2
        )

        result = SupportRolloutAgent.build_result(environment, outcome)

        self.assertEqual(result["reward"], 0.5)
        self.assertEqual(result["metrics"]["progress_steps"], 2)
        self.assertEqual(result["metrics"]["text_fallback_actions"], 1)
        self.assertEqual(result["metrics"]["policy_calls"], 2)
        self.assertEqual(result["metrics"]["policy_error"], "")
        self.assertEqual(result["metrics"]["action_order_seed"], 11)
        self.assertFalse(result["metrics"]["task_completed"])
        self.assertEqual(result["summary"], "done")

    def test_malformed_sampling_params_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "parameters"):
            SupportRolloutAgent._resolve_policy_config(
                {"inferenceParams": ["maxTokens", 512]},
                {"region": "us-west-2"},
            )


class FakeAgent:
    """Stands in for a Strands Agent: replays tool calls through the real tool."""

    created: list["FakeAgent"] = []

    def __init__(self, model, tools, system_prompt, **options) -> None:
        self.tool = tools[0]
        self.options = options
        self.messages: list[str] = []
        FakeAgent.created.append(self)

    def __call__(self, message, limits=None):
        self.messages.append(message)
        for action in (
            "check_outage",
            "inspect_router_lights",
            "check_account_config",
            "apply_config_fix",
        ):
            self.tool(action=action)
        return SimpleNamespace(message={"content": [{"text": "Resolved."}]})


class RunEpisodeTests(unittest.TestCase):
    """Verifies the reusable episode wiring shared by AgentCore and endpoints."""

    def setUp(self) -> None:
        FakeAgent.created = []

    @patch("app.Agent", FakeAgent)
    def test_episode_scores_a_full_resolution(self) -> None:
        result = SupportRolloutAgent.run_episode(
            object(), "Customer says: no internet.", action_order_seed=3
        )

        self.assertEqual(result["reward"], 1.0)
        self.assertTrue(result["metrics"]["task_completed"])
        self.assertEqual(result["metrics"]["action_order_seed"], 3)
        self.assertEqual(result["summary"], "Resolved.")

    @patch("app.Agent", FakeAgent)
    def test_agent_options_are_forwarded(self) -> None:
        SupportRolloutAgent.run_episode(
            object(), "ticket", action_order_seed=1, callback_handler=None
        )

        self.assertEqual(FakeAgent.created[0].options, {"callback_handler": None})

    @patch("app.Agent", FakeAgent)
    def test_opening_message_carries_ticket_and_observation(self) -> None:
        SupportRolloutAgent.run_episode(
            object(), "Customer says: help", action_order_seed=1
        )

        opening = FakeAgent.created[0].messages[0]
        self.assertIn("Customer says: help", opening)
        self.assertIn('"state": "new_ticket"', opening)


if __name__ == "__main__":
    unittest.main()

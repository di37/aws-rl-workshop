import unittest
import random

from environment import (
    ACTIONS,
    INITIAL_STATE,
    MAX_PROGRESS,
    MAX_TURNS,
    SUCCESS_PATH,
    TRANSITIONS,
    SupportEnvironment,
)


class SupportEnvironmentTests(unittest.TestCase):
    def test_successful_four_turn_trajectory(self) -> None:
        environment = SupportEnvironment()
        actions = (
            "check_outage",
            "inspect_router_lights",
            "check_account_config",
            "apply_config_fix",
        )

        results = [environment.apply(action) for action in actions]

        self.assertTrue(environment.completed)
        self.assertEqual(environment.reward, 1.0)
        self.assertEqual(environment.turns, 4)
        self.assertTrue(results[-1].terminal)

    def test_wrong_action_does_not_receive_reward(self) -> None:
        environment = SupportEnvironment()

        result = environment.apply("restart_router")

        self.assertFalse(result.accepted)
        self.assertFalse(environment.completed)
        self.assertEqual(environment.reward, 0.0)

    def test_turn_limit_is_enforced(self) -> None:
        environment = SupportEnvironment()

        for _ in range(MAX_TURNS):
            environment.apply("close_ticket")
        result = environment.apply("check_outage")

        self.assertTrue(result.terminal)
        self.assertEqual(environment.turns, MAX_TURNS)
        self.assertEqual(environment.reward, 0.0)

    def test_every_state_has_three_actions(self) -> None:
        self.assertTrue(all(len(actions) == 3 for actions in ACTIONS.values()))

    def test_each_correct_step_earns_partial_credit(self) -> None:
        environment = SupportEnvironment()

        rewards = []
        for action in ("check_outage", "inspect_router_lights", "check_account_config"):
            environment.apply(action)
            rewards.append(environment.reward)

        self.assertEqual(rewards, [0.25, 0.5, 0.75])
        self.assertFalse(environment.completed)

    def test_wrong_action_after_progress_keeps_earned_credit(self) -> None:
        environment = SupportEnvironment()

        environment.apply("check_outage")
        environment.apply("restart_router")

        self.assertEqual(environment.progress, 1)
        self.assertEqual(environment.reward, 0.25)

    def test_success_path_matches_transitions(self) -> None:
        for current, following in zip(SUCCESS_PATH, SUCCESS_PATH[1:]):
            self.assertIn(
                following,
                {
                    target
                    for (source, _), (target, _) in TRANSITIONS.items()
                    if source == current
                },
            )

    def test_every_transition_stays_on_the_success_path(self) -> None:
        for (source, _), (target, _) in TRANSITIONS.items():
            self.assertIn(source, SUCCESS_PATH)
            self.assertIn(target, SUCCESS_PATH)

    def test_unseeded_action_order_is_canonical(self) -> None:
        self.assertEqual(SupportEnvironment().available_actions, ACTIONS[INITIAL_STATE])

    def test_seeded_action_order_is_a_deterministic_permutation(self) -> None:
        first = SupportEnvironment(action_order_seed=7)
        second = SupportEnvironment(action_order_seed=7)

        self.assertEqual(first.available_actions, second.available_actions)
        self.assertCountEqual(first.available_actions, ACTIONS[INITIAL_STATE])

    def test_seeds_vary_which_position_holds_the_correct_action(self) -> None:
        positions = {
            SupportEnvironment(action_order_seed=seed).available_actions.index(
                "check_outage"
            )
            for seed in range(30)
        }

        self.assertEqual(positions, {0, 1, 2})

    def test_current_observation_reflects_state_and_order(self) -> None:
        environment = SupportEnvironment(action_order_seed=3)
        environment.apply("check_outage")

        observation = environment.current_observation()

        self.assertEqual(observation["state"], "no_outage")
        self.assertEqual(
            observation["available_actions"], list(environment.available_actions)
        )
        self.assertEqual(observation["turns_remaining"], MAX_TURNS - 1)

    def test_terminal_reflects_completion_or_turn_limit(self) -> None:
        environment = SupportEnvironment()
        self.assertFalse(environment.terminal)

        for _ in range(MAX_TURNS):
            environment.apply("close_ticket")

        self.assertTrue(environment.terminal)

    def test_random_rollouts_only_award_full_credit_on_completion(self) -> None:
        rng = random.Random(2026)

        for _ in range(500):
            environment = SupportEnvironment()
            for _ in range(MAX_TURNS):
                action = rng.choice(environment.available_actions)
                result = environment.apply(action)
                if result.terminal:
                    break
            self.assertEqual(environment.reward == 1.0, environment.completed)
            self.assertEqual(environment.reward, environment.progress / MAX_PROGRESS)


if __name__ == "__main__":
    unittest.main()

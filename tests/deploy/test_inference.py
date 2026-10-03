"""Unit tests for before/after inference with the same support agent."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

from aws.deploy.inference import InferenceComparison, side_by_side, trace
from aws.records.evidence import EvidenceStore

SOLVED = ["check_outage", "inspect_router_lights", "check_account_config", "apply_config_fix"]
SHORTCUT = ["restart_router", "check_outage", "inspect_router_lights", "check_account_config"]
LABELS = {"before": "Base model", "after": "Fine-tuned model"}


def fake_episode(model: str, ticket: str, seed: int) -> dict:
    """Plays a scripted episode: the base takes the shortcut, the tuned model solves it."""
    if model == "broken":
        raise RuntimeError("tool call could not be parsed")
    actions = SHORTCUT if model == "base-model" else SOLVED
    reward = 0.75 if model == "base-model" else 1.0
    return {"reward": reward, "metrics": {"actions_taken": actions}, "summary": "done"}


class TraceTests(unittest.TestCase):
    """Verifies the turn-by-turn replay of an episode."""

    def test_trace_marks_wasted_turns(self) -> None:
        steps = trace(SHORTCUT, seed=1)

        self.assertFalse(steps[0]["accepted"])
        self.assertEqual(steps[0]["reward"], 0.0)
        self.assertTrue(steps[1]["accepted"])
        self.assertEqual(steps[-1]["reward"], 0.75)

    def test_trace_of_solution_ends_restored(self) -> None:
        steps = trace(SOLVED, seed=1)

        self.assertEqual(steps[-1]["observation"], "The connection is restored.")
        self.assertEqual(steps[-1]["reward"], 1.0)


class InferenceComparisonTests(unittest.TestCase):
    """Verifies both models see the same ticket and results are saved."""

    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.store = EvidenceStore(Path(self.directory.name))
        self.seen: list[tuple[str, str, int]] = []

        def episode(model: str, ticket: str, seed: int) -> dict:
            self.seen.append((model, ticket, seed))
            return fake_episode(model, ticket, seed)

        self.comparison = InferenceComparison(
            self.store, episode_runner=episode, chat=lambda component, message: f"{component}: hi"
        )
        self.models = {"before": (LABELS["before"], "base-model"),
                       "after": (LABELS["after"], "tuned-model")}

    def tearDown(self) -> None:
        self.directory.cleanup()

    def test_both_models_see_the_same_ticket_and_seed(self) -> None:
        self.comparison.compare(self.models, ["T1"], seed=7)

        self.assertEqual(self.seen, [("base-model", "T1", 7), ("tuned-model", "T1", 7)])

    def test_results_include_labels_traces_and_are_saved(self) -> None:
        results = self.comparison.compare(self.models, ["T1"], seed=7)

        entry = results["tickets"][0]
        self.assertEqual(results["labels"], LABELS)
        self.assertEqual(entry["before"]["reward"], 0.75)
        self.assertEqual(entry["after"]["trace"][-1]["reward"], 1.0)
        self.assertEqual(self.store.load(InferenceComparison.RECORD)["seed"], 7)

    def test_model_errors_are_recorded_not_raised(self) -> None:
        models = {"before": ("Base model", "broken"), "after": ("Fine-tuned model", "tuned-model")}

        entry = self.comparison.compare(models, ["T1"], 1)["tickets"][0]

        self.assertIn("tool call could not be parsed", entry["before"]["error"])
        self.assertEqual(entry["after"]["reward"], 1.0)

    def test_chat_invokes_the_endpoint_component_and_saves_the_reply(self) -> None:
        reply = self.comparison.chat("tuned-ic", "hello")

        self.assertEqual(reply, "tuned-ic: hi")
        saved = self.store.load(InferenceComparison.CHAT_RECORD)
        self.assertEqual(saved, {"component": "tuned-ic", "message": "hello", "reply": "tuned-ic: hi"})


class SideBySideTests(unittest.TestCase):
    """Verifies the presentation table for one ticket."""

    def test_rows_pair_turns_and_end_with_reward(self) -> None:
        entry = {
            "ticket": "T1",
            "before": {"reward": 0.75, "trace": trace(SHORTCUT, 1)},
            "after": {"reward": 1.0, "trace": trace(SOLVED, 1)},
        }

        rows = side_by_side(entry, LABELS)

        self.assertEqual(rows[0]["Base model"], "restart_router  [wasted turn]")
        self.assertEqual(rows[0]["Fine-tuned model"], "check_outage  [correct]")
        self.assertEqual(rows[-1]["Turn"], "Reward")
        self.assertEqual(rows[-1]["Fine-tuned model"], "1.0")

    def test_errors_are_shown_in_place_of_actions(self) -> None:
        entry = {"ticket": "T1", "before": {"error": "boom"},
                 "after": {"reward": 1.0, "trace": trace(SOLVED, 1)}}

        rows = side_by_side(entry, LABELS)

        self.assertEqual(rows[0]["Base model"], "error: boom")


class LiveRunRecordTests(unittest.TestCase):
    """Verifies the first live run stays the record and the budget gate runs first."""

    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.store = EvidenceStore(Path(self.directory.name))
        self.gate = MagicMock()
        self.calls: list[str] = []

        def episode(model: str, ticket: str, seed: int) -> dict:
            self.calls.append(ticket)
            return fake_episode(model, ticket, seed)

        self.comparison = InferenceComparison(
            self.store, episode_runner=episode, chat=lambda component, message: "reply",
            budget_gate=self.gate, clock=lambda: datetime(2026, 10, 3, 9, 30, tzinfo=timezone.utc),
        )
        self.models = {"before": ("Base", "base-model"), "after": ("Tuned", "tuned-model")}

    def tearDown(self) -> None:
        self.directory.cleanup()

    def test_later_runs_are_saved_beside_the_first_record(self) -> None:
        self.comparison.compare(self.models, ["ticket a"], seed=7)
        second = self.comparison.compare(self.models, ["ticket b"], seed=7)

        self.assertEqual(self.store.load(InferenceComparison.RECORD)["tickets"][0]["ticket"], "ticket a")
        self.assertEqual(second["saved_as"], "inference_before_after.20261003T093000Z.json")
        self.assertEqual(self.store.load(second["saved_as"])["tickets"][0]["ticket"], "ticket b")

    def test_budget_gate_refusal_calls_no_model(self) -> None:
        self.gate.side_effect = PermissionError("over the $25 cap")

        with self.assertRaises(PermissionError):
            self.comparison.compare(self.models, ["ticket a"], seed=7)

        self.assertEqual(self.calls, [])

    def test_chat_replies_follow_the_same_write_once_rule(self) -> None:
        self.comparison.record_chat("component", "question", "first reply")
        name = self.comparison.record_chat("component", "question", "second reply")

        self.assertEqual(self.store.load(InferenceComparison.CHAT_RECORD)["reply"], "first reply")
        self.assertEqual(self.store.load(name)["reply"], "second reply")

if __name__ == "__main__":
    unittest.main()

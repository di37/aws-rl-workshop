"""Unit tests for reading SageMaker-recorded evaluation conversations."""

from __future__ import annotations

import json
import unittest
from unittest.mock import MagicMock

from aws.rl.trajectories import (
    TrajectoryReader,
    compare_tickets,
    comparison_record,
    pair_by_ticket,
    trace_uris,
)

TICKET = "Resolve the customer's internet outage. Customer says: No internet."


def raw_trace(actions: list[tuple[str, bool]], reward: str, thought: str) -> dict:
    """Builds a raw ``traces.json`` payload shaped like SageMaker's MTRL traces."""
    prompt = f"{TICKET}\n\nInitial environment:\n{{}}\nBegin troubleshooting using the tool."
    spans = [{"name": "trajectory", "start_time": 1, "attributes": {
        "mlflow.spanInputs": json.dumps({"prompt": prompt}), "reward": reward, "status": "reward_received"}}]
    for index, (action, accepted) in enumerate(actions, start=1):
        spans.append({"name": f"turn-{index}", "start_time": 10 * index, "attributes": {
            "mlflow.spanOutputs": json.dumps({"thinking": thought if index == 1 else "next"})}})
        spans.append({"name": "take_support_action", "start_time": 10 * index + 1, "attributes": {
            "mlflow.spanInputs": json.dumps({"arguments": json.dumps({"action": action})}),
            "mlflow.spanOutputs": json.dumps({"result": json.dumps(
                {"accepted": accepted, "observation": f"after {action}"})})}})
    return {"spans": spans}


SHORTCUT = raw_trace([("restart_router", False), ("check_outage", True)], "0.75", "Restart the router.")
SOLVED = raw_trace([("check_outage", True), ("inspect_router_lights", True)], "1.00", "Check outage first.")


class ParseTests(unittest.TestCase):
    """Verifies one raw trace becomes a readable conversation."""

    def test_parse_extracts_ticket_reward_thought_and_steps(self) -> None:
        conversation = TrajectoryReader.parse(SHORTCUT)

        self.assertEqual(conversation["ticket"], TICKET)
        self.assertEqual(conversation["reward"], 0.75)
        self.assertEqual(conversation["first_thought"], "Restart the router.")
        self.assertEqual(
            [(s["action"], s["accepted"]) for s in conversation["steps"]],
            [("restart_router", False), ("check_outage", True)],
        )

    def test_parse_tolerates_turns_without_tool_calls(self) -> None:
        trace = raw_trace([], "0.00", "unused")
        trace["spans"].append({"name": "turn-1", "start_time": 5, "attributes": {
            "mlflow.spanOutputs": json.dumps({"thinking": "I will just talk."})}})

        conversation = TrajectoryReader.parse(trace)

        self.assertEqual(conversation["steps"], [])
        self.assertEqual(conversation["first_thought"], "I will just talk.")


class ReaderTests(unittest.TestCase):
    """Verifies traces are listed, read, and saved per run."""

    def test_read_run_parses_every_listed_trace(self) -> None:
        stored = {"s3://b/t1": SHORTCUT, "s3://b/t2": SOLVED}
        reader = TrajectoryReader(
            list_traces=lambda experiment, run: ["s3://b/t1", "s3://b/t2"] if experiment == "7" else [],
            read_json=lambda uri: stored[uri],
            experiment_of=lambda run: "7",
        )

        conversations = reader.read_run("run-a")

        self.assertEqual([c["reward"] for c in conversations], [0.75, 1.0])


class PairingTests(unittest.TestCase):
    """Verifies before/after pairing and the honest outcome summary."""

    def test_pairs_keep_held_out_order_and_both_models(self) -> None:
        before = [TrajectoryReader.parse(SHORTCUT)]
        after = [TrajectoryReader.parse(SOLVED)]

        pairs = pair_by_ticket(before, after, order=[TICKET])

        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["before"][0]["reward"], 0.75)
        self.assertEqual(pairs[0]["after"][0]["reward"], 1.0)

    def test_compare_tickets_counts_improved_same_and_worse(self) -> None:
        pairs = [
            {"ticket": "a", "before": [{"reward": 0.75}], "after": [{"reward": 1.0}]},
            {"ticket": "b", "before": [{"reward": 1.0}], "after": [{"reward": 1.0}]},
            {"ticket": "c", "before": [{"reward": 1.0}], "after": [{"reward": 0.75}]},
        ]

        self.assertEqual(compare_tickets(pairs), {"improved": 1, "same": 1, "worse": 1})



class ComparisonRecordTests(unittest.TestCase):
    """Verifies both evaluation runs are read and paired into one record."""

    def test_record_reads_each_models_run_and_pairs_by_ticket(self) -> None:
        reader = MagicMock()
        reader.read_run.side_effect = lambda run_id: {
            "base-run": [TrajectoryReader.parse(SHORTCUT)],
            "tuned-run": [TrajectoryReader.parse(SOLVED)],
        }[run_id]
        results = {"base": {"run_id": "base-run"}, "fine_tuned": {"run_id": "tuned-run"}}

        record = comparison_record(reader, results, [TICKET])

        self.assertEqual(len(record["pairs"]), 1)
        self.assertEqual(record["pairs"][0]["after"][0]["reward"], 1.0)
        self.assertEqual(record["summary"], {"improved": 1, "same": 0, "worse": 0})


class RobustnessTests(unittest.TestCase):
    """Verifies malformed or partial traces are handled without data loss."""

    def test_trace_uris_skip_traces_without_s3_location(self) -> None:
        infos = [SimpleTrace({"mlflow.artifactLocation": "s3://b/t1/artifacts"}),
                 SimpleTrace({}), SimpleTrace({"mlflow.artifactLocation": "file:///tmp/x"})]

        uris, skipped = trace_uris(infos)

        self.assertEqual(uris, ["s3://b/t1/artifacts/traces.json"])
        self.assertEqual(skipped, 2)

    def test_missing_reward_is_unknown_not_zero(self) -> None:
        trace = raw_trace([("check_outage", True)], "", "think")
        del trace["spans"][0]["attributes"]["reward"]

        self.assertIsNone(TrajectoryReader.parse(trace)["reward"])

    def test_first_thought_comes_only_from_turn_one(self) -> None:
        trace = raw_trace([("check_outage", True)], "1.00", "turn one thought")
        trace["spans"] = [s for s in trace["spans"] if s["name"] != "turn-1"]

        self.assertEqual(TrajectoryReader.parse(trace)["first_thought"], "")

    def test_pairing_ignores_surrounding_whitespace(self) -> None:
        before = [{**TrajectoryReader.parse(SHORTCUT), "ticket": TICKET + "  "}]
        after = [TrajectoryReader.parse(SOLVED)]

        self.assertEqual(len(pair_by_ticket(before, after, order=[TICKET])), 1)

    def test_compare_skips_unknown_rewards_and_empty_groups(self) -> None:
        pairs = [{"ticket": "a", "before": [], "after": [{"reward": 1.0}]},
                 {"ticket": "b", "before": [{"reward": None}], "after": [{"reward": 1.0}]},
                 {"ticket": "c", "before": [{"reward": 0.75}], "after": [{"reward": 1.0}]}]

        self.assertEqual(compare_tickets(pairs), {"improved": 1, "same": 0, "worse": 0})


class SimpleTrace:
    """Minimal stand-in for an MLflow trace info with tags."""

    def __init__(self, tags: dict) -> None:
        self.tags = tags


if __name__ == "__main__":
    unittest.main()

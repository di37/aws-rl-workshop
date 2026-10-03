"""Tests for the reproducibility sheet builder."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from aws.records.evidence import EvidenceStore
from aws.reporting.invariants import run_all
from aws.reporting.repro_sheet import build_sheet
from aws.reporting.repro_sheet_materials import verify_trace_metrics

ROOT = Path(__file__).resolve().parents[2]
TASK = {"success_path": ["a", "b"], "max_turns": 4, "max_progress": 4, "max_policy_calls": 8}


class TraceMetricTests(unittest.TestCase):
    """Verifies the sheet refuses metric definitions the traces contradict."""

    def test_mismatching_reported_metric_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = EvidenceStore(Path(directory))
            store.save("evaluation_trajectories.json", {"pairs": [
                {"before": [{"reward": 1.0}, {"reward": 0.75}], "after": [{"reward": 1.0}, {"reward": 1.0}]}]})
            metrics = {"eval/reward/succeeded_rollouts": 1, "eval/reward/pass_at_1": 0.5,
                       "eval/reward/pass_at_2": 1.0, "eval/reward/mean": 0.875, "eval/reward/std": 0.125}
            tuned = {**metrics, "eval/reward/succeeded_rollouts": 2, "eval/reward/pass_at_1": 1.0,
                     "eval/reward/mean": 1.0, "eval/reward/std": 0.0}
            store.save("comparison_evaluation_metrics.json", {"base": {"metrics": metrics},
                                                              "fine_tuned": {"metrics": tuned}})
            self.assertEqual(verify_trace_metrics(store)["eval/reward/pass_at_2"], "recomputed")

            store.save("comparison_evaluation_metrics.json", {"base": {"metrics": {**metrics, "eval/reward/pass_at_2": 0.5}},
                                                              "fine_tuned": {"metrics": tuned}})
            with self.assertRaisesRegex(RuntimeError, "pass_at_2"):
                verify_trace_metrics(store)


class SheetBuildTests(unittest.TestCase):
    """Verifies the gate and, with the run-of-record evidence present, the full render."""

    def test_a_failing_invariant_blocks_the_build(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "failing invariants"):
            build_sheet(ROOT, [("datasets: sizes, uniqueness, held-out disjoint", False, "")], TASK)

    @unittest.skipUnless((ROOT / "artifacts" / "agent_provenance.json").exists(), "needs the run-of-record evidence")
    def test_sheet_renders_every_placeholder_from_the_evidence(self) -> None:
        results = run_all(ROOT)
        if not all(ok for _, ok, _ in results):
            self.skipTest("an invariant fails in this checkout; scripts/14 shows which")

        tex = build_sheet(ROOT, results, TASK)

        self.assertNotIn("<<", tex)
        self.assertIn(r"\begin{document}", tex)
        self.assertIn("0.781", tex)


if __name__ == "__main__":
    unittest.main()

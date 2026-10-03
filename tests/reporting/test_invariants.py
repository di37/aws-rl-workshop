"""Unit tests for the reproducibility invariant checks."""

from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import pandas as pd

from aws.config import DemoConfig
from aws.records.evidence import EvidenceStore
from aws.reporting.invariants import (
    RUN_OF_RECORD,
    Protocol,
    check_base_model,
    check_comparison,
    check_config_protocol,
    check_costs,
    check_dataset_fingerprints,
    check_datasets,
    check_environment,
    check_inference_protocol,
    check_learning_signal,
    check_no_live_endpoint,
    check_no_signed_links,
    check_notebooks,
    check_report_outputs,
    check_training,
    check_training_metrics,
    check_trajectories_match_metrics,
    run_all,
)
from aws.reporting.repro_artifacts import dataset_fingerprint

PROMPT = "Resolve the customer's internet outage. Customer says: {}"
SMALL = Protocol(training_steps=2, group_size=2, batch_size=2, eval_prompts=1, eval_rollouts=2, live_tickets=2)


def write_prompts(path: Path, texts: list[str]) -> None:
    """Writes a prompt CSV in the project's format."""
    path.write_text("prompt\n" + "".join(f'"{PROMPT.format(t)}"\n' for t in texts))


class TempStoreTest(unittest.TestCase):
    """Provides a temporary evidence store and data folder."""

    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.store = EvidenceStore(self.root / "artifacts")
        self.data = self.root / "data"
        self.data.mkdir()

    def tearDown(self) -> None:
        self.directory.cleanup()


class ConfigAndDataTests(TempStoreTest):
    """Verifies the protocol settings, prompt sets, and their fingerprints."""

    def test_config_must_match_the_protocol(self) -> None:
        self.assertTrue(check_config_protocol(DemoConfig())[0])
        ok, detail = check_config_protocol(replace(DemoConfig(), training_steps=3))
        self.assertFalse(ok)
        self.assertIn("training_steps", detail)

    def test_disjoint_sets_of_the_expected_size_pass_and_overlap_fails(self) -> None:
        write_prompts(self.data / "training_prompts.csv", [f"train {i}" for i in range(64)])
        write_prompts(self.data / "evaluation_prompts.csv", [f"eval {i}" for i in range(32)])
        self.assertTrue(check_datasets(self.data)[0])

        write_prompts(self.data / "evaluation_prompts.csv", [f"eval {i}" for i in range(31)] + ["train 0"])
        ok, detail = check_datasets(self.data)
        self.assertFalse(ok)
        self.assertIn("1 overlap", detail)

    def test_fingerprints_must_match_the_run_of_record(self) -> None:
        write_prompts(self.data / "training_prompts.csv", ["a"])
        expected = {"training_prompts.csv": dataset_fingerprint(self.data / "training_prompts.csv")["sha256"]}
        self.assertTrue(check_dataset_fingerprints(self.data, expected)[0])

        write_prompts(self.data / "training_prompts.csv", ["a", "edited"])
        ok, detail = check_dataset_fingerprints(self.data, expected)
        self.assertFalse(ok)
        self.assertIn("training_prompts.csv", detail)


class TrainingCheckTests(TempStoreTest):
    """Verifies training is checked against what the job itself recorded."""

    def save_job(self, batch: str = "2", base: str = RUN_OF_RECORD.base_model) -> None:
        self.store.save("training_job.json", {
            "job_name": "job-1", "job_status": "Completed", "output_model_package_arn": "arn:package/1",
            "progress_info": {"CurrentStep": 2, "MaxSteps": 2, "BatchSize": int(batch)},
            "training_config": {"BaseModelArn": f"arn:hub-content/SageMakerPublicHub/Model/{base}",
                                "HyperParameters": {"max_steps": "2", "group_size": "2",
                                                    "global_batch_size": batch, "sampling_max_tokens": "512"}}})

    def test_job_with_the_protocol_hyperparameters_passes(self) -> None:
        self.save_job()
        self.assertTrue(check_training(self.store, SMALL)[0])

    def test_job_with_another_batch_size_fails(self) -> None:
        self.save_job(batch="16")
        ok, detail = check_training(self.store, SMALL)
        self.assertFalse(ok)
        self.assertIn("global_batch_size", detail)

    def test_base_model_version_must_match(self) -> None:
        self.save_job(base="openai-reasoning-gpt-oss-20b/3.50.0")
        self.assertFalse(check_base_model(self.store)[0])

    def test_metrics_must_belong_to_the_job_and_have_full_steps(self) -> None:
        self.save_job()
        steps = [{"step": i, "training/num_trajectories": 4.0} for i in (1, 2)]
        self.store.save("training_metrics.json", {"job_name": "job-1", "steps": steps})
        self.assertTrue(check_training_metrics(self.store, SMALL)[0])

        self.store.save("training_metrics.json", {"job_name": "other-job", "steps": steps})
        self.assertFalse(check_training_metrics(self.store, SMALL)[0])
        self.store.save("training_metrics.json", {"job_name": "job-1", "steps": [
            {"step": 1, "training/num_trajectories": 4.0}, {"step": 2, "training/num_trajectories": 1.0}]})
        self.assertFalse(check_training_metrics(self.store, SMALL)[0])


class EvaluationCheckTests(TempStoreTest):
    """Verifies the baseline signal, comparison shape, and trace recomputation."""

    def save_comparison(self, base: list[float], tuned: list[float]) -> None:
        def metrics(rewards: list[float]) -> dict:
            solved = sum(1 for r in rewards if r == 1.0)
            return {"metrics": {"eval/reward/succeeded_rollouts": solved, "eval/reward/pass_at_1": solved / 2,
                                "eval/reward/mean": sum(rewards) / 2, "eval/reward/num_prompts": 1,
                                "eval/reward/rollouts_per_prompt": 2}}

        self.store.save("comparison_evaluation_metrics.json", {"base": metrics(base), "fine_tuned": metrics(tuned)})

    def save_traces(self, base: list[float], tuned: list[float]) -> None:
        self.store.save("evaluation_trajectories.json", {"pairs": [
            {"ticket": "a", "before": [{"reward": r} for r in base], "after": [{"reward": r} for r in tuned]}]})

    def test_traces_that_recompute_the_metrics_pass(self) -> None:
        self.save_comparison([1.0, 0.75], [1.0, 1.0])
        self.save_traces([1.0, 0.75], [1.0, 1.0])
        self.assertTrue(check_trajectories_match_metrics(self.store, SMALL)[0])

    def test_deleted_failed_rollouts_fail(self) -> None:
        self.save_comparison([1.0, 0.75], [1.0, 1.0])
        self.save_traces([1.0], [1.0, 1.0])
        ok, detail = check_trajectories_match_metrics(self.store, SMALL)
        self.assertFalse(ok)
        self.assertIn("base: 1 traces", detail)

    def test_traces_with_a_different_mean_fail(self) -> None:
        self.save_comparison([1.0, 0.75], [1.0, 1.0])
        self.save_traces([1.0, 0.5], [1.0, 1.0])
        self.assertFalse(check_trajectories_match_metrics(self.store, SMALL)[0])

    def test_comparison_needs_success_and_the_protocol_shape(self) -> None:
        self.save_comparison([1.0, 0.75], [1.0, 1.0])
        self.store.save("comparison_evaluation.json", {"status": "Succeeded"})
        self.assertTrue(check_comparison(self.store, SMALL)[0])
        self.assertFalse(check_comparison(self.store, RUN_OF_RECORD)[0])

    def test_learning_signal_needs_reward_variance(self) -> None:
        self.store.save("base_evaluation_metrics.json", {"eval/reward/min": 0.75, "eval/reward/max": 1})
        self.assertTrue(check_learning_signal(self.store)[0])
        self.store.save("base_evaluation_metrics.json", {"eval/reward/min": 1, "eval/reward/max": 1})
        self.assertFalse(check_learning_signal(self.store)[0])


class InferenceAndEndpointTests(TempStoreTest):
    """Verifies the live-inference protocol and the endpoint state."""

    def setUp(self) -> None:
        super().setUp()
        self.held_out = ["held out 0", "held out 1", "held out 2"]
        write_prompts(self.data / "evaluation_prompts.csv", self.held_out)

    def save_live(self, seed: int = 2026, after: dict | None = None) -> None:
        tickets = [{"ticket": PROMPT.format(t), "before": {"reward": 0.75}, "after": after or {"reward": 1.0}}
                   for t in self.held_out[:2]]
        self.store.save("inference_before_after.json", {"seed": seed, "tickets": tickets})

    def test_first_held_out_tickets_with_the_seed_pass(self) -> None:
        self.save_live()
        self.assertTrue(check_inference_protocol(self.store, self.data, SMALL)[0])

    def test_wrong_seed_fails(self) -> None:
        self.save_live(seed=1)
        self.assertFalse(check_inference_protocol(self.store, self.data, SMALL)[0])

    def test_failed_model_calls_fail(self) -> None:
        self.save_live(after={"error": "ThrottlingException"})
        ok, detail = check_inference_protocol(self.store, self.data, SMALL)
        self.assertFalse(ok)
        self.assertIn("0:after", detail)

    def test_live_endpoint_fails_and_deleted_or_absent_passes(self) -> None:
        self.assertTrue(check_no_live_endpoint(self.store)[0])
        self.store.save("endpoint.json", {"status": "InService"})
        self.assertFalse(check_no_live_endpoint(self.store)[0])
        self.store.save("endpoint.json", {"status": "Deleted"})
        self.assertTrue(check_no_live_endpoint(self.store)[0])


class CostCheckTests(TempStoreTest):
    """Verifies cost lines are recomputed, complete, published, and within the cap."""

    def setUp(self) -> None:
        super().setUp()
        usage = {"PrefillTokenCount": 1_000_000}
        self.items = [{"item": label, "component": "Evaluation", "usage": usage, "usd": 0.12,
                       "source": "tokens"} for label in ("Base evaluation", "Comparison: base model",
                                                         "Comparison: fine-tuned model")]
        self.items.append({"item": "Training", "component": "Finetuning", "usage": usage, "usd": 0.12,
                           "source": "tokens"})
        self.items.append({"item": "Copy", "usd": 0.84, "source": "computed"})
        self.store.save("extra_costs.json", {"items": [{"label": "Copy", "usd": 0.84}]})

    def publish(self, items: list[dict]) -> None:
        self.store.save("cost_accounting.json", {"items": items, "rates_usd_per_million_tokens": {
            "evaluation_prefill": "0.12", "training_prefill": "0.12"}})
        repro = self.root / "reports" / "repro"
        repro.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{k: i[k] for k in ("item", "usd", "source")} for i in items]).to_csv(
            repro / "compute_accounting.csv", index=False)

    def test_consistent_accounting_passes(self) -> None:
        self.publish(self.items)
        ok, detail = check_costs(self.root, self.store, 25)
        self.assertTrue(ok, detail)

    def test_mispriced_line_fails(self) -> None:
        self.publish([{**self.items[0], "usd": 0.01}, *self.items[1:]])
        self.assertFalse(check_costs(self.root, self.store, 25)[0])

    def test_extra_cost_missing_from_the_snapshot_fails(self) -> None:
        self.publish(self.items[:-1])
        self.assertFalse(check_costs(self.root, self.store, 25)[0])

    def test_hand_edited_report_fails(self) -> None:
        self.publish(self.items)
        path = self.root / "reports" / "repro" / "compute_accounting.csv"
        path.write_text(path.read_text().replace("0.84", "0.04"))
        ok, detail = check_costs(self.root, self.store, 25)
        self.assertFalse(ok)
        self.assertIn("current: False", detail)


class FileCheckTests(TempStoreTest):
    """Verifies scans fail on nothing, find problems, and outputs match a rebuild."""

    def write_notebook(self, source: str, text: str) -> Path:
        path = self.root / "demo.ipynb"
        output = {"output_type": "display_data", "data": {"text/plain": [text]}, "metadata": {}}
        cell = {"cell_type": "code", "source": [source], "outputs": [output], "metadata": {}, "execution_count": 1}
        path.write_text(json.dumps({"cells": [cell], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}))
        return path

    def test_signed_links_are_found_and_an_empty_scan_fails(self) -> None:
        clean, leaky = self.root / "clean.json", self.root / "leaky.log"
        clean.write_text(json.dumps({"note": "nothing here"}))
        leaky.write_text("MLflow presigned URL: https://a.example/auth?authToken=abc123")

        self.assertTrue(check_no_signed_links([clean])[0])
        self.assertIn("leaky.log", check_no_signed_links([clean, leaky])[1])
        self.assertFalse(check_no_signed_links([])[0])

    def test_ansi_styled_traceback_and_syntax_errors_are_found(self) -> None:
        styled = "\x1b[1;31mTrace\x1b[0mback \x1b[2m(most recent call last)\x1b[0m"
        self.assertFalse(check_notebooks([self.write_notebook("x = 1\n", styled)])[0])
        self.assertIn("syntax", check_notebooks([self.write_notebook("def (:\n", "")])[1])
        self.assertTrue(check_notebooks([self.write_notebook("x = 1\n", "\x1b[32mok\x1b[0m")])[0])
        self.assertFalse(check_notebooks([])[0])

    def test_environment_mismatch_fails(self) -> None:
        pins = self.root / "requirements.txt"
        pins.write_text("pytest==0.0.1\n")
        ok, detail = check_environment(pins)
        self.assertFalse(ok)
        self.assertIn("pytest", detail)

    def test_tables_must_equal_a_rebuild_and_figures_must_exist(self) -> None:
        tables = {"a.csv": [{"x": 1}]}
        (self.root / "reports" / "tables").mkdir(parents=True)
        (self.root / "reports" / "figures").mkdir(parents=True)
        (self.root / "reports" / "figures" / "b.png").write_bytes(b"png")
        (self.root / "reports" / "tables" / "a.csv").write_text("x\n2\n")
        self.assertFalse(check_report_outputs(self.root, tables, ["b.png"])[0])

        (self.root / "reports" / "tables" / "a.csv").write_text("x\n1\n")
        self.assertTrue(check_report_outputs(self.root, tables, ["b.png"])[0])
        self.assertFalse(check_report_outputs(self.root, tables, ["missing.png"])[0])


class RunAllTests(unittest.TestCase):
    """Verifies every registered check reports a label, a verdict, and detail, and fails on nothing."""

    def test_run_all_on_an_empty_project_fails_every_evidence_check(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            results = run_all(Path(directory))

        self.assertEqual(len(results), 19)
        for label, ok, detail in results:
            self.assertIsInstance(label, str)
            self.assertIsInstance(detail, str)
            if label not in ("settings in aws/config.py match the documented protocol",
                             "no endpoint left running", "environment matches the exact pins"):
                self.assertFalse(ok, label)


if __name__ == "__main__":
    unittest.main()

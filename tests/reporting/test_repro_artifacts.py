"""Unit tests for building the reproducibility record."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from aws.config import DemoConfig
from aws.records.evidence import EvidenceStore
from aws.reporting.repro_artifacts import (
    RUN_COMMANDS,
    dataset_fingerprint,
    environment_versions,
    pinned_requirements,
    requirements_file,
    study_metadata,
    write_repro_record,
)


class FingerprintTests(unittest.TestCase):
    """Verifies dataset fingerprints are stable and content-based."""

    def test_fingerprint_counts_rows_and_hashes_content(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prompts.csv"
            path.write_text('prompt\n"a"\n"b"\n"a"\n')

            first = dataset_fingerprint(path)
            path.write_text('prompt\n"a"\n"b"\n"c"\n')
            second = dataset_fingerprint(path)

        self.assertEqual((first["rows"], first["unique_rows"]), (3, 2))
        self.assertEqual(len(first["sha256"]), 64)
        self.assertNotEqual(first["sha256"], second["sha256"])


class EnvironmentTests(unittest.TestCase):
    """Verifies pins are parsed and compared with installed versions."""

    def test_pins_are_parsed_with_extras_and_comments_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "requirements.txt"
            path.write_text("# comment\nstrands-agents[openai]==1.57.2\npytest==9.1.1\n")

            self.assertEqual(
                pinned_requirements(path), {"strands-agents": "1.57.2", "pytest": "9.1.1"}
            )

    def test_environment_rows_flag_mismatches(self) -> None:
        rows = environment_versions({"pytest": "0.0.1"})

        python_row = rows[0]
        pytest_row = next(r for r in rows if r["package"] == "pytest")
        self.assertEqual(python_row["package"], "python")
        self.assertEqual(pytest_row["pinned"], "0.0.1")
        self.assertFalse(pytest_row["matches_pin"])


class RequirementsFileTests(unittest.TestCase):
    """Verifies the pinned file matches the operating system."""

    def test_macos_and_linux_use_their_own_pins(self) -> None:
        root = Path("/project")
        with patch("aws.reporting.repro_artifacts.platform.system", return_value="Darwin"):
            self.assertEqual(requirements_file(root).name, "requirements-macos.txt")
        with patch("aws.reporting.repro_artifacts.platform.system", return_value="Linux"):
            self.assertEqual(requirements_file(root).name, "requirements-linux.txt")


class StudyMetadataTests(unittest.TestCase):
    """Verifies the metadata carries data, settings, seeds, runs, and results."""

    def test_metadata_sections_come_from_evidence_and_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data").mkdir()
            for name in ("training_prompts.csv", "evaluation_prompts.csv"):
                (root / "data" / name).write_text('prompt\n"a"\n')
            store = EvidenceStore(root / "artifacts")
            store.save("agent_provenance.json", {
                "runtime": {"version": "7"}, "image": {"digest": "sha256:c", "base_image": "python@sha256:a"},
                "source_bundle": {"sha256": "abc"}})
            store.save("training_job.json", {
                "job_name": "job-1", "job_status": "Completed",
                "training_config": {"BaseModelArn": "arn:hub-content/m/3.43.0",
                                    "HyperParameters": {"max_steps": "10"}}})
            snapshot = {"billed_tokens_usd": 3.0, "computed_or_estimated_usd": 1.0, "total_usd": 4.0,
                        "total_with_allowance_usd": 6.0, "budget_cap_usd": 25}

            metadata = study_metadata(root, store, DemoConfig(), snapshot,
                                      task={"success_path": ["a", "b"], "max_turns": 4})

        self.assertEqual(metadata["datasets"][0]["file"], "training_prompts.csv")
        self.assertEqual(len(metadata["datasets"][0]["sha256"]), 64)
        self.assertEqual(metadata["training"]["job_hyperparameters"], {"max_steps": "10"})
        self.assertEqual(metadata["training"]["base_model_hub_content"], "arn:hub-content/m/3.43.0")
        self.assertEqual(metadata["seeds"]["live_inference_seed"], 2026)
        self.assertEqual(metadata["task"]["max_turns"], 4)
        self.assertEqual(metadata["cost_usd"]["total_usd"], 4.0)
        self.assertEqual(metadata["cost_usd"]["billed_tokens_usd"], 3.0)
        self.assertIn("run_of_record", metadata)
        self.assertEqual(metadata["provenance"]["agent_runtime_version"], "7")
        self.assertEqual(metadata["provenance"]["agent_source_sha256"], "abc")
        self.assertIn("results", metadata)


class WriteRecordTests(unittest.TestCase):
    """Verifies the five CS7641-style repro files are written."""

    def test_record_files_are_written(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repro = Path(directory) / "reports" / "repro"
            write_repro_record(
                repro,
                environment=[{"package": "python", "version": "3.12", "pinned": "", "matches_pin": True}],
                metadata={"study": "x"},
                cost_lines=[{"item": "Training", "usd": 2.83, "source": "BillableTokenUsage"}],
                inventory=[{"kind": "evidence", "file": "artifacts/a.json"}],
            )

            names = sorted(p.name for p in repro.iterdir())
            commands = list(csv.DictReader((repro / "run_commands.csv").open()))
            metadata = json.loads((repro / "study_metadata.json").read_text())

        self.assertEqual(names, ["artifact_inventory.csv", "compute_accounting.csv",
                                 "environment_versions.csv", "run_commands.csv",
                                 "study_metadata.json"])
        self.assertEqual(len(commands), len(RUN_COMMANDS))
        self.assertEqual(commands[0]["command"], RUN_COMMANDS[0][1])
        self.assertEqual(metadata["study"], "x")


if __name__ == "__main__":
    unittest.main()

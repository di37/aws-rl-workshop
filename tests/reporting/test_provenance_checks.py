"""Unit tests for the provenance invariants (what ran is what the repository says)."""

from __future__ import annotations

import hashlib
import tempfile
import unittest
import zipfile
from pathlib import Path

from aws.records.evidence import EvidenceStore
from aws.records.provenance import (
    AGENT_RECORD,
    AGENT_SOURCE,
    DATASET_RECORD,
    IMPORT_RECORD,
    zip_hashes,
)
from aws.reporting.provenance_checks import (
    check_agent_provenance,
    check_dataset_provenance,
    check_deployment,
)

APP = 'SYSTEM_PROMPT = """Fix the ticket."""\n'
FILES = {"app.py": APP, "environment.py": "STATES = 1\n", "rollout_driver.py": "DRIVER = 1\n",
         "Dockerfile": "FROM python:3.12-slim\n"}
JOBS = [{"job": "Base evaluation", "started": "2026-10-01T23:14:50+00:00"},
        {"job": "Training", "started": "2026-10-02T00:05:58+00:00"},
        {"job": "Comparison evaluation", "started": "2026-10-02T00:26:56+00:00"}]
PACKAGE = "arn:aws:sagemaker:us-west-2:1:model-package/group/1"


class ProvenanceTest(unittest.TestCase):
    """Builds a temporary project with an agent, a deployed bundle, and its record."""

    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.store = EvidenceStore(self.root / "artifacts")
        (self.root / "agent").mkdir()
        for name, text in FILES.items():
            (self.root / "agent" / name).write_text(text)
        (self.root / "agent" / "requirements.lock.txt").write_text("# lock\nboto3==1.0\n")
        self.store.save("placeholder.json", {})
        bundle = self.root / "artifacts" / AGENT_SOURCE
        with zipfile.ZipFile(bundle, "w") as archive:
            for name, text in FILES.items():
                archive.writestr(name, text)
        self.record = {
            "runtime": {"version": "7", "last_updated": "2026-10-01T23:13:23+00:00"},
            "image": {"digest": "sha256:" + "c" * 64},
            "build": {"later_builds": 0},
            "source_bundle": {"sha256": hashlib.sha256(bundle.read_bytes()).hexdigest(),
                              "files": zip_hashes(bundle)},
            "packages": {"boto3": "1.0", "sagemaker-mtrl-support-agent": "0.1.0"},
            "jobs": JOBS,
        }
        self.store.save(AGENT_RECORD, self.record)

    def tearDown(self) -> None:
        self.directory.cleanup()


class AgentProvenanceTests(ProvenanceTest):
    """Verifies the agent check catches edited code, prompts, packages, and stale jobs."""

    def test_matching_agent_passes(self) -> None:
        ok, detail = check_agent_provenance(self.root, self.store)
        self.assertTrue(ok, detail)

    def test_edited_driver_fails(self) -> None:
        (self.root / "agent" / "rollout_driver.py").write_text("DRIVER = 2\n")
        ok, detail = check_agent_provenance(self.root, self.store)
        self.assertFalse(ok)
        self.assertIn("rollout_driver.py", detail)

    def test_changed_system_prompt_fails_but_a_refactor_does_not(self) -> None:
        (self.root / "agent" / "app.py").write_text(APP + "\ndef run_episode():\n    return 1\n")
        self.assertTrue(check_agent_provenance(self.root, self.store)[0])
        (self.root / "agent" / "app.py").write_text('SYSTEM_PROMPT = """Restart the router."""\n')
        self.assertIn("system prompt", check_agent_provenance(self.root, self.store)[1])

    def test_job_started_before_the_runtime_update_fails(self) -> None:
        self.store.save(AGENT_RECORD, {**self.record, "jobs": [
            {**JOBS[0], "started": "2026-10-01T23:00:00+00:00"}, *JOBS[1:]]})
        self.assertFalse(check_agent_provenance(self.root, self.store)[0])

    def test_lock_that_differs_from_the_image_fails(self) -> None:
        (self.root / "agent" / "requirements.lock.txt").write_text("boto3==2.0\n")
        self.assertIn("requirements.lock.txt", check_agent_provenance(self.root, self.store)[1])

    def test_missing_record_fails(self) -> None:
        (self.root / "artifacts" / AGENT_RECORD).unlink()
        self.assertFalse(check_agent_provenance(self.root, self.store)[0])


class DatasetProvenanceTests(ProvenanceTest):
    """Verifies the datasets matched the CSVs and were written before the jobs read them."""

    def save_splits(self, training_modified: str, match: bool = True) -> None:
        self.store.save(DATASET_RECORD, {"splits": [
            {"split": "training", "last_modified": training_modified, "prompts_match_csv": match},
            {"split": "evaluation", "last_modified": "2026-10-01T22:37:50+00:00", "prompts_match_csv": True}]})

    def test_datasets_written_before_the_jobs_pass(self) -> None:
        self.save_splits("2026-10-02T00:05:39+00:00")
        ok, detail = check_dataset_provenance(self.store)
        self.assertTrue(ok, detail)

    def test_dataset_written_after_training_started_fails(self) -> None:
        self.save_splits("2026-10-02T00:10:00+00:00")
        self.assertIn("training: not written before", check_dataset_provenance(self.store)[1])

    def test_dataset_that_differs_from_the_csv_fails(self) -> None:
        self.save_splits("2026-10-02T00:05:39+00:00", match=False)
        self.assertFalse(check_dataset_provenance(self.store)[0])


class DeploymentTests(ProvenanceTest):
    """Verifies the deployed model must trace back to the trained package."""

    def save_import(self, weight_verdict: str = "same size", package: str = PACKAGE) -> None:
        self.store.save("training_job.json", {"output_model_package_arn": PACKAGE})
        self.store.save("bedrock_import.json", {"source_uri": "s3://copy/"})
        self.store.save(IMPORT_RECORD, {
            "model_package_arn": package,
            "import_job": {"status": "Completed", "source_uri": "s3://copy/"},
            "files": [{"file": "model-00000-of-00001.safetensors", "verdict": weight_verdict},
                      {"file": "config.json", "verdict": "identical"},
                      {"file": "tokenizer_config.json", "verdict": "changed on purpose"}]})

    def test_import_of_the_trained_weights_passes(self) -> None:
        self.save_import()
        ok, detail = check_deployment(self.store)
        self.assertTrue(ok, detail)

    def test_weight_mismatch_or_another_package_fails(self) -> None:
        self.save_import(weight_verdict="DIFFERENT")
        self.assertFalse(check_deployment(self.store)[0])
        self.save_import(package="arn:other")
        self.assertFalse(check_deployment(self.store)[0])

    def test_no_trained_package_fails_even_without_records(self) -> None:
        self.assertFalse(check_deployment(self.store)[0])

    def test_endpoint_that_served_the_package_passes_without_an_import(self) -> None:
        self.store.save("training_job.json", {"output_model_package_arn": PACKAGE})
        self.store.save("endpoint.json", {"model_package_arn": PACKAGE, "ready_at": "2026-10-02T05:00:00+00:00"})
        self.assertTrue(check_deployment(self.store)[0])


if __name__ == "__main__":
    unittest.main()

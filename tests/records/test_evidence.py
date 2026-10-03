"""Unit tests for redacted evidence persistence and notebook scrubbing."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from aws.records.evidence import REDACTED, EvidenceStore, log_safe, scrub_notebook

PRESIGNED = "https://app-1.mlflow.sagemaker.us-west-2.app.aws/auth?authToken=abc.def"


class EvidenceStoreTests(unittest.TestCase):
    """Verifies JSON evidence round-trips without leaking signed URLs."""

    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.store = EvidenceStore(Path(self.directory.name))

    def tearDown(self) -> None:
        self.directory.cleanup()

    def test_save_and_load_round_trip(self) -> None:
        self.store.save("job.json", {"job_name": "mtrl-1", "steps": 10})

        self.assertEqual(
            self.store.load("job.json"), {"job_name": "mtrl-1", "steps": 10}
        )

    def test_missing_evidence_loads_as_none(self) -> None:
        self.assertIsNone(self.store.load("absent.json"))

    def test_presigned_urls_are_redacted_on_save(self) -> None:
        self.store.save("eval.json", {"note": f"open {PRESIGNED} now"})

        saved = (Path(self.directory.name) / "eval.json").read_text()

        self.assertNotIn("authToken", saved)
        self.assertIn(REDACTED, saved)

    def test_redact_handles_aws_signature_urls(self) -> None:
        url = "https://bucket.s3.amazonaws.com/key?X-Amz-Signature=abc&X-Amz-Credential=x"

        self.assertEqual(EvidenceStore.redact(f"see {url}"), f"see {REDACTED}")

    def test_plain_urls_are_kept(self) -> None:
        text = "https://docs.aws.amazon.com/sagemaker/latest/dg/model-customize-mtrl.html"

        self.assertEqual(EvidenceStore.redact(text), text)

    def test_allowlisted_keeps_only_safe_scalar_fields(self) -> None:
        source = SimpleNamespace(
            arn="arn:aws:sagemaker:us-west-2:1:pipeline/x",
            mlflow_url=PRESIGNED,
            status={"nested": True},
        )

        payload = EvidenceStore.allowlisted(source, {"arn", "status"})

        self.assertEqual(payload, {"arn": "arn:aws:sagemaker:us-west-2:1:pipeline/x"})

    def test_allowlisted_accepts_mappings(self) -> None:
        payload = EvidenceStore.allowlisted({"job_name": "j", "secret": "s"}, {"job_name"})

        self.assertEqual(payload, {"job_name": "j"})


class RedactionCoverageTests(unittest.TestCase):
    """Verifies redaction never corrupts JSON and covers common token formats."""

    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.store = EvidenceStore(Path(self.directory.name))

    def tearDown(self) -> None:
        self.directory.cleanup()

    def test_url_followed_by_quote_keeps_json_valid(self) -> None:
        self.store.save("q.json", {"note": f'{PRESIGNED}"quoted" text'})

        loaded = self.store.load("q.json")

        self.assertEqual(loaded["note"], f'{REDACTED}"quoted" text')

    def test_nested_values_are_redacted(self) -> None:
        self.store.save("n.json", {"a": [{"b": PRESIGNED}], "c": 3})

        self.assertEqual(self.store.load("n.json"), {"a": [{"b": REDACTED}], "c": 3})

    def test_generic_token_and_signature_parameters_are_redacted(self) -> None:
        for url in (
            "https://host.example/x?token=abc123",
            "https://host.example/x?Signature=abc&Expires=1",
        ):
            self.assertEqual(EvidenceStore.redact(url), REDACTED)

    def test_bare_jwt_is_redacted(self) -> None:
        jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.c2lnbmF0dXJlLXBhcnQ"

        self.assertEqual(EvidenceStore.redact(f"token {jwt} end"), f"token {REDACTED} end")

    def test_writes_are_atomic(self) -> None:
        self.store.save("a.json", {"v": 1})

        leftovers = [p.name for p in Path(self.directory.name).iterdir() if p.name != "a.json"]

        self.assertEqual(leftovers, [])

    def test_names_lists_matching_evidence(self) -> None:
        self.store.save("training_job.json", {})
        self.store.save("training_job.old-1.json", {})

        self.assertEqual(self.store.names("training_job.*.json"), ["training_job.old-1.json"])


class ScrubNotebookTests(unittest.TestCase):
    """Verifies signed URLs are removed from every notebook output type."""

    def test_stream_html_and_plain_outputs_are_scrubbed(self) -> None:
        notebook = {
            "cells": [
                {
                    "cell_type": "code",
                    "source": ["job.wait()"],
                    "outputs": [
                        {"output_type": "stream", "text": [f"MLflow: {PRESIGNED}\n"]},
                        {
                            "output_type": "display_data",
                            "data": {
                                "text/html": [f'<a href="{PRESIGNED}">MLflow</a>'],
                                "text/plain": [PRESIGNED],
                            },
                        },
                    ],
                },
                {"cell_type": "markdown", "source": ["# Title"]},
            ],
            "metadata": {},
            "nbformat": 4,
            "nbformat_minor": 5,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.ipynb"
            path.write_text(json.dumps(notebook))

            removed = scrub_notebook(path)
            text = path.read_text()

        self.assertEqual(removed, 3)
        self.assertNotIn("authToken", text)
        self.assertEqual(text.count(REDACTED), 3)

    def test_errors_metadata_and_json_outputs_are_scrubbed(self) -> None:
        notebook = {
            "cells": [{
                "cell_type": "code", "source": [""], "metadata": {"note": PRESIGNED},
                "outputs": [
                    {"output_type": "error", "ename": "E", "evalue": PRESIGNED,
                     "traceback": [f"line {PRESIGNED}"]},
                    {"output_type": "display_data",
                     "data": {"application/json": {"url": PRESIGNED}}},
                ],
            }],
            "metadata": {"widgets": {"state": {"x": PRESIGNED}}},
            "nbformat": 4, "nbformat_minor": 5,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.ipynb"
            path.write_text(json.dumps(notebook))

            removed = scrub_notebook(path)
            text = path.read_text()

        self.assertEqual(removed, 5)
        self.assertNotIn("authToken", text)


class ArchiveAndLogSafetyTests(unittest.TestCase):
    """Verifies records are moved aside without clobbering, and logs are made safe."""

    def test_archive_moves_a_record_and_refuses_to_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = EvidenceStore(Path(directory))
            store.save("base_evaluation.json", {"arn": "a"})

            self.assertEqual(store.archive("base_evaluation.json", "x1"), "base_evaluation.x1.json")
            self.assertIsNone(store.load("base_evaluation.json"))
            self.assertIsNone(store.archive("base_evaluation.json", "x1"))
            store.save("base_evaluation.json", {"arn": "b"})
            with self.assertRaises(FileExistsError):
                store.archive("base_evaluation.json", "x1")

    def test_notebook_scrub_hides_the_home_folder_without_counting_it_as_a_link(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "demo.ipynb"
            path.write_text(json.dumps({"cells": [{"outputs": [{"text": f"config at {Path.home()}/x"}]}]}))

            removed = scrub_notebook(path)
            text = path.read_text()

        self.assertEqual(removed, 0)
        self.assertNotIn(str(Path.home()), text)
        self.assertIn("~/x", text)

    def test_log_safe_redacts_links_and_the_home_folder(self) -> None:
        text = f"MLflow presigned URL: https://x.example/auth?authToken=abc123 at {Path.home()}/project"

        safe = log_safe(text)

        self.assertNotIn("abc123", safe)
        self.assertNotIn(str(Path.home()), safe)
        self.assertIn("~/project", safe)

if __name__ == "__main__":
    unittest.main()

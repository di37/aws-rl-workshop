"""Unit tests for prompt dataset validation and idempotent upload."""

from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from botocore.exceptions import ClientError

from aws.setup.prompt_datasets import PromptDatasets


class PromptDatasetsTests(unittest.TestCase):
    """Verifies uploads happen only when bytes change, and CSV validation."""

    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.s3 = MagicMock()
        self.datasets = PromptDatasets(self.s3, "demo-bucket", self.root)

    def tearDown(self) -> None:
        self.directory.cleanup()

    def _file(self, content: bytes) -> Path:
        path = self.root / "prompts.parquet"
        path.write_bytes(content)
        return path

    def test_unchanged_dataset_is_not_reuploaded(self) -> None:
        self.s3.head_object.return_value = {
            "ETag": '"' + hashlib.md5(b"same-bytes").hexdigest() + '"'
        }

        uploaded = self.datasets.upload_if_changed(self._file(b"same-bytes"), "d/p.parquet")

        self.assertFalse(uploaded)
        self.s3.upload_file.assert_not_called()

    def test_changed_dataset_is_uploaded(self) -> None:
        self.s3.head_object.return_value = {"ETag": '"stale"'}
        path = self._file(b"new-bytes")

        self.assertTrue(self.datasets.upload_if_changed(path, "d/p.parquet"))
        self.s3.upload_file.assert_called_once_with(str(path), "demo-bucket", "d/p.parquet")

    def test_missing_dataset_object_is_uploaded(self) -> None:
        self.s3.head_object.side_effect = ClientError(
            {"Error": {"Code": "404", "Message": "Not Found"}}, "HeadObject"
        )

        self.assertTrue(self.datasets.upload_if_changed(self._file(b"x"), "d/p.parquet"))

    def test_other_s3_errors_propagate(self) -> None:
        self.s3.head_object.side_effect = ClientError(
            {"Error": {"Code": "AccessDenied", "Message": "no"}}, "HeadObject"
        )

        with self.assertRaises(ClientError):
            self.datasets.upload_if_changed(self._file(b"x"), "d/p.parquet")

    def test_validate_csv_counts_unique_prompts(self) -> None:
        path = self.root / "prompts.csv"
        path.write_text('prompt\n"a"\n"b"\n"a"\n')

        self.assertEqual(PromptDatasets.validate_csv(path, minimum=2), 2)
        with self.assertRaises(ValueError):
            PromptDatasets.validate_csv(path, minimum=3)

    def test_validate_csv_requires_single_prompt_column(self) -> None:
        path = self.root / "bad.csv"
        path.write_text("prompt,extra\nx,y\n")

        with self.assertRaisesRegex(ValueError, "only a prompt column"):
            PromptDatasets.validate_csv(path, minimum=1)


if __name__ == "__main__":
    unittest.main()

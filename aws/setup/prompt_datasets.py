"""Prompt dataset validation, Parquet conversion, and idempotent S3 upload."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Any

from botocore.exceptions import ClientError

_MISSING_CODES = {"404", "NoSuchKey", "NotFound"}


class PromptDatasets:
    """Validates prompt CSVs and keeps their Parquet copies in S3 current."""

    def __init__(self, s3: Any, bucket: str, artifacts_dir: Path) -> None:
        """Initializes the uploader.

        Args:
            s3: Boto3 S3 client.
            bucket: Demo bucket holding the datasets.
            artifacts_dir: Local folder for generated Parquet files.
        """
        self.s3 = s3
        self.bucket = bucket
        self.artifacts_dir = artifacts_dir

    def upload(self, sources: dict[str, tuple[Path, str]]) -> dict[str, dict[str, Any]]:
        """Uploads each prompt file unless S3 already holds identical bytes.

        Skipping matters: the training gate requires the baseline evaluation to
        be newer than the dataset object, so a needless re-upload would block it.

        Args:
            sources: Split name to (local CSV, destination S3 key).

        Returns:
            Per split, the S3 URI and whether a new upload happened.
        """
        report: dict[str, dict[str, Any]] = {}
        for split, (csv_path, key) in sources.items():
            uploaded = self.upload_if_changed(self.write_parquet(csv_path), key)
            report[split] = {"uri": f"s3://{self.bucket}/{key}", "uploaded": uploaded}
        return report

    def upload_if_changed(self, path: Path, key: str) -> bool:
        """Uploads one file unless S3 already holds identical bytes.

        Args:
            path: Local file to upload.
            key: Destination S3 key in the demo bucket.

        Returns:
            True when a new upload happened.
        """
        local_md5 = hashlib.md5(path.read_bytes(), usedforsecurity=False).hexdigest()
        try:
            remote = self.s3.head_object(Bucket=self.bucket, Key=key)
            if remote["ETag"].strip('"') == local_md5:
                return False
        except ClientError as error:
            if error.response["Error"]["Code"] not in _MISSING_CODES:
                raise
        self.s3.upload_file(str(path), self.bucket, key)
        return True

    def write_parquet(self, csv_path: Path) -> Path:
        """Converts prompts to an explicitly typed Parquet string column.

        Args:
            csv_path: Source CSV with one ``prompt`` column.

        Returns:
            Generated Parquet path in the artifacts directory.
        """
        import pyarrow as pa
        import pyarrow.parquet as parquet

        with csv_path.open(newline="") as handle:
            prompts = [row["prompt"] for row in csv.DictReader(handle)]
        table = pa.Table.from_arrays(
            [pa.array(prompts, type=pa.string())],
            schema=pa.schema([pa.field("prompt", pa.string())]),
        )
        self.artifacts_dir.mkdir(exist_ok=True)
        destination = self.artifacts_dir / f"{csv_path.stem}.parquet"
        parquet.write_table(table, destination)
        return destination

    @staticmethod
    def validate_csv(path: Path, minimum: int) -> int:
        """Validates prompt count, uniqueness, and the expected header.

        Args:
            path: Local CSV path.
            minimum: Minimum number of unique prompts.

        Returns:
            Unique prompt count.

        Raises:
            ValueError: If validation fails.
        """
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows or set(rows[0]) != {"prompt"}:
            raise ValueError(f"{path} must contain only a prompt column.")
        prompts = {row["prompt"].strip() for row in rows if row["prompt"].strip()}
        if len(prompts) < minimum:
            raise ValueError(
                f"{path} needs at least {minimum} unique prompts; found {len(prompts)}."
            )
        return len(prompts)

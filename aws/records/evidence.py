"""Redacted, atomically written JSON evidence and notebook output scrubbing.

The SageMaker SDK prints presigned MLflow links (they embed a short-lived sign-in
token). Evidence files and executed notebooks are meant to be shared, so every
string is redacted before it is written. To scrub an executed notebook:

    python scripts/dev/scrub_notebook.py notebooks/sagemaker_mtrl_real_demo.executed.ipynb
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

REDACTED = "[presigned URL removed]"
_URL_CHARS = r"[^\s\"'<>\\]"
_SENSITIVE_URL = re.compile(
    rf"https?://{_URL_CHARS}*?[?&#](?:authToken|auth_token|token|access_token|"
    rf"X-Amz-Signature|X-Amz-Security-Token|X-Amz-Credential|Signature)={_URL_CHARS}*",
    re.IGNORECASE,
)
_JWT = re.compile(r"eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}")
_SCALARS = (str, int, float, bool, type(None))


class EvidenceStore:
    """Reads and writes redacted JSON evidence in one directory."""

    def __init__(self, directory: Path) -> None:
        """Initializes the store.

        Args:
            directory: Folder holding evidence files; created on first save.
        """
        self.directory = directory

    def save(self, name: str, payload: Mapping[str, Any]) -> Path:
        """Writes one evidence file atomically, with every string redacted.

        Args:
            name: File name inside the evidence directory.
            payload: JSON-compatible evidence.

        Returns:
            Path of the written file.
        """
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / name
        text = json.dumps(redact_value(payload), indent=2, default=str) + "\n"
        _atomic_write(path, text)
        return path

    def load(self, name: str) -> dict[str, Any] | None:
        """Reads one evidence file.

        Args:
            name: File name inside the evidence directory.

        Returns:
            Parsed evidence, or None when the file does not exist.
        """
        path = self.directory / name
        if not path.exists():
            return None
        return json.loads(path.read_text())

    def names(self, pattern: str) -> list[str]:
        """Lists evidence file names matching a glob pattern.

        Args:
            pattern: Glob such as ``training_job.*.json``.

        Returns:
            Sorted matching file names.
        """
        if not self.directory.exists():
            return []
        return sorted(path.name for path in self.directory.glob(pattern))

    def archive(self, name: str, label: str) -> str | None:
        """Moves a record aside as ``<stem>.<label>.json``, keeping it as evidence.

        Args:
            name: Record file name, such as ``base_evaluation.json``.
            label: Suffix that identifies the archived attempt.

        Returns:
            The archived file name, or None when the record does not exist.

        Raises:
            FileExistsError: If an archive with that label already exists.
        """
        source = self.directory / name
        if not source.exists():
            return None
        target = self.directory / f"{source.stem}.{label}{source.suffix}"
        if target.exists():
            raise FileExistsError(f"{target.name} already exists; choose another label.")
        source.rename(target)
        return target.name

    @staticmethod
    def redact(text: str) -> str:
        """Replaces signed URLs and bare JWTs with a placeholder.

        Args:
            text: Arbitrary text that may contain signed links.

        Returns:
            The text with every signed link and token replaced.
        """
        return _JWT.sub(REDACTED, _SENSITIVE_URL.sub(REDACTED, text))

    @staticmethod
    def allowlisted(source: object, fields: Iterable[str]) -> dict[str, Any]:
        """Copies only allow-listed scalar fields from an SDK object.

        Args:
            source: SDK object or mapping, such as a pipeline execution.
            fields: Names that are safe to persist.

        Returns:
            Scalar values for the allow-listed names that are present.
        """
        values = source if isinstance(source, Mapping) else vars(source)
        allowed = set(fields)
        return {
            key: value
            for key, value in values.items()
            if key in allowed and isinstance(value, _SCALARS)
        }


def log_safe(text: str) -> str:
    """Makes text safe for shared logs: no signed links or tokens, no home folder path.

    Args:
        text: Output written to a log file.

    Returns:
        The text with signed links redacted and the home folder shown as ``~``.
    """
    return EvidenceStore.redact(text).replace(str(Path.home()), "~")


def redact_value(value: Any) -> Any:
    """Returns a copy of a JSON-like value with every string redacted.

    Args:
        value: Nested dicts, lists, and scalars.

    Returns:
        A new value; the input is not modified.
    """
    if isinstance(value, str):
        return EvidenceStore.redact(value)
    if isinstance(value, Mapping):
        return {key: redact_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact_value(item) for item in value]
    return value


def scrub_notebook(path: Path) -> int:
    """Removes signed URLs from every string in a notebook, outputs included.

    The home folder path (which shows the local user name) is also shown as ``~``.

    Args:
        path: Notebook file to rewrite in place.

    Returns:
        Number of strings that contained a signed link or token.
    """
    notebook = json.loads(path.read_text())
    counter = [0]
    cleaned = _scrub(notebook, counter)
    _atomic_write(path, json.dumps(cleaned, indent=1, ensure_ascii=False) + "\n")
    return counter[0]


def _scrub(value: Any, counter: list[int]) -> Any:
    """Recursively redacts strings and counts the changed ones.

    Args:
        value: Notebook JSON value.
        counter: Single-item tally of changed strings, shared by the walk.

    Returns:
        The redacted copy of ``value``.
    """
    if isinstance(value, str):
        cleaned = EvidenceStore.redact(value)
        counter[0] += cleaned != value
        return cleaned.replace(str(Path.home()), "~")
    if isinstance(value, dict):
        return {key: _scrub(item, counter) for key, item in value.items()}
    if isinstance(value, list):
        return [_scrub(item, counter) for item in value]
    return value


def _atomic_write(path: Path, text: str) -> None:
    """Writes a file so readers never observe partial content.

    Args:
        path: Destination file.
        text: Full file content.
    """
    handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(handle, "w") as stream:
            stream.write(text)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise

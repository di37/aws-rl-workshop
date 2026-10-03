"""Shared setup for the numbered scripts; every script imports this module first.

Puts the project (the ``aws`` package) and ``agent/`` on the import path and pins the AWS region before
boto3 or the SageMaker SDK load (a default profile region such as
``me-central-1`` must never leak in). Also silences noisy SDK loggers, tees
each run's output to ``reports/logs/``, and turns expected stops (a missing
confirmation phrase, a budget refusal, expired credentials) into one-line
messages while keeping the full traceback in the log.
"""

# region Imports & setup
from __future__ import annotations

import argparse
import contextlib
import logging
import os
import sys
import traceback
from collections.abc import Callable, Iterator, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

_ROOT = Path(__file__).resolve().parents[1]
for _folder in (_ROOT / "agent", _ROOT):
    if str(_folder) not in sys.path:
        sys.path.insert(0, str(_folder))

from aws.config import ARTIFACTS_DIR, DATA_DIR, PROJECT_ROOT, REPORTS_DIR, DemoConfig, ResourceState  # noqa: E402
from aws.records.evidence import log_safe  # noqa: E402

CONFIG = DemoConfig()
os.environ["AWS_DEFAULT_REGION"] = os.environ["AWS_REGION"] = CONFIG.region
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
for _name in ("sagemaker", "botocore", "boto3", "urllib3", "httpx", "mlflow", "strands", "openai"):
    logging.getLogger(_name).setLevel(logging.WARNING)

__all__ = ["ARTIFACTS_DIR", "DATA_DIR", "PROJECT_ROOT"]  # re-exported: scripts use bootstrap.DATA_DIR etc.
TABLES_DIR = REPORTS_DIR / "tables"
FIGURES_DIR = REPORTS_DIR / "figures"
REPRO_DIR = REPORTS_DIR / "repro"
LOGS_DIR = REPORTS_DIR / "logs"
LOGS_KEPT = 5
"""Newest logs kept per script; older ones are pruned."""
EXPECTED_STOPS = (PermissionError, RuntimeError, ValueError, TimeoutError, FileNotFoundError)
# endregion


# region Output helpers
def now() -> str:
    """Returns the current UTC time for log lines."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def banner(title: str, detail: str) -> None:
    """Prints a script header."""
    line = "=" * 78
    print(f"{line}\n{title}\n{detail}\n{line}", flush=True)


def show(rows: Sequence[Mapping[str, Any]]) -> None:
    """Prints rows as an aligned table."""
    import pandas as pd

    print(pd.DataFrame(rows).to_string(index=False) if rows else "  (none)", flush=True)


def show_fields(record: Mapping[str, Any], keys: Sequence[str]) -> None:
    """Prints the non-empty fields of a record as aligned ``name  value`` lines."""
    width = max(len(key) for key in keys)
    for key in keys:
        if record.get(key) not in (None, ""):
            print(f"  {key:<{width}}  {record[key]}", flush=True)


def report(results: Sequence[tuple[str, bool, str]]) -> int:
    """Prints PASS/FAIL lines and a summary.

    Args:
        results: ``(label, passed, detail)`` per check.

    Returns:
        Number of failed checks.
    """
    for label, ok, detail in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}\n         {detail}", flush=True)
    failures = sum(1 for _, ok, _ in results if not ok)
    print(f"\n{len(results) - failures}/{len(results)} checks passed", flush=True)
    return failures
# endregion


# region Arguments, state, and running
def parse_args(
    description: str,
    confirmation: str | None = None,
    extra: Callable[[argparse.ArgumentParser], None] | None = None,
    shows_record: bool = True,
) -> argparse.Namespace:
    """Parses a script's arguments, adding ``--confirm`` for billable steps.

    Args:
        description: Script summary for ``--help``.
        confirmation: Exact phrase that authorizes the step, if it bills.
        extra: Adds script-specific arguments.
        shows_record: Whether the script shows recorded results without the phrase.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description=description)
    if confirmation:
        fallback = "recorded results are shown" if shows_record else "the script stops"
        parser.add_argument("--confirm", default="", metavar="PHRASE",
                            help=f"pass exactly {confirmation} to run this step; without it, {fallback}")
    if extra:
        extra(parser)
    return parser.parse_args()


def require_state() -> ResourceState:
    """Loads provisioned resource identifiers.

    Raises:
        FileNotFoundError: If ``artifacts/resources.json`` does not exist yet.
    """
    if not ResourceState.STATE_FILE.exists():
        raise FileNotFoundError("artifacts/resources.json is missing: run scripts/01_provision.py first.")
    return ResourceState.load()


def load_workflow() -> Any:
    """Builds the region-pinned workflow facade over the provisioned resources."""
    from aws.workflow import MtrlDemoWorkflow

    return MtrlDemoWorkflow(CONFIG, require_state())


def run(main: Callable[[], None]) -> None:
    """Runs a script's ``main`` with a log file and clear messages for expected stops.

    Args:
        main: The script's entry point.
    """
    with _tee(Path(sys.argv[0]).stem) as log:
        print(f"[{now()}] start", flush=True)
        try:
            main()
        except KeyboardInterrupt:
            print("\nInterrupted. Submitted AWS jobs keep running; rerun this script to re-attach.")
            sys.exit(130)
        except Exception as error:
            log.write(log_safe(traceback.format_exc()))
            if not _expected(error):
                raise
            print(f"\nSTOPPED: {type(error).__name__}: {error}", flush=True)
            sys.exit(1)
        print(f"[{now()}] done", flush=True)


def _expected(error: Exception) -> bool:
    """Tells deliberate stops and AWS client errors apart from bugs."""
    from botocore.exceptions import BotoCoreError, ClientError

    return isinstance(error, (*EXPECTED_STOPS, BotoCoreError, ClientError))


class _Tee:
    """Writes to the console as is, and to the log without signed links or the home path."""

    def __init__(self, stream: TextIO, log: TextIO) -> None:
        self.stream, self.log = stream, log

    def write(self, data: str) -> int:
        self.stream.write(data)
        self.log.write(log_safe(data))
        return len(data)

    def flush(self) -> None:
        self.stream.flush()
        self.log.flush()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.stream, name)


@contextlib.contextmanager
def _tee(script: str) -> Iterator[TextIO]:
    """Tees stdout and stderr to a timestamped log, pruning old logs.

    Args:
        script: Script name, used as the log file prefix.

    Yields:
        The open log file.
    """
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    for old in sorted(LOGS_DIR.glob(f"{script}_*.log"))[:-(LOGS_KEPT - 1)]:
        old.unlink()
    with (LOGS_DIR / f"{script}_{stamp}.log").open("w") as log:
        stdout, stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = _Tee(stdout, log), _Tee(stderr, log)
        try:
            yield log
        finally:
            sys.stdout, sys.stderr = stdout, stderr
# endregion

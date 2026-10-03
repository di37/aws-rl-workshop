"""Invariants over the provenance records: what ran is what the repository says.

Each check returns ``(passed, detail)``, reads only local files (the records
that ``scripts/11b_snapshot_provenance.py`` saved), and fails when a record is
missing rather than passing.
"""

from __future__ import annotations

import hashlib
import zipfile
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from aws.records.evidence import EvidenceStore
from aws.records.provenance import (
    AGENT_RECORD,
    AGENT_SOURCE,
    DATASET_RECORD,
    FIXED_IMPORT_FILES,
    IMPORT_RECORD,
    module_constant,
    zip_hashes,
)
from aws.reporting.repro_artifacts import pinned_requirements

Check = tuple[bool, str]
AGENT_FILES_THAT_RAN = ("environment.py", "rollout_driver.py", "Dockerfile")
"""Agent files that must be byte-identical to the deployed source bundle."""
JOBS_ON_THE_AGENT = ("Base evaluation", "Training", "Comparison evaluation")
_AGENT_PACKAGE = "sagemaker-mtrl-support-agent"
_MISSING = "run scripts/11b_snapshot_provenance.py"


def check_agent_provenance(root: Path, store: EvidenceStore) -> Check:
    """The agent that served every rollout is the repository's: code, prompt, and packages."""
    record, bundle = store.load(AGENT_RECORD) or {}, root / "artifacts" / AGENT_SOURCE
    if not record or not bundle.is_file():
        return False, f"agent provenance or source bundle missing: {_MISSING}"
    problems = []
    if hashlib.sha256(bundle.read_bytes()).hexdigest() != record["source_bundle"]["sha256"]:
        problems.append("source bundle hash differs from its record")
    if record["build"].get("later_builds") != 0:
        problems.append("a later build may have replaced the source bundle")
    deployed = zip_hashes(bundle)
    problems += [f"agent/{name} differs from what ran" for name in AGENT_FILES_THAT_RAN
                 if deployed.get(name) != _sha256(root / "agent" / name)]
    with zipfile.ZipFile(bundle) as archive:
        deployed_app = archive.read("app.py").decode()
    if module_constant(deployed_app, "SYSTEM_PROMPT") != module_constant(
            (root / "agent" / "app.py").read_text(), "SYSTEM_PROMPT"):
        problems.append("system prompt differs from what ran")
    updated = _time(record["runtime"]["last_updated"])
    jobs = {job["job"]: _time(job["started"]) for job in record.get("jobs", [])}
    stale = [name for name in JOBS_ON_THE_AGENT if name not in jobs or jobs[name] <= updated]
    if stale:
        problems.append(f"not shown to run on this runtime version: {stale}")
    image_packages = {k: v for k, v in record.get("packages", {}).items() if k != _AGENT_PACKAGE}
    if pinned_requirements(root / "agent" / "requirements.lock.txt") != image_packages:
        problems.append("agent/requirements.lock.txt differs from the image's packages")
    return not problems, (f"runtime version {record['runtime']['version']}, image "
                          f"{(record['image']['digest'] or '')[:19]}...; problems: {problems or 'none'}")


def check_dataset_provenance(store: EvidenceStore) -> Check:
    """The S3 datasets the jobs read equal the CSVs and were written before those jobs started."""
    record, agent = store.load(DATASET_RECORD) or {}, store.load(AGENT_RECORD) or {}
    if not record or not agent:
        return False, f"dataset or agent provenance missing: {_MISSING}"
    splits = {split["split"]: split for split in record.get("splits", [])}
    starts = {job["job"]: _time(job["started"]) for job in agent.get("jobs", [])}
    readers = {"training": ("Training",), "evaluation": ("Base evaluation", "Comparison evaluation")}
    problems = []
    for split, jobs in readers.items():
        info = splits.get(split)
        if not info or not info.get("prompts_match_csv"):
            problems.append(f"{split}: not recorded or differs from the CSV")
            continue
        late = [job for job in jobs if job not in starts or _time(info["last_modified"]) >= starts[job]]
        if late:
            problems.append(f"{split}: not written before {late}")
    return not problems, f"problems: {problems or 'none'}"


def check_deployment(store: EvidenceStore) -> Check:
    """The deployed fine-tuned model traces back to the trained package (Bedrock import or endpoint)."""
    package = (store.load("training_job.json") or {}).get("output_model_package_arn")
    if not package:
        return False, "no trained model package recorded"
    imported, import_detail = _import_traces_to(store, package)
    endpoint = store.load("endpoint.json") or {}
    served = endpoint.get("model_package_arn") == package and bool(
        endpoint.get("ready_at") or endpoint.get("in_service_epoch"))
    return imported or served, f"Bedrock import: {import_detail}; endpoint served this package: {served}"


def _import_traces_to(store: EvidenceStore, package: str) -> Check:
    """Checks the import read the package's merged weights, changing only the documented files."""
    record, imported = store.load(IMPORT_RECORD) or {}, store.load("bedrock_import.json") or {}
    if not record:
        return False, f"no import provenance ({_MISSING})"
    verdicts = {row["file"]: row["verdict"] for row in record.get("files", [])}
    weights = [name for name in verdicts if name.endswith(".safetensors")]
    mismatched = [name for name, verdict in verdicts.items() if verdict in ("DIFFERENT", "only in copy")]
    changed = {name for name, verdict in verdicts.items() if verdict == "changed on purpose"}
    job: Mapping[str, Any] = record.get("import_job", {})
    ok = (record.get("model_package_arn") == package and job.get("status") == "Completed"
          and job.get("source_uri") == imported.get("source_uri") and bool(weights)
          and all(verdicts[name] in ("identical", "same size") for name in weights)
          and not mismatched and changed <= set(FIXED_IMPORT_FILES))
    return ok, (f"{len(weights)} weight files equal in size, changed on purpose {sorted(changed)}, "
                f"mismatched {mismatched or 'none'}")


def _sha256(path: Path) -> str | None:
    """Returns a file's SHA-256, or None when it does not exist."""
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def _time(value: str) -> datetime:
    """Parses an ISO 8601 timestamp."""
    return datetime.fromisoformat(value)

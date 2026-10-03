"""Records what actually ran, read once from AWS, so the study can be audited offline.

Three evidence records, each built from read-only calls:

- agent: the AgentCore runtime version that served the rollouts, its container
  image and base-image digests, every package installed in the image (from the
  CodeBuild log), and the exact source bundle the image was built from;
- datasets: the S3 objects the jobs read, compared prompt by prompt with data/;
- import: the Bedrock import job's source files, compared file by file with the
  trained model package's merged weights.
"""

from __future__ import annotations

import ast
import hashlib
import io
import re
import zipfile
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from aws.config import DemoConfig, ResourceState
from aws.deploy.bedrock_import import IMPORT_REGION, merged_weights_uri
from aws.records.evidence import EvidenceStore

AGENT_RECORD = "agent_provenance.json"
AGENT_SOURCE = "agent_deployed_source.zip"
DATASET_RECORD = "dataset_provenance.json"
IMPORT_RECORD = "bedrock_import_provenance.json"
FIXED_IMPORT_FILES = ("tokenizer_config.json", "generation_config.json")
"""Files the import deliberately changes in its copy: chat template and end-of-sequence IDs."""
DATASETS = (("training", "training_prompts"), ("evaluation", "evaluation_prompts"))
_BASE_IMAGE = re.compile(r"FROM docker\.io/library/(\S+@sha256:[0-9a-f]{64})")
_INSTALLED = "Successfully installed "
_RECENT_BUILDS = 20


def parse_build_log(text: str, image_tag: str) -> dict[str, Any]:
    """Reads the base image, the pushed image digest, and installed packages from a build log.

    Args:
        text: Full CodeBuild log of one image build.
        image_tag: Tag of the image the runtime serves.

    Returns:
        ``base_image``, ``image_digest`` (None if this log did not push the tag),
        and ``packages`` (lowercase name to version).
    """
    base = _BASE_IMAGE.search(text)
    pushed = re.search(rf"{re.escape(image_tag)}: digest: (sha256:[0-9a-f]{{64}})", text)
    installed = next((line for line in text.splitlines() if _INSTALLED in line), "")
    items = installed.split(_INSTALLED, 1)[-1].split() if installed else []
    packages = dict(sorted((name.lower(), version) for name, version in (i.rsplit("-", 1) for i in items)))
    return {"base_image": base.group(1) if base else None,
            "image_digest": pushed.group(1) if pushed else None,
            "packages": packages}


def zip_hashes(path: Path) -> dict[str, str]:
    """Returns the SHA-256 of every file inside a zip archive, by name."""
    with zipfile.ZipFile(path) as archive:
        return {name: hashlib.sha256(archive.read(name)).hexdigest()
                for name in sorted(archive.namelist()) if not name.endswith("/")}


def module_constant(source: str, name: str) -> str | None:
    """Returns a module-level string constant from Python source, without running it.

    Args:
        source: Python source text.
        name: Constant name, such as ``SYSTEM_PROMPT``.

    Returns:
        The string value, or None when it is not a module-level string assignment.
    """
    for node in ast.parse(source).body:
        targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(
            node, ast.AnnAssign) else []
        value = getattr(node, "value", None)
        if any(isinstance(t, ast.Name) and t.id == name for t in targets) and isinstance(value, ast.Constant) \
                and isinstance(value.value, str):
            return value.value
    return None


def compare_copies(
    source: Mapping[str, tuple[int, str]], copy: Mapping[str, tuple[int, str]], fixed: Iterable[str]
) -> list[dict[str, Any]]:
    """Gives each file of a copied model folder a verdict against its source.

    Args:
        source: File name to (size, ETag) in the source folder.
        copy: File name to (size, ETag) in the copy.
        fixed: Files the copy changes on purpose.

    Returns:
        One row per file: sizes and a verdict (``identical``, ``same size``,
        ``changed on purpose``, ``not copied``, ``only in copy``, or ``DIFFERENT``).
    """
    rows = []
    for name in sorted(set(source) | set(copy)):
        original, copied = source.get(name), copy.get(name)
        if copied is None:
            verdict = "not copied"
        elif original is None:
            verdict = "only in copy"
        elif name in set(fixed):
            verdict = "changed on purpose"
        elif original == copied:
            verdict = "identical"
        elif original[0] == copied[0]:
            verdict = "same size"
        else:
            verdict = "DIFFERENT"
        rows.append({"file": name, "source_bytes": original[0] if original else None,
                     "copy_bytes": copied[0] if copied else None, "verdict": verdict})
    return rows


def capture_agent(session: Any, config: DemoConfig, state: ResourceState, store: EvidenceStore) -> dict[str, Any]:
    """Records the runtime that served the rollouts, its image, packages, and source bundle.

    Args:
        session: Boto3 session in the demo region.
        config: Shared settings (agent name).
        state: Provisioned identifiers (runtime ID).
        store: Evidence store; the source bundle is saved next to the record.

    Returns:
        The saved record.
    """
    runtime = session.client("bedrock-agentcore-control").get_agent_runtime(agentRuntimeId=state.agent_runtime_id)
    image_uri = runtime["agentRuntimeArtifact"]["containerConfiguration"]["containerUri"]
    build, later_builds, parsed = _build_of(session, f"bedrock-agentcore-{config.agent_name}-builder",
                                            image_uri.rsplit(":", 1)[-1])
    bucket, key = build["source"]["location"].split("/", 1)
    s3 = session.client("s3")
    source_modified = s3.head_object(Bucket=bucket, Key=key)["LastModified"]
    bundle = store.directory / AGENT_SOURCE
    s3.download_file(bucket, key, str(bundle))
    record = {
        "captured_at": _iso(datetime.now(timezone.utc)),
        "runtime": {"arn": runtime["agentRuntimeArn"], "version": runtime["agentRuntimeVersion"],
                    "status": runtime["status"], "last_updated": _iso(runtime["lastUpdatedAt"])},
        "image": {"uri": image_uri, "digest": parsed["image_digest"], "base_image": parsed["base_image"]},
        "build": {"id": build["id"], "status": build["buildStatus"], "started": _iso(build["startTime"]),
                  "ended": _iso(build["endTime"]), "later_builds": later_builds,
                  "source": f"s3://{bucket}/{key}", "source_last_modified": _iso(source_modified)},
        "source_bundle": {"file": AGENT_SOURCE, "sha256": hashlib.sha256(bundle.read_bytes()).hexdigest(),
                          "files": zip_hashes(bundle)},
        "packages": parsed["packages"],
        "jobs": _jobs_of_record(session, store),
    }
    store.save(AGENT_RECORD, record)
    return record


def capture_datasets(session: Any, state: ResourceState, data_dir: Path, store: EvidenceStore) -> dict[str, Any]:
    """Records the S3 datasets the jobs read and whether they equal the local CSVs.

    Args:
        session: Boto3 session in the demo region.
        state: Provisioned identifiers (bucket).
        data_dir: Folder with the prompt CSVs.
        store: Evidence store.

    Returns:
        The saved record.
    """
    s3, splits = session.client("s3"), []
    for split, stem in DATASETS:
        key = f"datasets/{stem}.parquet"
        head = s3.head_object(Bucket=state.bucket, Key=key)
        remote = pd.read_parquet(io.BytesIO(s3.get_object(Bucket=state.bucket, Key=key)["Body"].read()))
        local = pd.read_csv(data_dir / f"{stem}.csv")
        splits.append({"split": split, "s3_uri": f"s3://{state.bucket}/{key}",
                       "last_modified": _iso(head["LastModified"]), "etag": head["ETag"].strip('"'),
                       "rows": len(remote), "prompts_match_csv": remote["prompt"].tolist() == local["prompt"].tolist()})
    record = {"captured_at": _iso(datetime.now(timezone.utc)), "splits": splits}
    store.save(DATASET_RECORD, record)
    return record


def capture_import(session: Any, store: EvidenceStore) -> dict[str, Any] | None:
    """Compares the files the Bedrock import read with the trained package's merged weights.

    Args:
        session: Boto3 session in the demo region.
        store: Evidence store with the import and training records.

    Returns:
        The saved record, or None when no import or training is recorded.
    """
    imported, training = store.load("bedrock_import.json"), store.load("training_job.json") or {}
    package = training.get("output_model_package_arn")
    if not imported or not package:
        return None
    job = session.client("bedrock", region_name=IMPORT_REGION).get_model_import_job(jobIdentifier=imported["job_arn"])
    weights = merged_weights_uri(session.client("sagemaker"), package)
    copy_uri = job["modelDataSource"]["s3DataSource"]["s3Uri"]
    record = {
        "captured_at": _iso(datetime.now(timezone.utc)),
        "import_job": {"arn": job["jobArn"], "status": job["status"], "source_uri": copy_uri,
                       "imported_model_arn": job.get("importedModelArn"), "ended": _iso(job.get("endTime"))},
        "model_package_arn": package,
        "merged_weights_uri": weights,
        "files": compare_copies(_listing(session, weights), _listing(session, copy_uri, IMPORT_REGION),
                                FIXED_IMPORT_FILES),
    }
    store.save(IMPORT_RECORD, record)
    return record


def _build_of(session: Any, project: str, tag: str) -> tuple[dict[str, Any], int, dict[str, Any]]:
    """Finds the recent build whose log pushed the image tag; counts builds after it."""
    codebuild = session.client("codebuild")
    ids = codebuild.list_builds_for_project(projectName=project, sortOrder="DESCENDING")["ids"][:_RECENT_BUILDS]
    builds = sorted(codebuild.batch_get_builds(ids=ids)["builds"], key=lambda b: b["startTime"], reverse=True)
    for later, build in enumerate(builds):
        parsed = parse_build_log(_log_text(session, build), tag)
        if parsed["image_digest"]:
            return build, later, parsed
    raise RuntimeError(f"No recent {project} build log pushed image tag {tag}.")


def _log_text(session: Any, build: Mapping[str, Any]) -> str:
    """Reads a CodeBuild log from CloudWatch Logs, page by page."""
    client, logs = session.client("logs"), build["logs"]
    messages: list[str] = []
    token = None
    while True:
        kwargs = {"logGroupName": logs["groupName"], "logStreamName": logs["streamName"], "startFromHead": True}
        page = client.get_log_events(**kwargs, **({"nextToken": token} if token else {}))
        messages.extend(event["message"] for event in page["events"])
        if not page["events"] or page["nextForwardToken"] == token:
            return "".join(messages)
        token = page["nextForwardToken"]


def _jobs_of_record(session: Any, store: EvidenceStore) -> list[dict[str, Any]]:
    """Lists when each recorded job that called the runtime started."""
    sagemaker, jobs = session.client("sagemaker"), []
    for label, name in (("Base evaluation", "base_evaluation.json"), ("Comparison evaluation",
                                                                      "comparison_evaluation.json")):
        record = store.load(name)
        if record:
            started = sagemaker.describe_pipeline_execution(PipelineExecutionArn=record["arn"])["CreationTime"]
            jobs.append({"job": label, "id": record["arn"].rsplit("/", 1)[-1], "started": _iso(started)})
    training = store.load("training_job.json") or {}
    if training.get("creation_time"):
        jobs.append({"job": "Training", "id": training["job_name"], "started": _iso(training["creation_time"])})
    return jobs


def _listing(session: Any, uri: str, region: str | None = None) -> dict[str, tuple[int, str]]:
    """Lists an S3 folder as file name to (size, ETag)."""
    bucket, prefix = uri.removeprefix("s3://").split("/", 1)
    client = session.client("s3", region_name=region) if region else session.client("s3")
    pages = client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix)
    return {item["Key"][len(prefix):].lstrip("/"): (item["Size"], item["ETag"])
            for page in pages for item in page.get("Contents", [])}


def _iso(value: Any) -> str | None:
    """Formats a datetime, or an ISO string, as UTC ISO 8601; keeps None."""
    if value in (None, ""):
        return None
    moment = datetime.fromisoformat(value) if isinstance(value, str) else value
    return moment.astimezone(timezone.utc).isoformat(timespec="seconds")

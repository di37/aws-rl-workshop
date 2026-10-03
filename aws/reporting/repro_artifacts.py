"""Builds the reproducibility record in ``reports/repro/`` (CS7641 REPRO layout).

Five files let a reviewer reproduce and audit the study: package versions
against the pins, the ordered run commands, ``study_metadata.json`` (datasets,
hyperparameters, seeds, every AWS job ID of the run of record, and results),
the cost accounting, and an inventory of evidence files.
"""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import re
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

from aws.config import MODEL_ID, DemoConfig
from aws.deploy.bedrock_import import IMPORT_REGION
from aws.records.evidence import EvidenceStore
from aws.reporting.report_tables import headline_results, run_of_record

STUDY = "Multi-turn RL (SageMaker MTRL) of a customer-support troubleshooting agent"
GUIDE = "https://docs.aws.amazon.com/sagemaker/latest/dg/model-customize-mtrl.html"
DATASETS = ("training_prompts.csv", "evaluation_prompts.csv")
STUDY_NOTEBOOKS = "notebooks/sagemaker_mtrl_real_demo*.ipynb"
RUN_COMMANDS: tuple[tuple[str, str, str], ...] = (
    ("00", "python scripts/00_verify_setup.py", "offline: exact pins and datasets (--aws adds live checks)"),
    ("01", "python scripts/01_provision.py --confirm CONFIRM_PROVISION_MTRL_DEMO", "one-time setup"),
    ("02", "python scripts/02_request_quotas.py --submit", "free; approval can take hours"),
    ("03", "python scripts/03_deploy_agent.py --confirm CONFIRM_DEPLOY_AGENT_MTRL_DEMO",
     "CodeBuild minutes"),
    ("04", "python scripts/04_upload_datasets.py", "skips unchanged files; then 00 --aws checks the setup"),
    ("05", "python scripts/05_base_evaluation.py --confirm CONFIRM_BASE_EVAL_MTRL_DEMO", "billable"),
    ("06", "python scripts/06_train.py --confirm CONFIRM_TRAIN_MTRL_DEMO", "billable"),
    ("07", "python scripts/07_compare_evaluation.py --confirm CONFIRM_COMPARISON_EVAL_MTRL_DEMO",
     "billable"),
    ("08", "python scripts/08_recorded_trajectories.py", "read-only"),
    ("09a", "python scripts/09a_deploy_endpoint.py --confirm CONFIRM_DEPLOY_MTRL_DEMO",
     "optional option 1, about $13 per hour; start scripts/09a_watchdog.py first"),
    ("09b", "python scripts/09b_bedrock_import.py --confirm CONFIRM_BEDROCK_IMPORT_MTRL_DEMO",
     "option 2; copy, then storage and per-minute use"),
    ("10", "python scripts/10_live_inference.py --confirm CONFIRM_LIVE_INFERENCE_MTRL_DEMO", "billable"),
    ("11", "python scripts/11_snapshot_costs.py", "read-only; freezes billed usage and prices"),
    ("11b", "python scripts/11b_snapshot_provenance.py",
     "read-only; records the agent image and source, datasets, and imported weights"),
    ("12", "python scripts/12_make_report_tables_and_figures.py", "offline"),
    ("13", "python scripts/13_build_repro_artifacts.py", "offline"),
    ("14", "python scripts/14_verify_invariants.py", "offline; exits non-zero on failure"),
    ("15", "python scripts/15_build_repro_sheet.py", "offline; writes REPRO_SageMaker_MTRL.pdf (needs tectonic)"),
    ("99", "python scripts/99_teardown.py --confirm DELETE_MTRL_DEMO",
     "deletes the tracked demo resources (see README for leftovers); artifacts/ is kept"),
)
"""Ordered run of record: (step, command, cost or note)."""

_PIN = re.compile(r"^\s*([A-Za-z0-9_.-]+)(?:\[[^\]]*\])?\s*==\s*([^\s;#]+)")


def requirements_file(root: Path) -> Path:
    """Returns the pinned requirements file for this operating system.

    Args:
        root: Project root.

    Returns:
        ``requirements-macos.txt`` on macOS, else ``requirements-linux.txt``.
    """
    name = "requirements-macos.txt" if platform.system() == "Darwin" else "requirements-linux.txt"
    return root / name


def exact_requirements_file(root: Path) -> Path:
    """Returns the exact pins for this operating system: the full lock when one exists.

    Args:
        root: Project root.

    Returns:
        ``requirements-macos.lock.txt`` on macOS when present, else :func:`requirements_file`.
    """
    lock = root / "requirements-macos.lock.txt"
    return lock if platform.system() == "Darwin" and lock.is_file() else requirements_file(root)


def pinned_requirements(path: Path) -> dict[str, str]:
    """Parses ``name==version`` pins, ignoring extras, comments, and includes.

    Args:
        path: Requirements file.

    Returns:
        Package name to pinned version.
    """
    pins: dict[str, str] = {}
    for line in path.read_text().splitlines():
        match = _PIN.match(line)
        if match:
            pins[match.group(1)] = match.group(2)
    return pins


def environment_versions(pins: Mapping[str, str]) -> list[dict[str, Any]]:
    """Compares installed package versions with their pins.

    Args:
        pins: Package name to pinned version.

    Returns:
        Rows of package, installed version, pin, and whether they match;
        the first row is the Python interpreter (pinned to 3.12).
    """
    python = platform.python_version()
    rows = [{"package": "python", "version": python, "pinned": "3.12",
             "matches_pin": python.startswith("3.12.")}]
    for name, pinned in pins.items():
        try:
            installed = metadata.version(name)
        except metadata.PackageNotFoundError:
            installed = "not installed"
        rows.append({"package": name, "version": installed, "pinned": pinned,
                     "matches_pin": installed == pinned})
    return rows


def dataset_fingerprint(path: Path) -> dict[str, Any]:
    """Fingerprints a prompt CSV by content.

    Args:
        path: CSV with one ``prompt`` column.

    Returns:
        Row count, unique row count, and SHA-256 of the file bytes.
    """
    with path.open(newline="") as handle:
        prompts = [row["prompt"] for row in csv.DictReader(handle)]
    return {
        "file": path.name,
        "rows": len(prompts),
        "unique_rows": len(set(prompts)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def study_metadata(
    root: Path,
    store: EvidenceStore,
    config: DemoConfig,
    snapshot: Mapping[str, Any],
    task: Mapping[str, Any],
) -> dict[str, Any]:
    """Collects everything needed to reproduce and audit the study in one mapping.

    Args:
        root: Project root.
        store: Evidence store of the run of record.
        config: Shared demo settings.
        snapshot: Cost snapshot (``cost_accounting.json``).
        task: Environment facts, such as the success path and turn limit.

    Returns:
        Study, platform, task, datasets, model, settings, seeds, run of
        record, headline results, and cost totals.
    """
    training = store.load("training_job.json") or {}
    job_config = training.get("training_config") or {}
    return {
        "study": STUDY,
        "official_guide": GUIDE,
        "platform": {"system": platform.system(), "release": platform.release(),
                     "machine": platform.machine(), "python": platform.python_version()},
        "aws": {"region": config.region, "bedrock_import_region": IMPORT_REGION},
        "task": {**task, "reward": "0.25 per correct step; 1.0 when the connection is restored"},
        "datasets": [dataset_fingerprint(root / "data" / name) for name in DATASETS],
        "model": {"sagemaker_hub_model_id": MODEL_ID, "bedrock_base_model_id": config.bedrock_base_model_id,
                  "imported_model_name": config.imported_model_name},
        "training": {
            "job_name": training.get("job_name"),
            "base_model_hub_content": job_config.get("BaseModelArn"),
            "job_hyperparameters": job_config.get("HyperParameters"),
            "approved_size": {"steps": config.training_steps, "prompts_per_step": config.training_batch_size,
                              "rollouts_per_prompt": config.training_group_size},
            "other_hyperparameters": "service defaults for the hub content version above",
        },
        "evaluation": {"rollouts_per_prompt": config.eval_group_size,
                       "sampling_max_tokens": config.sampling_max_tokens,
                       "pass_k_values": list(config.pass_k_values)},
        "seeds": {
            "live_inference_seed": config.inference_seed,
            "live_inference_tickets": f"first {config.inference_ticket_count} held-out prompts, file order",
            "rollout_action_order": "random per episode in the AgentCore runtime; recorded per rollout",
            "service_side": "MTRL sampling and LoRA updates expose no seed; reruns vary within sampling noise",
        },
        "run_of_record": run_of_record(store),
        "provenance": _provenance_summary(store),
        "results": headline_results(store),
        "cost_usd": {key: snapshot.get(key) for key in (
            "billed_tokens_usd", "computed_or_estimated_usd", "total_usd", "total_with_allowance_usd",
            "budget_cap_usd")},
    }


def _provenance_summary(store: EvidenceStore) -> dict[str, Any]:
    """Summarizes what ran, from the provenance records (None where not recorded)."""
    agent = store.load("agent_provenance.json") or {}
    imported = store.load("bedrock_import_provenance.json") or {}
    return {
        "agent_runtime_version": agent.get("runtime", {}).get("version"),
        "agent_image_digest": agent.get("image", {}).get("digest"),
        "agent_base_image": agent.get("image", {}).get("base_image"),
        "agent_source_sha256": agent.get("source_bundle", {}).get("sha256"),
        "imported_weights": {row["file"]: row["verdict"] for row in imported.get("files", [])} or None,
    }


def artifact_inventory(root: Path) -> list[dict[str, str]]:
    """Lists the evidence, tables, figures, and notebooks of the study.

    Args:
        root: Project root.

    Returns:
        Rows of kind and project-relative file path.
    """
    groups = (
        ("evidence", "artifacts/*.json"),
        ("evidence", "artifacts/*.zip"),
        ("superseded", "artifacts/superseded/*"),
        ("executed-notebook", "artifacts/*.ipynb"),
        ("table", "reports/tables/*.csv"),
        ("figure", "reports/figures/*.png"),
        ("notebook", STUDY_NOTEBOOKS),
        ("data", "data/*.csv"),
    )
    return [
        {"kind": kind, "file": str(path.relative_to(root))}
        for kind, pattern in groups
        for path in sorted(root.glob(pattern))
        if not path.name.startswith(".")
    ]


def write_repro_record(
    repro_dir: Path,
    environment: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any],
    cost_lines: Sequence[Mapping[str, Any]],
    inventory: Sequence[Mapping[str, str]],
) -> list[Path]:
    """Writes the five reproducibility files.

    Args:
        repro_dir: Destination folder, created if missing.
        environment: Rows from :func:`environment_versions`.
        metadata: Study metadata for ``study_metadata.json``.
        cost_lines: Rows of item, USD, and source.
        inventory: Rows from :func:`artifact_inventory`.

    Returns:
        Paths of the written files.
    """
    repro_dir.mkdir(parents=True, exist_ok=True)
    written = [
        _write_csv(repro_dir / "environment_versions.csv",
                   ["package", "version", "pinned", "matches_pin"], environment),
        _write_csv(repro_dir / "run_commands.csv", ["step", "command", "note"],
                   [{"step": s, "command": c, "note": n} for s, c, n in RUN_COMMANDS]),
        _write_csv(repro_dir / "compute_accounting.csv", ["item", "usd", "source"], cost_lines),
        _write_csv(repro_dir / "artifact_inventory.csv", ["kind", "file"], inventory),
    ]
    path = repro_dir / "study_metadata.json"
    stamped = {"generated": datetime.now(timezone.utc).isoformat(timespec="seconds"), **metadata}
    path.write_text(json.dumps(stamped, indent=2, default=str) + "\n")
    return [*written, path]


def _write_csv(path: Path, columns: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> Path:
    """Writes rows to a CSV with a fixed column order.

    Args:
        path: Destination file.
        columns: Column names, in order.
        rows: Row mappings; extra keys are ignored.

    Returns:
        The written path.
    """
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path

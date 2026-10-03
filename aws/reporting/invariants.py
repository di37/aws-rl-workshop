"""Reproducibility invariants: PASS/FAIL checks over data, evidence, provenance, and outputs.

Each check returns ``(passed, detail)``, reads only local files, and fails when
something it needs is missing instead of passing on nothing. The protocol is
pinned to the run of record here, not read from the editable configuration
(CS7641 ``verify_invariants``).
"""

from __future__ import annotations

import ast
import csv
import json
import re
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd

from aws.config import DemoConfig
from aws.costs.cost_guard import MtrlCostGuard
from aws.records.evidence import EvidenceStore
from aws.reporting.provenance_checks import (
    check_agent_provenance,
    check_dataset_provenance,
    check_deployment,
)
from aws.reporting.report_tables import FIGURE_FILES, SOLVED, build_tables
from aws.reporting.repro_artifacts import (
    STUDY_NOTEBOOKS,
    dataset_fingerprint,
    environment_versions,
    exact_requirements_file,
    pinned_requirements,
)

Check = tuple[bool, str]
_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]|\x1b\]8;[^\x1b]*\x1b\\")


@dataclass(frozen=True)
class Protocol:
    """The documented protocol of the run of record."""

    training_steps: int = 10
    group_size: int = 4
    batch_size: int = 32
    sampling_max_tokens: int = 512
    base_model: str = "openai-reasoning-gpt-oss-20b/3.43.0"
    eval_prompts: int = 32
    eval_rollouts: int = 2
    live_tickets: int = 6
    live_seed: int = 2026

    @property
    def hyperparameters(self) -> dict[str, str]:
        """Job hyperparameters as the training job records them (strings)."""
        return {"max_steps": str(self.training_steps), "group_size": str(self.group_size),
                "global_batch_size": str(self.batch_size), "sampling_max_tokens": str(self.sampling_max_tokens)}


RUN_OF_RECORD = Protocol()
RUN_OF_RECORD_DATASETS = {
    "training_prompts.csv": "1c61077a792df2c7ce19a00f38abe8b96a2bf146abe031c41fd8ab7c6c807277",
    "evaluation_prompts.csv": "213538946018bdc334ed3b764ccaf16bee8f9bc9bef514adc5899608221e60cf",
}
"""SHA-256 of the prompt files of the run of record (also checked against S3 by provenance)."""
REQUIRED_COST_LINES = ("Base evaluation", "Training", "Comparison: base model", "Comparison: fine-tuned model")


def _prompts(path: Path) -> list[str]:
    """Reads prompts from a CSV with one ``prompt`` column."""
    with path.open(newline="") as handle:
        return [row["prompt"].strip() for row in csv.DictReader(handle)]


def check_config_protocol(config: DemoConfig, protocol: Protocol = RUN_OF_RECORD) -> Check:
    """aws/config.py still configures the documented protocol."""
    actual = {"training_steps": config.training_steps, "group_size": config.training_group_size,
              "batch_size": config.training_batch_size, "sampling_max_tokens": config.sampling_max_tokens,
              "eval_rollouts": config.eval_group_size, "live_tickets": config.inference_ticket_count,
              "live_seed": config.inference_seed}
    differs = {name: value for name, value in actual.items() if getattr(protocol, name) != value}
    return not differs, f"settings that differ from the protocol: {differs or 'none'}"


def check_datasets(data_dir: Path, train_size: int = 64, eval_size: int = 32) -> Check:
    """Training and held-out prompts have the expected sizes and do not overlap."""
    train = _prompts(data_dir / "training_prompts.csv")
    held_out = _prompts(data_dir / "evaluation_prompts.csv")
    overlap = set(train) & set(held_out)
    ok = (len(set(train)) == len(train) == train_size and len(set(held_out)) == len(held_out) == eval_size
          and not overlap)
    return ok, f"{len(train)} training / {len(held_out)} held-out unique prompts; {len(overlap)} overlap"


def check_dataset_fingerprints(data_dir: Path, expected: dict[str, str]) -> Check:
    """Prompt files are byte-identical to those of the run of record."""
    changed = [name for name, sha in expected.items() if dataset_fingerprint(data_dir / name)["sha256"] != sha]
    return not changed, f"{len(expected)} files checked; changed since the run of record: {changed or 'none'}"


def check_training(store: EvidenceStore, protocol: Protocol = RUN_OF_RECORD) -> Check:
    """The training job completed every step with the job hyperparameters of the protocol."""
    record = store.load("training_job.json") or {}
    progress = record.get("progress_info") or {}
    recorded = (record.get("training_config") or {}).get("HyperParameters") or {}
    differs = {key: recorded.get(key) for key, value in protocol.hyperparameters.items() if recorded.get(key) != value}
    ok = (record.get("job_status") == "Completed" and not differs
          and progress.get("CurrentStep") == progress.get("MaxSteps") == protocol.training_steps
          and progress.get("BatchSize") == protocol.batch_size and bool(record.get("output_model_package_arn")))
    return ok, (f"{record.get('job_name')}: {record.get('job_status')}, step {progress.get('CurrentStep')}/"
                f"{progress.get('MaxSteps')}, batch {progress.get('BatchSize')}; job hyperparameters that "
                f"differ from the protocol: {differs or 'none'}")


def check_base_model(store: EvidenceStore, protocol: Protocol = RUN_OF_RECORD) -> Check:
    """Training started from the base-model version of the run of record."""
    arn = ((store.load("training_job.json") or {}).get("training_config") or {}).get("BaseModelArn") or ""
    version = arn.rsplit("/Model/", 1)[-1] or "unknown"
    return arn.endswith("/" + protocol.base_model), f"training started from {version} (run of record: {protocol.base_model})"


def check_training_metrics(store: EvidenceStore, protocol: Protocol = RUN_OF_RECORD) -> Check:
    """Per-step metrics of this training job cover every step, each with batch x group trajectories."""
    record = store.load("training_metrics.json") or {}
    job = (store.load("training_job.json") or {}).get("job_name")
    steps = record.get("steps", [])
    numbers = sorted(int(step["step"]) for step in steps)
    expected = protocol.batch_size * protocol.group_size
    short = [step["step"] for step in steps if step.get("training/num_trajectories") != expected]
    ok = bool(job) and record.get("job_name") == job and not short and numbers == list(
        range(1, protocol.training_steps + 1))
    return ok, (f"{len(numbers)} of {protocol.training_steps} steps for job {record.get('job_name')}; "
                f"steps without {expected} trajectories: {short or 'none'}")


def check_learning_signal(store: EvidenceStore) -> Check:
    """The baseline rewards varied, so training had a signal to learn from."""
    metrics = store.load("base_evaluation_metrics.json") or {}
    low, high = metrics.get("eval/reward/min"), metrics.get("eval/reward/max")
    ok = isinstance(low, (int, float)) and isinstance(high, (int, float)) and low < high
    return ok, f"baseline reward range [{low}, {high}]"


def check_comparison(store: EvidenceStore, protocol: Protocol = RUN_OF_RECORD) -> Check:
    """The comparison pipeline succeeded with the protocol's prompts and rollouts for each model."""
    record = store.load("comparison_evaluation.json") or {}
    metrics = store.load("comparison_evaluation_metrics.json") or {}
    shapes = {key: (metrics.get(key, {}).get("metrics", {}).get("eval/reward/num_prompts"),
                    metrics.get(key, {}).get("metrics", {}).get("eval/reward/rollouts_per_prompt"))
              for key in ("base", "fine_tuned")}
    expected = (protocol.eval_prompts, protocol.eval_rollouts)
    ok = record.get("status") == "Succeeded" and all(shape == expected for shape in shapes.values())
    return ok, f"pipeline {record.get('status')}; prompts x rollouts per model {shapes}"


def check_trajectories_match_metrics(store: EvidenceStore, protocol: Protocol = RUN_OF_RECORD) -> Check:
    """Every recorded conversation is present, and together they recompute the reported metrics."""
    metrics = store.load("comparison_evaluation_metrics.json") or {}
    pairs = (store.load("evaluation_trajectories.json") or {}).get("pairs", [])
    details, ok = [], len(pairs) == protocol.eval_prompts
    for side, key in (("before", "base"), ("after", "fine_tuned")):
        rewards = [conversation.get("reward") for pair in pairs for conversation in pair[side]]
        reported = metrics.get(key, {}).get("metrics", {})
        complete = len(rewards) == protocol.eval_prompts * protocol.eval_rollouts and all(
            isinstance(reward, (int, float)) for reward in rewards)
        solved = sum(1 for reward in rewards if reward == SOLVED)
        recomputed = {"eval/reward/succeeded_rollouts": solved,
                      "eval/reward/pass_at_1": solved / len(rewards) if complete else None,
                      "eval/reward/mean": sum(rewards) / len(rewards) if complete else None}
        ok = ok and complete and all(_close(value, reported.get(name)) for name, value in recomputed.items())
        details.append(f"{key}: {len(rewards)} traces, {solved} solved, mean {recomputed['eval/reward/mean']} "
                       f"(reported {reported.get('eval/reward/succeeded_rollouts')}, {reported.get('eval/reward/mean')})")
    return ok, "; ".join(details)


def check_inference_protocol(store: EvidenceStore, data_dir: Path, protocol: Protocol = RUN_OF_RECORD) -> Check:
    """Live inference replayed the first held-out tickets in order, with the documented seed, without errors."""
    results = store.load("inference_before_after.json") or {}
    entries = results.get("tickets", [])
    tickets = [entry["ticket"].strip() for entry in entries]
    expected = _prompts(data_dir / "evaluation_prompts.csv")[: protocol.live_tickets]
    failed = [f"{index}:{side}" for index, entry in enumerate(entries) for side in ("before", "after")
              if (entry.get(side) or {}).get("error")
              or not isinstance((entry.get(side) or {}).get("reward"), (int, float))]
    in_order = bool(tickets) and tickets == expected
    ok = in_order and results.get("seed") == protocol.live_seed and not failed
    return ok, (f"{len(tickets)} tickets; first {protocol.live_tickets} held-out in order: {in_order}; seed "
                f"{results.get('seed')} (documented {protocol.live_seed}); failed calls: {failed or 'none'}")


def check_no_live_endpoint(store: EvidenceStore) -> Check:
    """No endpoint is recorded as still running."""
    status = (store.load("endpoint.json") or {}).get("status", "never deployed")
    return status in ("Deleted", "never deployed"), f"endpoint record: {status}"


def check_costs(root: Path, store: EvidenceStore, cap_usd: float) -> Check:
    """Every cost line is present and recomputable, the report matches it, and the total is within the cap."""
    snapshot = store.load("cost_accounting.json")
    if not snapshot:
        return False, "artifacts/cost_accounting.json missing: run scripts/11_snapshot_costs.py"
    rates = {name: Decimal(rate) for name, rate in snapshot["rates_usd_per_million_tokens"].items()}
    guard = MtrlCostGuard(DemoConfig(), rates, Decimal(snapshot.get("hosting_usd_per_hour") or "0"))
    items = snapshot["items"]
    mispriced = [item["item"] for item in items if "usage" in item
                 and round(float(guard.billed_cost(item["component"], item["usage"])), 4) != item["usd"]]
    amounts = {item["item"]: item["usd"] for item in items}
    extras = (store.load("extra_costs.json") or {}).get("items", [])
    missing = [extra["label"] for extra in extras if amounts.get(extra["label"]) != round(float(extra["usd"]), 4)]
    missing += [label for label in REQUIRED_COST_LINES if label not in amounts]
    published = _csv_rows(root / "reports" / "repro" / "compute_accounting.csv")
    stale = published != [{"item": item["item"], "usd": str(item["usd"]), "source": item["source"]} for item in items]
    total = round(sum(amounts.values()), 2)
    ok = not mispriced and not missing and not stale and total <= cap_usd
    return ok, (f"total ${total:.2f} (with allowance) of the ${cap_usd:g} cap; mispriced {mispriced or 'none'}; "
                f"missing {missing or 'none'}; compute_accounting.csv current: {not stale}")


def check_no_signed_links(paths: Sequence[Path]) -> Check:
    """No evidence file, notebook, report, or log contains a presigned URL or token."""
    leaks = [path.name for path in paths if EvidenceStore.redact(path.read_text()) != path.read_text()]
    return bool(paths) and not leaks, f"{len(paths)} files scanned; with signed links: {leaks or 'none'}"


def check_environment(pins: Path) -> Check:
    """Installed packages match the exact pins of the run of record."""
    rows = environment_versions(pinned_requirements(pins))
    mismatched = [f"{row['package']} {row['version']} (pinned {row['pinned']})" for row in rows
                  if not row["matches_pin"]]
    return len(rows) > 1 and not mismatched, f"{pins.name}: {len(rows)} checked; mismatched {mismatched[:6] or 'none'}"


def check_notebooks(paths: Sequence[Path]) -> Check:
    """Every code cell parses, and executed outputs contain no exceptions."""
    problems = []
    for path in paths:
        for index, cell in enumerate(json.loads(path.read_text()).get("cells", [])):
            if cell.get("cell_type") != "code":
                continue
            try:
                ast.parse("".join(cell.get("source", [])))
            except SyntaxError:
                problems.append(f"{path.name} cell {index}: syntax")
            for output in cell.get("outputs", []):
                text = _ANSI.sub("", "\n".join(_strings(output)))
                if output.get("output_type") == "error" or (
                        "Traceback" in text and "most recent call last" in text):
                    problems.append(f"{path.name} cell {index}: exception")
    return bool(paths) and not problems, f"{len(paths)} notebooks; problems: {problems or 'none'}"


def check_report_outputs(
    root: Path, tables: Mapping[str, list[dict[str, Any]]], figures: Iterable[str] = FIGURE_FILES
) -> Check:
    """Every report table equals a fresh rebuild from the evidence; every figure exists and is not empty."""
    folder = root / "reports"
    stale = [name for name, rows in tables.items() if not (folder / "tables" / name).is_file()
             or (folder / "tables" / name).read_text() != pd.DataFrame(rows).to_csv(index=False)]
    missing = [name for name in figures if not (folder / "figures" / name).is_file()
               or (folder / "figures" / name).stat().st_size == 0]
    ok = bool(tables) and not stale and not missing
    return ok, f"{len(tables)} tables; stale or missing: {stale or 'none'}; figures missing or empty: {missing or 'none'}"


def run_all(root: Path) -> list[tuple[str, bool, str]]:
    """Runs every invariant; a check that cannot run is reported as FAIL.

    Args:
        root: Project root.

    Returns:
        ``(label, passed, detail)`` per invariant.
    """
    store, data = EvidenceStore(root / "artifacts"), root / "data"
    notebooks = sorted(root.glob(STUDY_NOTEBOOKS))
    scanned = [*sorted((root / "artifacts").glob("*.json")), *sorted((root / "artifacts").glob("*.ipynb")),
               *notebooks, *sorted((root / "reports").glob("**/*.csv")), *sorted((root / "reports").glob("**/*.json")),
               *sorted((root / "reports" / "logs").glob("*.log"))]
    checks: list[tuple[str, Callable[[], Check]]] = [
        ("settings in aws/config.py match the documented protocol", lambda: check_config_protocol(DemoConfig())),
        ("datasets: sizes, uniqueness, held-out disjoint", lambda: check_datasets(data)),
        ("datasets unchanged since the run of record",
         lambda: check_dataset_fingerprints(data, RUN_OF_RECORD_DATASETS)),
        ("S3 datasets equal the CSVs and predate the jobs", lambda: check_dataset_provenance(store)),
        ("agent that ran matches the repository (code, prompt, packages)",
         lambda: check_agent_provenance(root, store)),
        ("training job ran the documented protocol", lambda: check_training(store)),
        ("base model version matches the run of record", lambda: check_base_model(store)),
        ("training metrics recorded for every step", lambda: check_training_metrics(store)),
        ("baseline had a learning signal", lambda: check_learning_signal(store)),
        ("comparison evaluation succeeded, 32 x 2 per model", lambda: check_comparison(store)),
        ("recorded traces recompute the evaluation metrics", lambda: check_trajectories_match_metrics(store)),
        ("deployed model traces back to the trained package", lambda: check_deployment(store)),
        ("live inference: first held-out tickets, documented seed, no errors",
         lambda: check_inference_protocol(store, data)),
        ("no endpoint left running", lambda: check_no_live_endpoint(store)),
        ("cost accounting complete, recomputable, within the cap",
         lambda: check_costs(root, store, DemoConfig().budget_limit_usd)),
        ("no signed links in evidence, notebooks, reports, or logs", lambda: check_no_signed_links(scanned)),
        ("environment matches the exact pins", lambda: check_environment(exact_requirements_file(root))),
        ("notebooks parse and ran without exceptions", lambda: check_notebooks(notebooks)),
        ("report tables equal a rebuild from evidence; figures present",
         lambda: check_report_outputs(root, build_tables(store))),
    ]
    results = []
    for label, check in checks:
        try:
            ok, detail = check()
        except Exception as error:  # noqa: BLE001 - a check that cannot run is a FAIL
            ok, detail = False, f"could not run: {type(error).__name__}: {error}"
        results.append((label, ok, detail))
    return results


def _close(value: Any, reported: Any) -> bool:
    """Compares a recomputed metric with the reported one."""
    return isinstance(value, (int, float)) and isinstance(reported, (int, float)) and abs(value - reported) < 1e-9


def _csv_rows(path: Path) -> list[dict[str, str]] | None:
    """Reads a CSV as rows of strings; None when the file does not exist."""
    if not path.is_file():
        return None
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _strings(value: Any) -> Iterator[str]:
    """Yields every string inside a notebook output (texts, lists, and mappings)."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)

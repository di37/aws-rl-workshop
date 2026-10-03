"""Builds the reproducibility sheet (LaTeX) from evidence: sections 1 to 9.

``build_sheet`` fills ``templates/repro_sheet.tex``. Every number comes from
``artifacts/`` or ``reports/`` at build time, the metric definitions are
recomputed from the recorded conversations, and the build refuses to run while
any invariant fails.
"""

from __future__ import annotations

import math
import platform
import subprocess
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from aws.config import MODEL_ID, DemoConfig
from aws.deploy.bedrock_import import IMPORT_REGION
from aws.records.evidence import EvidenceStore
from aws.records.provenance import AGENT_RECORD, DATASET_RECORD
from aws.reporting.invariants import RUN_OF_RECORD
from aws.reporting.latex import Raw, breakable, latex, render, table
from aws.reporting.repro_artifacts import DATASETS, dataset_fingerprint, pinned_requirements, requirements_file
from aws.reporting.repro_sheet_materials import materials_values, require, tt, utc, verify_trace_metrics

TEMPLATE = Path(__file__).with_name("templates") / "repro_sheet.tex"
SHEET_NAME = "REPRO_SageMaker_MTRL"
RUN_OF_RECORD_PYTHON = "3.12.10"
RUN_OF_RECORD_HOST = f"macOS 26 (Darwin 25.3), Apple silicon (arm64), Python {RUN_OF_RECORD_PYTHON}"
SDK_DEFAULTS = ("learning rate 1e-5, LoRA rank/alpha 32/64, PPO loss: service defaults for this model version, "
                "as reported by the SDK (notebook section 6); the job does not echo them back")
GUIDE = "https://docs.aws.amazon.com/sagemaker/latest/dg/model-customize-mtrl.html"


def build_sheet(root: Path, invariants: Sequence[tuple[str, bool, str]], task: Mapping[str, Any]) -> str:
    """Renders the complete LaTeX sheet.

    Args:
        root: Project root.
        invariants: ``(label, passed, detail)`` from :func:`invariants.run_all`.
        task: Agent facts: ``success_path``, ``max_turns``, ``max_progress``, ``max_policy_calls``.

    Returns:
        The LaTeX document.

    Raises:
        RuntimeError: If any invariant fails, or a reported metric does not match its recomputation.
    """
    failed = [label for label, ok, _ in invariants if not ok]
    if failed:
        raise RuntimeError(f"Fix the failing invariants before building the sheet: {failed}")
    store, config = EvidenceStore(root / "artifacts"), DemoConfig()
    values = {**_identity_values(root, store), **_hardware_values(store), **_environment_values(root),
              **_data_values(root, store), **_seed_values(store, config), **_cost_values(store, config),
              **_protocol_values(store), **_config_values(store, config, task), **_metric_values(store),
              **materials_values(root, store, invariants)}
    return render(TEMPLATE.read_text(), values)


def _identity_values(root: Path, store: EvidenceStore) -> dict[str, str]:
    """Section 1: commit, account, regions, job identifiers, and the run dates."""
    training, agent = require(store, "training_job.json"), require(store, AGENT_RECORD)
    imported = store.load("bedrock_import.json") or {}
    arn = training["job_arn"].split(":")
    start = min(job["started"] for job in agent["jobs"])
    commit = _git_commit(root)
    rows = [
        ("Repository commit", Raw(tt(commit)) if commit else "not yet under version control"),
        ("AWS account and regions", f"{arn[4]}; {arn[3]} (SageMaker AI, AgentCore, MLflow, S3), "
                                    f"{IMPORT_REGION} (Bedrock import)"),
        ("Training job", Raw(tt(training["job_name"]))),
        ("Model package", Raw(tt(training["output_model_package_arn"].split("/", 1)[1]))),
        ("Imported model", Raw(breakable(imported.get("imported_model_arn", "none")))),
        ("Official guide", Raw(rf"\url{{{GUIDE}}}")),
    ]
    note = "" if commit else ("The commit row fills in automatically once the project is under version control; "
                              "rebuild this sheet after the first commit.")
    return {"RUN_DATES": latex(f"{utc(start)} to {utc(imported.get('ended_at') or start)}"),
            "IDENTIFIERS_TABLE": table(["Item", "Value"], rows, "P{3.6cm}P{11.8cm}"),
            "IDENTIFIERS_NOTE": latex(note)}


def _git_commit(root: Path) -> str | None:
    """Returns the commit of the repository rooted exactly at ``root``, or None."""
    try:
        result = subprocess.run(["git", "-C", str(root), "rev-parse", "--show-toplevel", "HEAD"],
                                capture_output=True, text=True, check=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    top, _, sha = result.stdout.strip().partition("\n")
    return sha.strip() if Path(top).resolve() == root.resolve() and sha.strip() else None


def _hardware_values(store: EvidenceStore) -> dict[str, str]:
    """Section 2: where each part of the study ran."""
    agent, endpoint = require(store, AGENT_RECORD), store.load("endpoint.json") or {}
    base_model = (require(store, "training_job.json").get("training_config") or {}).get("BaseModelArn", "")
    capacity = "InsufficientInstanceCapacity" in (endpoint.get("failure_reason") or "")
    rows = [
        ("Run of record (orchestration, live inference, reports)", RUN_OF_RECORD_HOST),
        ("This build of the sheet", f"{platform.system()} {platform.release()} ({platform.machine()}), "
                                    f"Python {platform.python_version()}"),
        ("Training and evaluation", f"Amazon SageMaker AI multi-turn RL, serverless (instances are managed and "
                                    f"not exposed); base model {base_model.rsplit('/Model/', 1)[-1]}"),
        ("Agent environment", f"Amazon Bedrock AgentCore Runtime version {agent['runtime']['version']}, "
                              f"linux/arm64 container on {agent['image']['base_image'].split('@')[0]} "
                              "(digests in Section 12.6)"),
        ("Deployment option 1", f"SageMaker real-time endpoint on {endpoint.get('instance_type', 'none')}: "
                                + ("not provisioned (InsufficientInstanceCapacity); nothing billed" if capacity
                                   else str(endpoint.get("status", "not attempted")))),
        ("Deployment option 2", f"Amazon Bedrock Custom Model Import, {IMPORT_REGION}; serverless, billed per "
                                "active model minute"),
        ("Linux", "requirements-linux.txt carries the same direct pins; the pipeline has not been run on Linux"),
    ]
    return {"HARDWARE_TABLE": table(["Role", "Where it ran"], rows, "P{4.6cm}P{10.8cm}")}


def _environment_values(root: Path) -> dict[str, str]:
    """Section 3: lock sizes and the direct dependency table."""
    entries = [("python", RUN_OF_RECORD_PYTHON), *sorted(pinned_requirements(requirements_file(root)).items())]
    half = math.ceil(len(entries) / 2)
    right = entries[half:] + [(Raw(""), Raw(""))] * (2 * half - len(entries))
    rows = [(*left, *other) for left, other in zip(entries[:half], right)]
    return {
        "LOCK_COUNT": str(len(pinned_requirements(root / "requirements-macos.lock.txt"))),
        "AGENT_PACKAGES": str(len(pinned_requirements(root / "agent" / "requirements.lock.txt"))),
        "PACKAGE_TABLE": table(["Package", "Version", "Package", "Version"], rows, "llll",
                               caption=r"Direct dependencies (\texttt{requirements-macos.txt})."),
    }


def _data_values(root: Path, store: EvidenceStore) -> dict[str, str]:
    """Section 4: dataset fingerprints and the S3 provenance of what the jobs read."""
    prints = [dataset_fingerprint(root / "data" / name) for name in DATASETS]
    rows = [(Raw(tt(fp["file"])), fp["rows"], fp["unique_rows"], Raw(rf"\texttt{{\scriptsize {fp['sha256']}}}"))
            for fp in prints]
    starts = {job["job"]: job["started"] for job in require(store, AGENT_RECORD)["jobs"]}
    readers = {"training": ("Training",), "evaluation": ("Base evaluation", "Comparison evaluation")}
    provenance = [(split["split"], Raw(tt(split["s3_uri"].rsplit("/", 1)[1])), utc(split["last_modified"], True),
                   split["prompts_match_csv"], "; ".join(f"{job} {utc(starts[job], True)}"
                                                        for job in readers[split["split"]]))
                  for split in require(store, DATASET_RECORD)["splits"]]
    return {
        "BATCH_SIZE": str(RUN_OF_RECORD.batch_size),
        "DATASET_TABLE": "{\\small\n" + table(["File (in data/)", "Rows", "Unique", "SHA-256"], rows, "lrrl") + "\n}",
        "DATA_PROVENANCE_TABLE": "{\\small\n" + table(
            ["Split", "S3 object (datasets/)", "Written", "Equals CSV", "Read by (start)"], provenance,
            "lP{4.8cm}P{2.3cm}cP{4.1cm}") + "\n}",
    }


def _seed_values(store: EvidenceStore, config: DemoConfig) -> dict[str, str]:
    """Section 5: seeds, unseedable randomness, and the size of sampling noise."""
    rows = [
        ("Live inference: action order", str(config.inference_seed),
         "aws/config.py (inference_seed); recorded in inference_before_after.json"),
        ("Live inference: tickets", f"first {config.inference_ticket_count} held-out, in file order",
         "aws/config.py (inference_ticket_count)"),
        ("Rollout action order (training and evaluation)", "random per episode",
         "agent/app.py (secrets.randbits); recorded per rollout as action_order_seed"),
        ("SageMaker MTRL sampling and LoRA updates", "not seedable", "service-side"),
        ("Bedrock sampling (live inference)", "not seedable", "service-side"),
    ]
    comparison = require(store, "comparison_evaluation_metrics.json")
    shape = comparison["base"]["metrics"]
    rollouts = int(shape["eval/reward/num_prompts"] * shape["eval/reward/rollouts_per_prompt"])
    errors = {key: math.sqrt(p * (1 - p) / rollouts) for key in ("fine_tuned", "base")
              for p in [comparison[key]["metrics"]["eval/reward/pass_at_1"]]}
    return {
        "SEEDS_TABLE": table(["Purpose", "Value", "Where set"], rows, "P{4.6cm}P{3.6cm}P{6.8cm}"),
        "EVAL_ROLLOUTS": str(rollouts),
        "PASS1_SE": latex(f"{errors['fine_tuned']:.3f} (fine-tuned) to {errors['base']:.3f} (base)"),
        "BASELINE_PASS1": f"{require(store, 'base_evaluation_metrics.json')['eval/reward/pass_at_1']:.3f}",
        "COMPARISON_BASE_PASS1": f"{comparison['base']['metrics']['eval/reward/pass_at_1']:.3f}",
    }


def _cost_values(store: EvidenceStore, config: DemoConfig) -> dict[str, str]:
    """Section 6: cost lines with their inputs, totals, and job durations."""
    snapshot = require(store, "cost_accounting.json")
    rates = snapshot["rates_usd_per_million_tokens"]
    groups = {prefix: ", ".join(f"{name.split('_', 1)[1]} {float(rate):.2f}" for name, rate in rates.items()
                                if name.startswith(prefix)) for prefix in ("training", "evaluation")}
    short = {"billed tokens": "billed tokens x rate", "computed": "computed or estimated", "allowance": "allowance"}
    rows = [(item["item"], _tokens(item.get("usage")), f"{item['usd']:.4f}",
             next(label for key, label in short.items() if item["source"].startswith(key))) for item in snapshot["items"]]
    totals = (f"Billed tokens \\${snapshot['billed_tokens_usd']:.2f} plus computed or estimated "
              f"\\${snapshot['computed_or_estimated_usd']:.2f} gives \\${snapshot['total_usd']:.2f}; with the "
              f"allowance for unmeasured AgentCore, CodeBuild, S3, and logs, \\${snapshot['total_with_allowance_usd']:.2f}"
              f", against the \\${config.budget_limit_usd} hard cap.")
    return {
        "RATES": latex(f"training {groups['training']}; evaluation {groups['evaluation']}"),
        "COST_TABLE": table(["Item", "Billed tokens", "USD", "Source"], rows, "P{5.6cm}P{4.4cm}rP{2.6cm}",
                            caption="Compute accounting.", long=True),
        "COST_TOTALS": totals,
        "DURATION_TABLE": _duration_table(store),
        "BUDGET_CAP": str(config.budget_limit_usd),
        "HOSTING_RATE": f"{float(snapshot['hosting_usd_per_hour']):.2f}",
    }


def _tokens(usage: Mapping[str, int] | None) -> str:
    """Formats billed token counts compactly; ``--`` when the line is not token-priced."""
    if not usage:
        return "--"
    names = {"PrefillTokenCount": "prefill", "SampleTokenCount": "sample", "TrainTokenCount": "train"}
    return ", ".join(f"{names[key]} {value:,}" for key, value in usage.items() if key in names)


def _duration_table(store: EvidenceStore) -> str:
    """Job start times and durations, marking what was not captured."""
    starts = {job["job"]: job["started"] for job in require(store, AGENT_RECORD)["jobs"]}
    training, imported = require(store, "training_job.json"), store.load("bedrock_import.json") or {}
    endpoint = store.load("endpoint.json") or {}
    minutes = (_minutes(imported.get("started_at"), imported.get("ended_at"))
               if imported.get("ended_at") else None)
    rows = [
        ("Base evaluation (of record)", utc(starts["Base evaluation"], True), "n/c"),
        ("Training (10 steps)", utc(starts["Training"], True), f"{training['duration_minutes']} min"),
        ("Comparison evaluation", utc(starts["Comparison evaluation"], True), "n/c"),
        ("Endpoint attempt (option 1)", utc(endpoint["created_at"], True) if endpoint else "--",
         f"{endpoint.get('uptime_minutes')} min until deleted; never InService" if endpoint else "--"),
        ("Bedrock import job (option 2)", utc(imported["started_at"], True) if imported else "--",
         f"{minutes} min" if minutes is not None else "--"),
    ]
    note = (r"{\small n/c = not captured: the pipelines' end times were not recorded. They can be read from "
            r"SageMaker in the run's account.}\par")
    return table(["Job", "Started", "Duration"], rows, "P{5.2cm}P{4.2cm}P{5.6cm}") + "\n" + note


def _minutes(start: str, end: str) -> float:
    """Minutes between two ISO timestamps, to one decimal."""
    return round((datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds() / 60, 1)


def _protocol_values(store: EvidenceStore) -> dict[str, str]:
    """Section 7: protocol choices and every deviation, stated with their numbers."""
    first = require(store, "base_evaluation_metrics.pre_fix.json")
    rejected = next((store.load(name) for name in store.names("training_job.*.json")), {})
    endpoint = store.load("endpoint.json") or {}
    live = require(store, "inference_before_after.json")["tickets"]
    calls = {side: sorted({entry[side]["metrics"]["policy_calls"] for entry in live}) for side in ("before", "after")}
    texted = sorted({entry["after"]["metrics"]["text_fallback_actions"] for entry in live})
    baseline = require(store, "base_evaluation_metrics.json")["eval/reward/pass_at_1"]
    compared = require(store, "comparison_evaluation_metrics.json")["base"]["metrics"]["eval/reward/pass_at_1"]
    rows = [
        ("Held-out evaluation", "One pipeline evaluates both models on the same held-out tickets, through the same "
                                "agent and serving stack, with 2 tries per ticket."),
        ("Headline metric", "pass@1 (ticket fully solved). Partial credit (0.25 per correct step) shapes the reward "
                            "for training but is not the headline."),
        ("Agent harness", "The agent re-prompts the model when it stops mid-task and reads actions written as text "
                          "(agent/rollout_driver.py), identically before and after training."),
        ("First baseline", f"Scored pass@1 {first['eval/reward/pass_at_1']} because rollouts ended after "
                           f"{first['eval/turns/mean']:g} model calls, before the agent fix. It is kept and costed."),
        ("Rejected training job", f"{rejected.get('job_name', 'none')}: {(rejected.get('failure_reason') or 'none')[:150]} "
                                  "Nothing was billed."),
        ("Deployment option 1", "The endpoint was never provisioned (InsufficientInstanceCapacity), so it never "
                                f"reached InService and cost {endpoint.get('billed_usd', 0)} USD."),
        ("Live demo", f"An illustration, not the measurement. The base model called the tool natively "
                      f"({'/'.join(map(str, calls['before']))} model call per ticket). The imported model returned "
                      f"each action as JSON text that the agent parsed ({'/'.join(map(str, texted))} text actions, "
                      f"{'/'.join(map(str, calls['after']))} model calls per ticket)."),
        ("Run-to-run noise", f"The same base model scored pass@1 {baseline:.3f} in the baseline and {compared:.3f} in "
                             "the comparison, on the same tickets."),
        ("Retrofits", "bedrock_import.json gained model_package_arn after the import (copied from training_job.json); "
                      "the file comparison in Section 12 verifies it independently. training_job.json's training and "
                      "agent config were read back from AWS on 2026-10-02. endpoint.json was corrected from an "
                      "assumed 8.39 USD to 0."),
    ]
    return {"PROTOCOL_TABLE": table(["Topic", "Disclosure"], rows, "P{3.6cm}P{11.8cm}")}


def _config_values(store: EvidenceStore, config: DemoConfig, task: Mapping[str, Any]) -> dict[str, str]:
    """Section 8: the study configuration, from the job record, the config, and the agent."""
    job_config = require(store, "training_job.json").get("training_config") or {}
    hyper = ", ".join(f"{key} {value}" for key, value in sorted((job_config.get("HyperParameters") or {}).items()))
    per_step = RUN_OF_RECORD.batch_size * RUN_OF_RECORD.group_size
    rows = [
        ("Task", f"{task['max_progress']}-step path {' > '.join(task['success_path'])}; {task['max_turns']} turns; "
                 f"reward = correct steps / {task['max_progress']} (1.0 when solved)"),
        ("Agent", f"Strands agent with one tool on Bedrock AgentCore; the driver allows up to "
                  f"{task['max_policy_calls']} model calls per episode"),
        ("Base model", f"{MODEL_ID} ({job_config.get('BaseModelArn', '').rsplit('/Model/', 1)[-1]}); live "
                       f"baseline on Bedrock as {config.bedrock_base_model_id}"),
        ("Training (job-recorded)", f"{hyper}; {per_step} rollouts per step, "
                                    f"{per_step * RUN_OF_RECORD.training_steps:,} in total"),
        ("Training (service defaults)", SDK_DEFAULTS),
        ("Evaluation", f"{config.eval_group_size} rollouts per ticket; pass@k for k in "
                       f"{', '.join(map(str, config.pass_k_values))}; sampling_max_tokens {config.sampling_max_tokens}"),
        ("Deployment option 1", f"ModelBuilder(ModelPackage) on {config.endpoint_instance_type}; lifetime limit "
                                f"{config.endpoint_max_minutes} min, enforced by the watchdog"),
        ("Deployment option 2", f"Bedrock Custom Model Import in {IMPORT_REGION}; copy of the merged weights with "
                                "tokenizer_config.json (chat template) and generation_config.json (end-of-sequence "
                                "IDs) fixed"),
        ("Live inference", f"first {config.inference_ticket_count} held-out tickets, seed {config.inference_seed}, "
                           "the same agent episode (run_episode) for both models"),
        ("Budget", f"hard cap {config.budget_limit_usd} USD; 1.5x token margin; 2 USD allowance for unmeasured items"),
    ]
    return {"CONFIG_TABLE": table(["Element", "Setting"], rows, "P{3.6cm}P{11.8cm}",
                                  caption="Study configuration.")}


def _metric_values(store: EvidenceStore) -> dict[str, str]:
    """Section 9: exact metric definitions, each recomputed from the traces where possible."""
    checks = verify_trace_metrics(store)
    rows = [
        ("pass@1", "share of rollouts with reward 1.0 (the success threshold): solved / (tickets x tries)",
         checks["eval/reward/pass_at_1"]),
        ("pass@2", "share of tickets with at least one of their 2 rollouts at reward 1.0", checks["eval/reward/pass_at_2"]),
        ("Mean reward", "mean rollout reward; reward = correct steps x 0.25, 1.0 when solved", checks["eval/reward/mean"]),
        ("Reward std", "population standard deviation (ddof 0) of rollout rewards", checks["eval/reward/std"]),
        ("Rollouts fully solved", "count of rollouts with reward 1.0", checks["eval/reward/succeeded_rollouts"]),
        ("Model calls per rollout", "policy-model calls per rollout (eval/turns/mean)", "as reported"),
        ("Per-ticket change", "mean of a ticket's 2 rewards, after minus before: improved, same, or worse",
         "recomputed (report tables)"),
        ("Live fully solved", "a ticket whose live episode reached reward 1.0", "from the recorded run"),
        ("Cost line", "billed tokens / 1,000,000 x the rate for that token type", "recomputed (invariants)"),
    ]
    return {"METRICS_TABLE": table(["Metric", "Exact definition", "Check"], rows, "P{3.2cm}P{8.6cm}P{3.4cm}")}

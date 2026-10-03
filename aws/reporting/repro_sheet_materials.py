"""Reproducibility sheet, sections 10 to 14: commands, results, extended materials, provenance, invariants.

Also holds the small shared helpers (evidence loading, formatting, and the
trace-based metric check) used by ``repro_sheet``.
"""

from __future__ import annotations

import hashlib
import re
import statistics
import zipfile
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from aws.config import DemoConfig
from aws.records.evidence import EvidenceStore
from aws.records.provenance import AGENT_RECORD, AGENT_SOURCE, IMPORT_RECORD, module_constant, zip_hashes
from aws.reporting.latex import Raw, breakable, latex, table, verbatim
from aws.reporting.report_tables import PRODUCED_BY, SOLVED, build_tables, headline_results
from aws.reporting.repro_artifacts import RUN_COMMANDS, pinned_requirements

OFFLINE_STEPS = ("00", "12", "13", "14", "15")
REATTACH_STEPS = ("05", "06", "07", "08", "11", "11b")
INVARIANT_REFERENCES = {
    "INV_DISJOINT": "held-out disjoint", "INV_FINGERPRINTS": "datasets unchanged", "INV_S3": "S3 datasets",
    "INV_AGENT": "agent that ran", "INV_TRACES": "recorded traces", "INV_DEPLOY": "deployed model traces back",
    "INV_LIVE": "live inference",
}
"""Placeholder to a label fragment of the invariant it refers to; numbers follow run_all's order."""
FIGURE_CAPTIONS = {
    "training_reward_curve.png": "Mean rollout reward per training step (32 tickets x 4 rollouts per step).",
    "evaluation_before_after.png": "Held-out tickets before vs after training (32 tickets x 2 rollouts per model).",
    "per_ticket_before_after.png": "Mean reward per held-out ticket before (x) and after (y) training; point size "
                                   "counts tickets with the same pair.",
    "live_inference_before_after.png": "Live inference on held-out tickets with the same agent and seed.",
}
_METRICS = ("eval/reward/succeeded_rollouts", "eval/reward/pass_at_1", "eval/reward/pass_at_2",
            "eval/reward/mean", "eval/reward/std")


def require(store: EvidenceStore, name: str) -> Any:
    """Loads one evidence file, naming the step that produces it when missing."""
    record = store.load(name)
    if record is None:
        producer = PRODUCED_BY.get(name, "11b_snapshot_provenance.py")
        raise FileNotFoundError(f"artifacts/{name} is missing: run scripts/{producer} first.")
    return record


def tt(text: str) -> str:
    """Typewriter text, escaped."""
    return rf"\texttt{{{latex(text)}}}"


def utc(value: str, seconds: bool = False) -> str:
    """Formats an ISO timestamp as ``YYYY-MM-DD HH:MM[:SS] UTC``."""
    moment = datetime.fromisoformat(value).astimezone(timezone.utc)
    return moment.strftime("%Y-%m-%d %H:%M:%S UTC" if seconds else "%Y-%m-%d %H:%M UTC")


def verify_trace_metrics(store: EvidenceStore) -> dict[str, str]:
    """Recomputes the reported evaluation metrics from the recorded conversations.

    Returns:
        Metric name to ``"recomputed"`` for each metric that matched for both models.

    Raises:
        RuntimeError: If any recomputed value differs from the reported one.
    """
    pairs = require(store, "evaluation_trajectories.json")["pairs"]
    reported = require(store, "comparison_evaluation_metrics.json")
    for side, key in (("before", "base"), ("after", "fine_tuned")):
        groups = [[conversation["reward"] for conversation in pair[side]] for pair in pairs]
        rewards = [reward for group in groups for reward in group]
        solved = sum(1 for reward in rewards if reward == SOLVED)
        recomputed = {
            "eval/reward/succeeded_rollouts": solved,
            "eval/reward/pass_at_1": solved / len(rewards),
            "eval/reward/pass_at_2": sum(1 for group in groups if SOLVED in group) / len(groups),
            "eval/reward/mean": statistics.fmean(rewards),
            "eval/reward/std": statistics.pstdev(rewards),
        }
        for name, value in recomputed.items():
            if abs(value - reported[key]["metrics"][name]) > 1e-9:
                raise RuntimeError(f"{key} {name}: traces give {value}, the evaluator reported "
                                   f"{reported[key]['metrics'][name]}.")
    return {name: "recomputed" for name in _METRICS}


def materials_values(root: Path, store: EvidenceStore, invariants: Sequence[tuple[str, bool, str]]) -> dict[str, str]:
    """Builds every placeholder of sections 10 to 14 and the shared references."""
    config, tables = DemoConfig(), build_tables(store)
    shape = require(store, "comparison_evaluation_metrics.json")["base"]["metrics"]
    return {
        **_command_values(), **_headline_values(store), **_extended_values(tables),
        **_provenance_values(root, store), **_invariant_values(invariants),
        "EVAL_PROMPTS": str(int(shape["eval/reward/num_prompts"])),
        "LIVE_TICKETS": str(config.inference_ticket_count),
        "TEST_COUNT": str(_test_count(root)),
    }


def _command_values() -> dict[str, str]:
    """Section 10: command blocks derived from RUN_COMMANDS, plus the notebooks."""
    commands = {step: command for step, command, _ in RUN_COMMANDS}
    reattach = ["python scripts/00_verify_setup.py --aws"] + [
        re.sub(r" --confirm \S+", "", commands[step]) for step in REATTACH_STEPS]
    notebooks = [
        (Raw(tt("sagemaker_mtrl_real_demo.ipynb")), "demo notebook, sections 1 to 14; replays the recorded run by "
                                                    "default and submits nothing"),
        (Raw(tt("sagemaker_mtrl_real_demo.executed.ipynb")), "the replay with all outputs: present from this copy"),
        (Raw(tt("sagemaker_mtrl_real_demo.training-run.ipynb")), "log of the live training run, kept as recorded "
                                                                 "(read it; its code predates the folder layout)"),
    ]
    return {
        "COMMANDS_OFFLINE": verbatim([commands[step] for step in OFFLINE_STEPS]),
        "COMMANDS_REATTACH": verbatim(reattach),
        "COMMANDS_FULL": verbatim([command for _, command, _ in RUN_COMMANDS]),
        "NOTEBOOKS_TABLE": "{\\small\n" + table(["Notebook (in notebooks/)", "Purpose"], notebooks,
                                                 "P{7.6cm}P{7.8cm}") + "\n}",
    }


def _headline_values(store: EvidenceStore) -> dict[str, str]:
    """Section 11: the headline numbers, read from the evidence."""
    head = headline_results(store)
    comparison = require(store, "comparison_evaluation_metrics.json")
    base, tuned = comparison["base"]["metrics"], comparison["fine_tuned"]["metrics"]
    training, cost = require(store, "training_job.json"), require(store, "cost_accounting.json")
    first, last = head["training_mean_reward"]["first_step"], head["training_mean_reward"]["last_step"]
    changes, live = head["recorded_ticket_changes"], head["live_inference_solved"]
    rows = [
        ("Baseline", f"pass@1 {head['baseline_pass_at_1']:.3f}; the first attempt, before the agent fix, scored 0"),
        ("Training", f"{training['progress_info']['CurrentStep']}/{training['progress_info']['MaxSteps']} steps in "
                     f"{training['duration_minutes']} min; mean training reward {first:.3f} → {last:.3f}"),
        ("Held-out evaluation", f"pass@1 {base['eval/reward/pass_at_1']:.3f} → {tuned['eval/reward/pass_at_1']:.3f}; "
                                f"pass@2 {base['eval/reward/pass_at_2']:.3f} → {tuned['eval/reward/pass_at_2']:.3f}; "
                                f"mean reward {base['eval/reward/mean']:.3f} → {tuned['eval/reward/mean']:.3f}; "
                                f"fully solved {base['eval/reward/succeeded_rollouts']:g} → "
                                f"{tuned['eval/reward/succeeded_rollouts']:g} of {_rollouts(base)}"),
        ("Per held-out ticket", f"improved {changes['improved']}, same {changes['same']}, worse {changes['worse']}"),
        ("Deployment", "endpoint: no GPU capacity, nothing billed; Bedrock import: completed"),
        ("Live inference", f"fully solved: base {live['before']}/{live['tickets']} → fine-tuned "
                           f"{live['after']}/{live['tickets']}"),
        ("Cost", f"{cost['total_usd']:.2f} USD ({cost['billed_tokens_usd']:.2f} billed tokens) of the "
                 f"{cost['budget_cap_usd']} USD cap"),
    ]
    return {"HEADLINE_TABLE": table(["Part", "Headline numbers"], rows, "P{3.6cm}P{11.8cm}")}


def _rollouts(metrics: Mapping[str, float]) -> int:
    """Rollouts per model in an evaluation."""
    return int(metrics["eval/reward/num_prompts"] * metrics["eval/reward/rollouts_per_prompt"])


def _figure(name: str) -> str:
    """Includes one report figure with its caption."""
    return (f"\\begin{{center}}\n\\includegraphics[width=0.82\\linewidth]{{../figures/{name}}}\n"
            f"\\captionof{{figure}}{{{latex(FIGURE_CAPTIONS[name])}}}\n\\end{{center}}")


def _extended_values(tables: Mapping[str, list[dict[str, Any]]]) -> dict[str, str]:
    """Section 12: figures and the full report tables."""
    def rows(name: str, keys: Sequence[str]) -> list[list[Any]]:
        return [[row[key] for key in keys] for row in tables[name]]

    small = ("{\\small\n", "\n}")
    return {
        "FIG_TRAINING": _figure("training_reward_curve.png"),
        "TABLE_TRAINING": table(["Step", "Mean reward", "Mean turns", "Tokens", "Trajectories"],
                                rows("training_steps.csv", ("step", "mean_reward", "mean_turns", "total_tokens",
                                                            "trajectories")), "rrrrr",
                                caption=r"Training metrics per step (\texttt{training\_steps.csv})."),
        "FIG_EVALUATION": _figure("evaluation_before_after.png"),
        "TABLE_EVALUATION": table(["Metric", "Base", "Fine-tuned", "Change"],
                                  rows("evaluation_comparison.csv", ("label", "base", "fine_tuned", "delta")),
                                  "P{6.4cm}rrr", caption=r"Held-out evaluation (\texttt{evaluation\_comparison.csv})."),
        "TABLE_BASELINES": table(["Metric", "First attempt", "Baseline of record"],
                                 rows("baseline_evaluations.csv", ("label", "first_attempt", "baseline")),
                                 "P{6.4cm}rr", caption=r"The two baselines (\texttt{baseline\_evaluations.csv})."),
        "FIG_TICKETS": _figure("per_ticket_before_after.png"),
        "TABLE_TICKETS": small[0] + table(
            ["Held-out ticket", "Before", "After", "Mean before", "Mean after", "Change"],
            rows("per_ticket_rewards.csv", ("ticket", "before_rewards", "after_rewards", "before_mean", "after_mean",
                                            "change")),
            "P{5.6cm}llrrl", caption=r"Every held-out ticket (\texttt{per\_ticket\_rewards.csv}).", long=True) + small[1],
        "FIG_LIVE": _figure("live_inference_before_after.png"),
        "TABLE_LIVE": small[0] + table(
            ["Ticket", "Before", "After", "Before: actions", "After: actions"],
            rows("live_inference.csv", ("ticket", "before_reward", "after_reward", "before_actions", "after_actions")),
            "P{3.3cm}rrP{4.8cm}P{4.8cm}", caption=r"Live inference (\texttt{live\_inference.csv}).") + small[1],
        "TABLE_RUN_OF_RECORD": small[0] + table(
            ["Stage", "Identifier", "Status", "Detail"],
            rows("run_of_record.csv", ("stage", "identifier", "status", "detail")),
            "P{3.2cm}P{3.6cm}P{1.8cm}P{6.6cm}", caption=r"Run of record (\texttt{run\_of\_record.csv}).",
            long=True) + small[1],
    }


def _provenance_values(root: Path, store: EvidenceStore) -> dict[str, str]:
    """Section 12.6: the agent bundle and the imported weights, compared with the repository and the package."""
    agent, imported = require(store, AGENT_RECORD), require(store, IMPORT_RECORD)
    bundle = root / "artifacts" / AGENT_SOURCE
    deployed = zip_hashes(bundle)
    with zipfile.ZipFile(bundle) as archive:
        same_prompt = module_constant(archive.read("app.py").decode(), "SYSTEM_PROMPT") == module_constant(
            (root / "agent" / "app.py").read_text(), "SYSTEM_PROMPT")
    pins_match = pinned_requirements(root / "agent" / "requirements.txt").items() <= agent["packages"].items()
    notes = {"app.py": "refactored after the deploy (run_episode); same system prompt" if same_prompt
             else "DIFFERS, including the system prompt",
             "requirements.txt": "ranges when built; now exact pins, equal to the image's versions" if pins_match
             else "pins differ from the image",
             "pyproject.toml": "ranges when built; now exact pins"}
    rows = [(Raw(tt(name)), Raw(tt(digest[:16])), _repository_status(root, name, digest, notes))
            for name, digest in deployed.items()]
    text = ("Every rollout of the baseline of record, the training job, and the comparison was served by one "
            "AgentCore runtime version, last updated before all three jobs started. The Bedrock import read the "
            "us-east-1 copy of the trained package's merged weights. Both are recorded by "
            "scripts/11b_snapshot_provenance.py and checked by the invariants:")
    small_tt = lambda value: Raw(rf"\texttt{{\scriptsize {latex(value)}}}")  # noqa: E731 - local formatter
    facts = [
        ("AgentCore runtime", f"version {agent['runtime']['version']}, last updated "
                              f"{utc(agent['runtime']['last_updated'], True)}"),
        ("Container image", small_tt(agent["image"]["digest"])),
        ("Base image", Raw(small_tt(agent["image"]["base_image"].partition("@")[0]) + r"\newline "
                           + small_tt(agent["image"]["base_image"].partition("@")[2]))),
        ("CodeBuild build", small_tt(agent["build"]["id"].rsplit(":", 1)[-1])),
        ("Source bundle", Raw(rf"{tt('artifacts/' + AGENT_SOURCE)}, SHA-256 "
                              rf"\texttt{{\scriptsize {agent['source_bundle']['sha256']}}}")),
        ("Import read from", Raw(breakable(imported["import_job"]["source_uri"]))),
    ]
    files = [(Raw(tt(row["file"])), row["source_bytes"], row["copy_bytes"], row["verdict"]) for row in imported["files"]]
    return {
        "PROVENANCE_TEXT": latex(text),
        "TABLE_PROVENANCE": table(["Record", "Value"], facts, "P{3.4cm}P{12cm}"),
        "TABLE_AGENT_FILES": table(["Deployed file", "SHA-256 (start)", "Repository today"], rows, "P{4.4cm}lP{8.2cm}",
                                   caption=r"The agent's deployed source bundle against the repository's \texttt{agent/}."),
        "TABLE_IMPORT_FILES": "{\\small\n" + table(
            ["File", "Package bytes", "Copy bytes", "Verdict"], files, "P{5.6cm}rrl",
            caption="The imported model's files against the trained package's merged weights.") + "\n}",
    }


def _repository_status(root: Path, name: str, digest: str, notes: Mapping[str, str]) -> str:
    """Describes how one deployed agent file compares with the repository today."""
    path = root / "agent" / name
    if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == digest:
        return "identical"
    return notes.get(name, "test file; not used at runtime" if name.startswith("test_") else "differs")


def _invariant_values(invariants: Sequence[tuple[str, bool, str]]) -> dict[str, str]:
    """Section 13: the numbered invariant list and the numbers prose refers to."""
    items = "\n".join(rf"\item {latex(label)}" for label, _, _ in invariants)
    references = {}
    for key, fragment in INVARIANT_REFERENCES.items():
        number = next((str(index) for index, (label, _, _) in enumerate(invariants, start=1) if fragment in label), None)
        if number is None:
            raise KeyError(f"No invariant label contains {fragment!r}.")
        references[key] = number
    count = len(invariants)
    return {"INVARIANT_LIST": f"\\begin{{enumerate}}[leftmargin=*,itemsep=1pt]\n{items}\n\\end{{enumerate}}",
            "INVARIANT_SUMMARY": f"All {count} pass on the evidence in this repository.", **references}


def _test_count(root: Path) -> int:
    """Counts the unit test functions in tests/ and agent/."""
    files = [*root.glob("tests/**/test_*.py"), *root.glob("agent/test_*.py")]
    return sum(len(re.findall(r"^\s*def test_", path.read_text(), flags=re.MULTILINE)) for path in files)

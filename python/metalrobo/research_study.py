"""Run preregistered Numi studies without turning partial runs into claims."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Sequence


SCHEMA = "numi.research-study.v1"


def _canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def protocol_hash(protocol: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(protocol)).hexdigest()


def load_protocol(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("protocol must be a JSON object")
    validate_protocol(value)
    return value


def validate_protocol(protocol: dict[str, Any]) -> None:
    if protocol.get("schema") != SCHEMA:
        raise ValueError(f"protocol schema must be {SCHEMA}")
    if protocol.get("status") != "preregistered":
        raise ValueError("protocol must be frozen with status=preregistered")
    seeds = protocol.get("learner_seeds")
    if not isinstance(seeds, list) or len(seeds) < 5:
        raise ValueError("confirmatory protocol requires at least five learner seeds")
    if len(set(seeds)) != len(seeds) or not all(isinstance(seed, int) for seed in seeds):
        raise ValueError("learner_seeds must be unique integers")
    tasks = protocol.get("tasks")
    methods = protocol.get("methods")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("protocol requires tasks")
    if not isinstance(methods, list) or len(methods) < 2:
        raise ValueError("protocol requires at least two methods")
    method_ids = {method.get("id") for method in methods if isinstance(method, dict)}
    if len(method_ids) != len(methods) or None in method_ids:
        raise ValueError("method ids must be unique strings")
    if not any(bool(method.get("teacher_disconnected_evaluation")) for method in methods):
        raise ValueError("at least one method must require teacher-disconnected evaluation")
    for task in tasks:
        if not isinstance(task, dict) or not task.get("id") or not task.get("train_args"):
            raise ValueError("each task requires id and train_args")
    if not protocol.get("primary_outcomes"):
        raise ValueError("protocol requires primary_outcomes")
    if not protocol.get("claim_gates"):
        raise ValueError("protocol requires claim_gates")


def expand_runs(protocol: dict[str, Any], output_root: Path) -> list[dict[str, Any]]:
    digest = protocol_hash(protocol)
    runs: list[dict[str, Any]] = []
    for task in protocol["tasks"]:
        for method in protocol["methods"]:
            for seed in protocol["learner_seeds"]:
                run_id = f"{task['id']}--{method['id']}--seed-{seed}"
                run_dir = output_root / run_id
                arguments = [
                    *task["train_args"],
                    *method.get("train_args", []),
                    *task.get("method_args", {}).get(method["id"], []),
                ]
                arguments.extend(["--seed", str(seed), "--learner-seed", str(seed)])
                runs.append({
                    "run_id": run_id,
                    "task": task["id"],
                    "method": method["id"],
                    "seed": seed,
                    "protocol_sha256": digest,
                    "run_directory": str(run_dir),
                    "arguments": arguments,
                    "teacher_disconnected_evaluation": bool(
                        method.get("teacher_disconnected_evaluation")
                    ),
                })
    return runs


def _git(repo: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=repo, check=True, text=True,
        capture_output=True,
    ).stdout.strip()


def run_study(protocol_path: Path, output_root: Path, *, limit: int | None) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    repo = Path.cwd().resolve()
    if _git(repo, "branch", "--show-current") != protocol["branch"]:
        raise RuntimeError(f"study must run on {protocol['branch']}")
    if _git(repo, "status", "--porcelain"):
        raise RuntimeError("study requires a clean worktree")
    revision = _git(repo, "rev-parse", "HEAD")
    output_root.mkdir(parents=True, exist_ok=True)
    plan = expand_runs(protocol, output_root)
    completed_count = 0
    for run in plan:
        run_dir = Path(run["run_directory"])
        completion = run_dir / "study-run.json"
        if completion.exists() and json.loads(completion.read_text()).get("status") == "complete":
            continue
        if limit is not None and completed_count >= limit:
            break
        run_dir.mkdir(parents=True, exist_ok=True)
        train_dir = run_dir / "train"
        record = {**run, "revision": revision, "status": "running",
                  "started_at": datetime.now(timezone.utc).isoformat()}
        completion.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        environment = dict(os.environ)
        environment["NUMI_RUN_DIR"] = str(train_dir)
        command = ["numi", "train", *run["arguments"]]
        result = subprocess.run(command, cwd=repo, env=environment, text=True,
                                capture_output=True, check=False)
        (run_dir / "study.stdout.log").write_text(result.stdout)
        (run_dir / "study.stderr.log").write_text(result.stderr)
        record.update({
            "status": "complete" if result.returncode == 0 else "failed",
            "returncode": result.returncode,
            "finished_at": datetime.now(timezone.utc).isoformat(),
        })
        completion.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        if result.returncode != 0:
            raise RuntimeError(f"study run failed: {run['run_id']}")
        completed_count += 1
    return summarize(protocol_path, output_root)


def summarize(protocol_path: Path, output_root: Path) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    plan = expand_runs(protocol, output_root)
    records: list[dict[str, Any]] = []
    for run in plan:
        run_dir = Path(run["run_directory"])
        status_path = run_dir / "study-run.json"
        evidence_path = run_dir / "train" / "evidence.json"
        selection_path = run_dir / "train" / "selection" / "selection.json"
        status = json.loads(status_path.read_text()) if status_path.exists() else {}
        evidence = json.loads(evidence_path.read_text()) if evidence_path.exists() else {}
        selection = json.loads(selection_path.read_text()) if selection_path.exists() else {}
        records.append({
            "run_id": run["run_id"], "task": run["task"], "method": run["method"],
            "seed": run["seed"], "status": status.get("status", "missing"),
            "failed_environment_steps": evidence.get("failed_environment_steps"),
            "environment_steps_per_second": evidence.get(
                "end_to_end_environment_steps_per_second"
            ),
            "selected": selection.get("selected"),
            "candidate_advanced_deployment": selection.get(
                "candidate_advanced_deployment"
            ),
        })
    complete = all(record["status"] == "complete" for record in records)
    zero_failures = complete and all(record["failed_environment_steps"] == 0 for record in records)
    summary = {
        "schema": "numi.research-study-summary.v1",
        "study": protocol["id"],
        "protocol_sha256": protocol_hash(protocol),
        "planned_runs": len(records),
        "completed_runs": sum(record["status"] == "complete" for record in records),
        "confirmatory_claims_unlocked": bool(complete and zero_failures),
        "claim_gate_reason": (
            "all preregistered runs complete with zero failed environment steps"
            if complete and zero_failures else
            "locked until every preregistered run completes with zero failed environment steps"
        ),
        "runs": records,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "study-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a preregistered Numi research study")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("validate", "plan", "run", "analyze"):
        command = commands.add_parser(name)
        command.add_argument("--protocol", type=Path, required=True)
        if name != "validate":
            command.add_argument("--output-root", type=Path, required=True)
        if name == "run":
            command.add_argument("--limit", type=int)
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    options = _parser().parse_args(arguments)
    protocol = load_protocol(options.protocol)
    if options.command == "validate":
        result = {"valid": True, "protocol_sha256": protocol_hash(protocol)}
    elif options.command == "plan":
        result = {"protocol_sha256": protocol_hash(protocol),
                  "runs": expand_runs(protocol, options.output_root)}
    elif options.command == "run":
        result = run_study(options.protocol, options.output_root, limit=options.limit)
    else:
        result = summarize(options.protocol, options.output_root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

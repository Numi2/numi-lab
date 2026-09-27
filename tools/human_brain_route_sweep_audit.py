#!/usr/bin/env python3
"""Verify one matched native motor-program sweep before off-rollout learning."""

import argparse
import json
from pathlib import Path

from human_brain_recovery_audit import manifest, normalized_command, source_without_program
from human_brain_training_cohort import receipt, sha256


def audit(directory):
    cohort = json.loads((directory / "manifest.json").read_text())
    summary = json.loads((directory / "summary.json").read_text())
    workers = cohort["workers"]
    push = cohort.get("push")
    if (not 2 <= workers <= 16 or not cohort["sameSeedBenchmark"] or
            not cohort["sensorAudit"] or not push or
            cohort["stepsPerWorker"] < 1500 or
            len(cohort["workerPrograms"]) != workers or
            len(summary["results"]) != workers or
            summary["status"] != "ACCEPTED_NATIVE_COHORT"):
        raise ValueError("sweep lacks matched accepted native workers")
    if (len(set(cohort["seeds"])) != 1 or
            len(set(push["xForceNewtons"])) != 1 or
            len(set(push["yForceNewtons"])) != 1):
        raise ValueError("sweep changed seed or push between motor candidates")
    baseline_command = baseline_hashes = None
    rows = []
    for index, program in enumerate(cohort["workerPrograms"]):
        worker = directory / f"worker-{index:02d}"
        if (worker / "launch.exit").read_text().splitlines() != [
                "binary_exit_code=0", "tee_exit_code=0", "launcher_exit_code=0"]:
            raise ValueError(f"worker {index} lacks native completion")
        argv, hashes = manifest(worker)
        command = normalized_command(argv)
        source_hashes = source_without_program(hashes)
        if baseline_command is None:
            baseline_command, baseline_hashes = command, source_hashes
        elif command != baseline_command or source_hashes != baseline_hashes:
            raise ValueError(f"worker {index} differs beyond its motor program")
        program_hash = next(line.split()[1] for line in hashes
                            if line.startswith("locomotor_program "))
        if (program_hash != program["sha256"] or
                sha256(Path(program["path"])) != program_hash):
            raise ValueError(f"worker {index} motor bytes disagree")
        actual = receipt(worker / "launch.log", cohort["stepsPerWorker"], 0,
                         True, push["startStep"])
        recorded = summary["results"][index]
        if not actual["accepted"] or not recorded["accepted"] or any(
                actual[key] != recorded[key] for key in (
                    "physicalProgressSHA256", "sensorAuditSHA256",
                    "jointCommitFingerprints", "minimumContacts",
                    "maximumPenetrationM")):
            raise ValueError(f"worker {index} accepted trace disagrees with receipt")
        rows.append({"worker": index, "seed": cohort["seeds"][index],
                     "programPath": program["path"], "programSHA256": program_hash,
                     "physicalSHA256": actual["physicalProgressSHA256"],
                     "sensorSHA256": actual["sensorAuditSHA256"],
                     "minimumContacts": actual["minimumContacts"],
                     "maximumPenetrationM": actual["maximumPenetrationM"],
                     "recoveryTask": actual["recoveryTask"]})
    if len({row["programSHA256"] for row in rows}) != workers:
        raise ValueError("sweep reuses a motor candidate")
    return {"schema": "numi.human.brain.route-sweep-audit.v1",
            "acceptedSweep": True, "recoveredCount": sum(
                row["recoveryTask"]["recovered"] for row in rows),
            "task": {"steps": cohort["stepsPerWorker"], "seed": cohort["seeds"][0],
                     "push": push},
            "sourceHashesWithoutProgram": baseline_hashes,
            "nativeCohortManifestSHA256": sha256(directory / "manifest.json"),
            "nativeCohortSummarySHA256": sha256(directory / "summary.json"),
            "workers": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.directory)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"acceptedSweep": result["acceptedSweep"],
                      "workerCount": len(result["workers"]),
                      "recoveredCount": result["recoveredCount"]}, sort_keys=True))


if __name__ == "__main__":
    main()

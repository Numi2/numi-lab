#!/usr/bin/env python3
"""Audit matched native baseline/candidate recovery on training and unseen pushes."""

import argparse
import json
from pathlib import Path

from human_brain_recovery_audit import manifest, normalized_command, source_without_program
from human_brain_training_cohort import receipt, sha256


def audit(directory):
    cohort = json.loads((directory / "manifest.json").read_text())
    summary = json.loads((directory / "summary.json").read_text())
    count = cohort["workers"]
    push = cohort.get("push")
    if (count < 4 or count % 2 or cohort["stepsPerWorker"] < 1500 or
            not cohort["sameSeedBenchmark"] or not cohort["sensorAudit"] or
            not push or summary["status"] != "ACCEPTED_NATIVE_COHORT" or
            len(cohort["workerPrograms"]) != count or
            len(summary["results"]) != count):
        raise ValueError("matched native recovery cohort is incomplete")
    half = count // 2
    arms = []
    for index in range(count):
        worker = directory / f"worker-{index:02d}"
        if (worker / "launch.exit").read_text().splitlines() != [
                "binary_exit_code=0", "tee_exit_code=0", "launcher_exit_code=0"]:
            raise ValueError(f"worker {index} did not finish")
        argv, hashes = manifest(worker)
        program = cohort["workerPrograms"][index]
        if (sha256(Path(program["path"])) != program["sha256"] or
                not any(line.startswith("locomotor_program " + program["sha256"])
                        for line in hashes)):
            raise ValueError(f"worker {index} program bytes disagree")
        actual = receipt(worker / "launch.log", cohort["stepsPerWorker"], 0,
                         True, push["startStep"])
        recorded = summary["results"][index]
        if not actual["accepted"] or not recorded["accepted"] or any(
                actual[key] != recorded[key] for key in (
                    "physicalProgressSHA256", "sensorAuditSHA256",
                    "jointCommitFingerprints", "minimumContacts",
                    "maximumPenetrationM")):
            raise ValueError(f"worker {index} trace disagrees with receipt")
        arms.append((argv, hashes, actual, program["sha256"]))
    if (len({arms[index][3] for index in range(half)}) != 1 or
            len({arms[index][3] for index in range(half, count)}) != 1 or
            arms[0][3] == arms[half][3]):
        raise ValueError("candidate pair does not freeze one baseline and one update")
    pairs = []
    for index in range(half):
        before, after = arms[index], arms[index + half]
        if (normalized_command(before[0]) != normalized_command(after[0]) or
                source_without_program(before[1]) != source_without_program(after[1]) or
                cohort["seeds"][index] != cohort["seeds"][index + half] or
                any(push[axis][index] != push[axis][index + half]
                    for axis in ("xForceNewtons", "yForceNewtons"))):
            raise ValueError(f"pair {index} changed its native task or source")
        left, right = before[2]["recoveryTask"], after[2]["recoveryTask"]
        pairs.append({"pair": index,
                      "pushXNewtons": push["xForceNewtons"][index],
                      "pushYNewtons": push["yForceNewtons"][index],
                      "baseline": left, "candidate": right,
                      "successDifference": int(right["recovered"]) - int(left["recovered"]),
                      "contactSurvivalStepDifference":
                          (right["firstContactLossStep"] or cohort["stepsPerWorker"] + 1) -
                          (left["firstContactLossStep"] or cohort["stepsPerWorker"] + 1),
                      "peakPenetrationM": {"baseline": before[2]["maximumPenetrationM"],
                                           "candidate": after[2]["maximumPenetrationM"]}})
    return {"schema": "numi.human.brain.balance-pair-audit.v1",
            "sourceBound": True,
            "baselineProgramSHA256": arms[0][3],
            "candidateProgramSHA256": arms[half][3],
            "nativeCohortManifestSHA256": sha256(directory / "manifest.json"),
            "nativeCohortSummarySHA256": sha256(directory / "summary.json"),
            "baselineSuccesses": sum(pair["baseline"]["recovered"] for pair in pairs),
            "candidateSuccesses": sum(pair["candidate"]["recovered"] for pair in pairs),
            "pairs": pairs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.directory)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: result[key] for key in (
        "sourceBound", "baselineSuccesses", "candidateSuccesses",
        "baselineProgramSHA256", "candidateProgramSHA256")}, sort_keys=True))


if __name__ == "__main__":
    main()

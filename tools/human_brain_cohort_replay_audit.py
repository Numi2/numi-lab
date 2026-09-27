#!/usr/bin/env python3
"""Qualify exact replay and supported throughput for two native 16-Human cohorts."""

import argparse
import json
from pathlib import Path

from human_brain_recovery_audit import manifest, normalized_command
from human_brain_training_cohort import GIB, receipt


def read_cohort(directory):
    cohort = json.loads((directory / "manifest.json").read_text())
    summary = json.loads((directory / "summary.json").read_text())
    if (cohort["workers"] != 16 or cohort["stepsPerWorker"] < 2000 or
            not cohort["push"] or
            summary["status"] != "ACCEPTED_NATIVE_COHORT" or
            len(summary["results"]) != 16 or
            not all(result.get("sensorAuditSHA256") for result in summary["results"])):
        raise ValueError(f"incomplete supported native cohort: {directory}")
    return cohort, summary


def audit(first, replay):
    first_cohort, first_summary = read_cohort(first)
    replay_cohort, replay_summary = read_cohort(replay)
    if first_cohort != replay_cohort:
        raise ValueError("cohort source, program, seeds, or task differs")
    steps = first_cohort["stepsPerWorker"]
    rows = []
    for index in range(16):
        pair = []
        for directory, summary in ((first, first_summary), (replay, replay_summary)):
            worker = directory / f"worker-{index:02d}"
            if (worker / "launch.exit").read_text().splitlines() != [
                    "binary_exit_code=0", "tee_exit_code=0", "launcher_exit_code=0"]:
                raise ValueError(f"worker {index} did not finish its native owner")
            argv, hashes = manifest(worker)
            actual = receipt(worker / "launch.log", steps, 0, True)
            recorded = summary["results"][index]
            if not actual["accepted"] or not recorded["accepted"] or any(
                    actual[key] != recorded[key] for key in (
                        "physicalProgressSHA256", "sensorAuditSHA256",
                        "jointCommitFingerprints", "minimumContacts",
                        "maximumPenetrationM")):
                raise ValueError(f"worker {index} receipt disagrees with native trace")
            pair.append((argv, hashes, actual))
        left, right = pair
        exact = (normalized_command(left[0]) == normalized_command(right[0])
                 and left[1] == right[1]
                 and all(left[2][key] == right[2][key] for key in (
                     "physicalProgressSHA256", "sensorAuditSHA256",
                     "jointCommitFingerprints")))
        supported = all(result[2]["minimumContacts"] >= 6 and
                        result[2]["maximumPenetrationM"] <= 5e-6
                        for result in pair)
        rows.append({"worker": index, "seed": first_cohort["seeds"][index],
                     "exact": exact, "supported": supported,
                     "physicalSHA256": left[2]["physicalProgressSHA256"],
                     "sensorSHA256": left[2]["sensorAuditSHA256"],
                     "maximumPenetrationM": max(left[2]["maximumPenetrationM"],
                                                right[2]["maximumPenetrationM"])})
    summaries = (first_summary, replay_summary)
    distinct = len({row["physicalSHA256"] for row in rows}) == 16
    throughput = min(summary["acceptedSimulatedSecondsPerWallHour"]
                     for summary in summaries)
    resource_safe = all(summary["maximumSwapUsedMB"] <= summary["swapUsedMBBefore"]
                        and summary["freeBytesAfter"] >= 10 * GIB
                        and summary["peakAggregateRSSKiB"] < 16 * GIB / 1024
                        for summary in summaries)
    passed = (all(row["exact"] and row["supported"] for row in rows)
              and distinct and resource_safe and throughput >= 864)
    return {"schema": "numi.human.brain.supported-cohort-replay.v1",
            "passed": passed, "workers": rows, "distinctPhysicalTraces": distinct,
            "exactReplayCount": sum(row["exact"] for row in rows),
            "supportedCount": sum(row["supported"] for row in rows),
            "acceptedSimulatedSecondsPerHour": [
                summary["acceptedSimulatedSecondsPerWallHour"] for summary in summaries],
            "minimumAcceptedSimulatedSecondsPerHour": throughput,
            "resourceSafe": resource_safe,
            "peakAggregateRSSKiB": max(summary["peakAggregateRSSKiB"]
                                       for summary in summaries),
            "swapUsedMBBefore": [summary["swapUsedMBBefore"] for summary in summaries],
            "maximumSwapUsedMB": max(summary["maximumSwapUsedMB"]
                                     for summary in summaries),
            "freeBytesAfter": [summary["freeBytesAfter"] for summary in summaries]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first", type=Path)
    parser.add_argument("replay", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.first, args.replay)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in result.items()
                      if key != "workers"}, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

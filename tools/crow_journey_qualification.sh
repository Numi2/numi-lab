#!/bin/bash

# Final deployment qualification for a promoted neural Crow PolicyPack.
# Produces the exact 11-milestone x 3-seed evidence matrix consumed by
# BirdFlow's presentation gate and one accepted-state CrowReplayPack.

set -eu

root=${NUMI_CROW_QUALIFICATION_ROOT:?set NUMI_CROW_QUALIFICATION_ROOT}
build=${NUMI_CROW_QUALIFICATION_BUILD:?set NUMI_CROW_QUALIFICATION_BUILD}
policy=${NUMI_CROW_QUALIFICATION_POLICY:?set NUMI_CROW_QUALIFICATION_POLICY}
runs=${NUMI_CROW_QUALIFICATION_RUNS:?set NUMI_CROW_QUALIFICATION_RUNS}
course=${NUMI_CROW_QUALIFICATION_COURSE:-sensor-fast}
seeds=${NUMI_CROW_QUALIFICATION_SEEDS:-2650443581,2650443582,2650443583}
environments=${NUMI_CROW_QUALIFICATION_ENVIRONMENTS:-32}
steps=${NUMI_CROW_QUALIFICATION_STEPS:-1600}
replay=${NUMI_CROW_QUALIFICATION_REPLAY:-$runs/accepted-full-journey.crowreplay.json}
python=${NUMI_CROW_QUALIFICATION_PYTHON:-/usr/bin/python3}

case "$course" in
  state) variant=v8-neural; task=birdflow_american_crow_journey_v8_neural ;;
  sensor-fast) variant=v9-visual-neural; task=birdflow_american_crow_journey_v9_visual_neural ;;
  *) echo "NUMI_CROW_QUALIFICATION_COURSE must be state or sensor-fast" >&2; exit 2 ;;
esac
[[ "$environments" =~ ^[1-9][0-9]*$ && "$steps" =~ ^[1-9][0-9]*$ ]] || {
  echo "Crow qualification environments and steps must be positive integers" >&2
  exit 2
}
[ -x "$build/bin/metalrobo_task_rollout" ] && [ -s "$policy" ] && [ -x "$python" ] || {
  echo "Crow qualification requires a built rollout, PolicyPack, and Python" >&2
  exit 2
}

old_ifs=$IFS
IFS=,
set -- $seeds
IFS=$old_ifs
[ "$#" -eq 3 ] || {
  echo "Crow qualification requires exactly three comma-separated seeds" >&2
  exit 2
}
seed_a=$1; seed_b=$2; seed_c=$3
[[ "$seed_a" =~ ^[0-9]+$ && "$seed_b" =~ ^[0-9]+$ && "$seed_c" =~ ^[0-9]+$ ]] || {
  echo "Crow qualification seeds must be non-negative integers" >&2
  exit 2
}
[ "$seed_a" != "$seed_b" ] && [ "$seed_a" != "$seed_c" ] && [ "$seed_b" != "$seed_c" ] || {
  echo "Crow qualification seeds must be distinct" >&2
  exit 2
}

mkdir -p "$runs"
policy_sha=$(shasum -a 256 "$policy" | awk '{print $1}')

milestone_for_band() {
  case "$1" in
    0) echo standing ;; 1) echo walking ;; 2) echo takeoff ;;
    3) echo cruise ;; 4) echo takeoff-cruise ;; 5) echo turn-left ;;
    6) echo turn-right ;; 7) echo approach ;; 8) echo touchdown ;;
    9) echo landed-hold ;; 10) echo full-journey ;;
  esac
}

run_evaluation() {
  qualification_run=$1
  qualification_milestone=$2
  qualification_seed=$3
  shift 3
  NUMI_LAB_ROOT="$root" NUMI_BUILD_DIR="$build" \
    NUMI_RUN_DIR="$qualification_run" \
    "$root/tools/numi" crow journey evaluate \
      --variant "$variant" --milestone "$qualification_milestone" \
      --policy-pack "$policy" --envs "$environments" --steps "$steps" \
      --repeats 1 --chunk 1 --seed "$qualification_seed" \
      --no-scheduled-resets "$@" >/dev/null
}

for seed in "$seed_a" "$seed_b" "$seed_c"; do
  band=0
  while [ "$band" -le 10 ]; do
    milestone=$(milestone_for_band "$band")
    run="$runs/band${band}-${milestone}-seed${seed}"
    evidence="$run/evidence.json"
    if [ ! -s "$evidence" ]; then
      [ ! -e "$run" ] || {
        echo "existing Crow qualification run is incomplete: $run" >&2
        exit 5
      }
      if [ "$band" -eq 10 ] && [ "$seed" = "$seed_a" ]; then
        [ ! -e "$replay" ] || {
          echo "Crow replay exists without its completed qualification run: $replay" >&2
          exit 5
        }
      fi
      echo "qualifying neural Crow $course band $band ($milestone), seed $seed"
      if [ "$band" -eq 10 ] && [ "$seed" = "$seed_a" ]; then
        run_evaluation "$run" "$milestone" "$seed" \
          --crow-replay-pack "$replay"
      else
        run_evaluation "$run" "$milestone" "$seed"
      fi
    fi
    band=$((band + 1))
  done
done

[ -s "$replay" ] || {
  echo "Crow qualification completed without an accepted-state replay" >&2
  exit 5
}

"$python" - "$runs" "$task" "$policy" "$policy_sha" "$replay" \
  "$environments" "$steps" "$seed_a" "$seed_b" "$seed_c" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

runs, task, policy, policy_sha, replay, envs, steps, *seeds = sys.argv[1:]
root = Path(runs)
paths = sorted(root.glob("band*-seed*/evidence.json"))
if len(paths) != 33:
    raise SystemExit(f"qualification requires 33 evidence files, found {len(paths)}")
records = [json.loads(path.read_text()) for path in paths]

def floor(band):
    return 0.95 if band in (0, 9) else 0.85 if band == 1 else 0.65

for path, record in zip(paths, records):
    band = record.get("minimum_sampled_difficulty_band")
    failures = []
    if record.get("task") != task or record.get("world_source") != task:
        failures.append("task fingerprint mismatch")
    if band != record.get("maximum_sampled_difficulty_band") or band not in range(11):
        failures.append("difficulty band mismatch")
    if record.get("benchmark_seed") not in map(int, seeds):
        failures.append("seed mismatch")
    if record.get("action_source") != "policy_pack" or record.get("birdflow_journey_teacher") is not False:
        failures.append("evaluation was not autonomous PolicyPack control")
    if record.get("failed_environment_steps") != 0:
        failures.append("failed environment steps")
    if record.get("termination_count") != record.get("timeout_count"):
        failures.append("physical termination")
    if record.get("mean_tracking_score", float("-inf")) < floor(band):
        failures.append("tracking below milestone floor")
    if record.get("mean_tilt", float("inf")) > 0.35 or record.get("maximum_tilt", float("inf")) >= 0.8:
        failures.append("tilt envelope exceeded")
    if band in (2, 3, 4, 5, 6, 7, 10) and record.get("maximum_root_height", float("-inf")) < 0.55:
        failures.append("flight height below floor")
    outcomes = record.get("outcomes", {})
    if band in (7, 8, 9, 10):
        warning = outcomes.get("approach_pitch_warning_fraction", {}).get("mean", float("inf"))
        full = outcomes.get("approach_pitch_full_envelope_fraction", {}).get("mean", float("inf"))
        if warning > 0.05 or full > 0.000001:
            failures.append("approach diagnostic envelope exceeded")
    if failures:
        raise SystemExit(f"{path}: " + "; ".join(failures))

bands = sorted({r["minimum_sampled_difficulty_band"] for r in records})
seen_seeds = sorted({r["benchmark_seed"] for r in records})
if bands != list(range(11)) or seen_seeds != sorted(map(int, seeds)):
    raise SystemExit("qualification matrix does not cover every band and seed")

replay_path = Path(replay)
pack = json.loads(replay_path.read_text())
if pack.get("schema") != "numi.crow-replay.v1" or pack.get("payload", {}).get("task") != task:
    raise SystemExit("CrowReplayPack identity mismatch")
if pack.get("payload", {}).get("classification") != "simulated accepted-state replay":
    raise SystemExit("CrowReplayPack classification mismatch")
if len(pack.get("payload_sha256", "")) != 64 or len(pack.get("payload", {}).get("frames", [])) < 2:
    raise SystemExit("CrowReplayPack is incomplete")

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

summary = {
    "schema": "numi.crow-qualification.v1",
    "classification": "simulated all-milestone neural policy qualification",
    "task": task,
    "policy_pack": policy,
    "policy_pack_sha256": policy_sha,
    "run_count": len(records),
    "environment_count": len(records) * int(envs),
    "steps_per_run": int(steps),
    "benchmark_seeds": seen_seeds,
    "minimum_mean_tracking_score": min(r["mean_tracking_score"] for r in records),
    "maximum_mean_tilt": max(r["mean_tilt"] for r in records),
    "maximum_tilt": max(r["maximum_tilt"] for r in records),
    "maximum_root_height": max(r["maximum_root_height"] for r in records),
    "failed_environment_steps": sum(r["failed_environment_steps"] for r in records),
    "non_timeout_terminations": sum(r["termination_count"] - r["timeout_count"] for r in records),
    "crow_replay_pack": str(replay_path),
    "crow_replay_pack_sha256": sha(replay_path),
    "crow_replay_payload_sha256": pack["payload_sha256"],
    "evidence_sha256": {str(path): sha(path) for path in paths},
}
(root / "qualification.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(f"qualified {task}: {len(records)} runs; replay={replay_path}")
PY

shasum -a 256 "$runs/qualification.json" "$replay" > "$runs/qualification.sha256"
echo "neural Crow $course qualification completed: $runs/qualification.json"

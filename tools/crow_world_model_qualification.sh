#!/bin/bash

# Deterministic incumbent-versus-candidate qualification for Crow v10.
# The candidate is promotable only after three seeds on the authored training
# course and both held-out geometry splits. Model predictions never satisfy a
# gate; every comparison below comes from native accepted Metal state.

set -eu

root=${NUMI_CROW_WORLD_MODEL_ROOT:?set NUMI_CROW_WORLD_MODEL_ROOT}
build=${NUMI_CROW_WORLD_MODEL_BUILD:?set NUMI_CROW_WORLD_MODEL_BUILD}
baseline=${NUMI_CROW_WORLD_MODEL_BASELINE:?set NUMI_CROW_WORLD_MODEL_BASELINE}
candidate=${NUMI_CROW_WORLD_MODEL_CANDIDATE:?set NUMI_CROW_WORLD_MODEL_CANDIDATE}
runs=${NUMI_CROW_WORLD_MODEL_RUNS:?set NUMI_CROW_WORLD_MODEL_RUNS}
seeds=${NUMI_CROW_WORLD_MODEL_SEEDS:-2650705001,2650705002,2650705003}
environments=${NUMI_CROW_WORLD_MODEL_ENVIRONMENTS:-32}
steps=${NUMI_CROW_WORLD_MODEL_STEPS:-1600}
python=${NUMI_CROW_WORLD_MODEL_PYTHON:-/usr/bin/python3}
rollout=$build/bin/metalrobo_task_rollout
metallib=$build/shaders/MetalRobo.metallib
visual=$root/assets/crow_navigation_course/crow-navigation.sensor-fast.visual-observation.json

[[ "$environments" =~ ^[1-9][0-9]*$ && "$steps" =~ ^[1-9][0-9]*$ ]] || {
  echo "Crow world-model qualification counts must be positive integers" >&2
  exit 2
}
for required in "$rollout" "$metallib" "$visual" "$baseline" "$candidate" "$python"; do
  [ -s "$required" ] || {
    echo "Crow world-model qualification input is unavailable: $required" >&2
    exit 2
  }
done

old_ifs=$IFS
IFS=,
set -- $seeds
IFS=$old_ifs
[ "$#" -eq 3 ] || {
  echo "Crow world-model qualification requires exactly three seeds" >&2
  exit 2
}
seed_a=$1; seed_b=$2; seed_c=$3
[[ "$seed_a" =~ ^[0-9]+$ && "$seed_b" =~ ^[0-9]+$ && "$seed_c" =~ ^[0-9]+$ ]] || {
  echo "Crow world-model qualification seeds must be non-negative integers" >&2
  exit 2
}
[ "$seed_a" != "$seed_b" ] && [ "$seed_a" != "$seed_c" ] && [ "$seed_b" != "$seed_c" ] || {
  echo "Crow world-model qualification seeds must be distinct" >&2
  exit 2
}

mkdir -p "$runs"
for controller in baseline candidate; do
  if [ "$controller" = baseline ]; then policy=$baseline; else policy=$candidate; fi
  for course in training held-out-a held-out-b; do
    for seed in "$seed_a" "$seed_b" "$seed_c"; do
      evidence=$runs/$controller-$course-seed$seed.json
      if [ ! -s "$evidence" ]; then
        [ ! -e "$evidence" ] || {
          echo "incomplete qualification evidence exists: $evidence" >&2
          exit 5
        }
        echo "qualifying Crow v10 $controller on $course, seed $seed"
        "$rollout" \
          --metallib "$metallib" \
          --birdflow-american-crow-journey \
          --birdflow-journey-variant v10-world-model \
          --birdflow-navigation-course "$course" \
          --visual-observation-config "$visual" \
          --policy-pack "$policy" \
          --envs "$environments" --steps "$steps" --repeats 1 --chunk 1 \
          --no-scheduled-resets \
          --minimum-difficulty-band 10 --maximum-difficulty-band 10 \
          --seed "$seed" > "$evidence"
      fi
    done
  done
done

"$python" - "$runs" "$baseline" "$candidate" "$environments" "$steps" \
  "$seed_a" "$seed_b" "$seed_c" <<'PY'
import hashlib
import json
import math
import sys
from pathlib import Path

runs, baseline, candidate, environments, steps, *seed_values = sys.argv[1:]
root = Path(runs)
seeds = [int(value) for value in seed_values]
task = "birdflow_american_crow_navigation_v10_world_model"
courses = ("training", "held-out-a", "held-out-b")
controllers = ("baseline", "candidate")

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

records = {}
failures = []
for controller in controllers:
    for course in courses:
        values = []
        for seed in seeds:
            path = root / f"{controller}-{course}-seed{seed}.json"
            try:
                value = json.loads(path.read_text())
            except Exception as error:
                failures.append(f"{path}: unreadable evidence ({error})")
                continue
            checks = {
                "task identity": value.get("task") == task,
                "course identity": value.get("birdflow_navigation_course") == course,
                "seed identity": value.get("benchmark_seed") == seed,
                "environment count": value.get("environments") == int(environments),
                "step count": value.get("steps_per_repeat") == int(steps),
                "autonomous policy": value.get("action_source") == "policy_pack"
                    and value.get("birdflow_journey_teacher") is False,
                "scheduled resets disabled": value.get("scheduled_resets") is False,
                "visual observation": value.get("visual_observation") is True,
                "no failed device steps": value.get("failed_environment_steps") == 0,
            }
            for label, passed in checks.items():
                if not passed:
                    failures.append(f"{path}: {label} failed")
            values.append((path, value))
        records[(controller, course)] = values

def aggregate(values):
    items = [value for _, value in values]
    forbidden = sum(int(value.get("termination_reason_counts", {}).get("3", 0)) for value in items)
    return {
        "run_count": len(items),
        "termination_count": sum(int(value.get("termination_count", 0)) for value in items),
        "forbidden_contact_termination_count": forbidden,
        "timeout_count": sum(int(value.get("timeout_count", 0)) for value in items),
        "mean_root_height_m": sum(float(value.get("mean_root_height", math.nan)) for value in items) / len(items),
        "mean_tracking_score": sum(float(value.get("mean_tracking_score", math.nan)) for value in items) / len(items),
        "maximum_tilt": max(float(value.get("maximum_tilt", math.inf)) for value in items),
        "failed_environment_steps": sum(int(value.get("failed_environment_steps", 0)) for value in items),
    }

metrics = {
    controller: {
        course: aggregate(records[(controller, course)])
        for course in courses if records[(controller, course)]
    }
    for controller in controllers
}

gates = []
for course in courses:
    if course not in metrics.get("baseline", {}) or course not in metrics.get("candidate", {}):
        gates.append({"course": course, "gate": "complete matrix", "passed": False})
        continue
    base = metrics["baseline"][course]
    cand = metrics["candidate"][course]
    course_gates = (
        ("forbidden contacts non-increasing", cand["forbidden_contact_termination_count"] <= base["forbidden_contact_termination_count"]),
        ("terminations non-increasing", cand["termination_count"] <= base["termination_count"]),
        ("mean height retained", cand["mean_root_height_m"] >= base["mean_root_height_m"] - 0.05),
        ("tracking retained", cand["mean_tracking_score"] >= base["mean_tracking_score"] - 0.10),
        ("tilt safety envelope", cand["maximum_tilt"] < 0.85),
    )
    for label, passed in course_gates:
        gates.append({"course": course, "gate": label, "passed": bool(passed)})

promoted = not failures and all(item["passed"] for item in gates)
evidence_hashes = {
    str(path.relative_to(root)): sha(path)
    for values in records.values() for path, _ in values
}
payload = {
    "classification": "simulated native held-out controller qualification",
    "task": task,
    "decision": "promote" if promoted else "reject",
    "baseline_policy_pack": str(Path(baseline).resolve()),
    "baseline_policy_pack_sha256": sha(baseline),
    "candidate_policy_pack": str(Path(candidate).resolve()),
    "candidate_policy_pack_sha256": sha(candidate),
    "environment_count_per_run": int(environments),
    "steps_per_run": int(steps),
    "seeds": seeds,
    "metrics": metrics,
    "gates": gates,
    "contract_failures": failures,
    "evidence_sha256": evidence_hashes,
}
canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
envelope = {
    "schema": "numi.crow-world-model-qualification.v1",
    "payload_sha256": hashlib.sha256(canonical).hexdigest(),
    "payload": payload,
}
(root / "qualification.json").write_text(json.dumps(envelope, indent=2, sort_keys=True) + "\n")
print(json.dumps({"decision": payload["decision"], "qualification": str(root / "qualification.json")}, sort_keys=True))
raise SystemExit(0 if promoted else 6)
PY

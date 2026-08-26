#!/bin/bash

# Durable neural-only Crow journey curriculum. Each rung is trained and then
# subjected to Numi's matched incumbent/candidate held-out selector. A rejected
# candidate is retained as evidence but never becomes the next rung's parent.

set -eu

root=${NUMI_CROW_CURRICULUM_ROOT:?set NUMI_CROW_CURRICULUM_ROOT}
build=${NUMI_CROW_CURRICULUM_BUILD:?set NUMI_CROW_CURRICULUM_BUILD}
mlx=${NUMI_CROW_CURRICULUM_MLX:?set NUMI_CROW_CURRICULUM_MLX}
runs=${NUMI_CROW_CURRICULUM_RUNS:?set NUMI_CROW_CURRICULUM_RUNS}
parent_policy=${NUMI_CROW_PARENT_POLICY:-}
parent_state=${NUMI_CROW_PARENT_STATE:-}
start_band=${NUMI_CROW_START_BAND:-0}
maximum_band=${NUMI_CROW_MAXIMUM_BAND:-10}
course=${NUMI_CROW_COURSE:-state}
environments=${NUMI_CROW_ENVIRONMENTS:-2048}
steps=${NUMI_CROW_STEPS:-32}
updates=${NUMI_CROW_UPDATES:-250}
chunk=${NUMI_CROW_CHUNK:-16}
selection_environments=${NUMI_CROW_SELECTION_ENVIRONMENTS:-512}
selection_seed=${NUMI_CROW_SELECTION_SEED:-2650443581}
seed_base=${NUMI_CROW_SEED_BASE:-2650445000}
maximum_retries=${NUMI_CROW_MAXIMUM_RETRIES:-3}

case "$course" in
  state) journey_variant=v8-neural; visual_arguments=() ;;
  sensor-fast)
    journey_variant=v9-visual-neural
    visual_arguments=(
      --visual-observation-config
      "$root/assets/crow_journey_window/crow-journey.sensor-fast.visual-observation.json"
    )
    ;;
  *) echo "NUMI_CROW_COURSE must be state or sensor-fast" >&2; exit 2 ;;
esac
[[ "$start_band" =~ ^([0-9]|10)$ && "$maximum_band" =~ ^([0-9]|10)$ ]] || {
  echo "Crow curriculum bands must be integers in 0...10" >&2; exit 2;
}
[ "$start_band" -le "$maximum_band" ] || {
  echo "Crow start band exceeds maximum band" >&2; exit 2;
}
[ -x "$build/bin/metalrobo_task_train" ] && [ -x "$mlx" ] || {
  echo "Crow curriculum requires a built trainer and MLX Python" >&2; exit 2;
}
mkdir -p "$runs"

milestone_for_band() {
  case "$1" in
    0) echo standing ;; 1) echo walking ;; 2) echo takeoff ;;
    3) echo cruise ;; 4) echo takeoff-cruise ;; 5) echo turn-left ;;
    6) echo turn-right ;; 7) echo approach ;; 8) echo touchdown ;;
    9) echo landed-hold ;; 10) echo full-journey ;;
  esac
}

band=$start_band
retry=0
while [ "$band" -le "$maximum_band" ]; do
  milestone=$(milestone_for_band "$band")
  seed=$((seed_base + band + retry * 1000))
  run="$runs/${journey_variant}-band${band}-${milestone}-seed${seed}-r${retry}"
  selection="$run/selection/selection.json"

  if [ ! -s "$selection" ]; then
    [ ! -e "$run" ] || {
      echo "existing Crow run lacks a completed selection: $run" >&2
      exit 7
    }
    mkdir -p "$run"
    if [ -n "$parent_state" ] && [ -s "$parent_state" ]; then
      cp "$parent_state" "$run/learner.safetensors"
    fi
    common=(
      journey train --variant "$journey_variant" --milestone "$milestone"
      --envs "$environments" --steps "$steps" --updates "$updates"
      --chunk "$chunk" --seed "$seed" --learner-seed "$seed"
      --checkpoint-directory "$run/checkpoints" --checkpoint-interval 20
      "${visual_arguments[@]}" --verbose
    )
    if [ -n "$parent_policy" ] && [ -s "$parent_policy" ]; then
      common+=(--policy-pack "$parent_policy")
    fi
    echo "launching neural Crow $course band $band ($milestone), retry $retry"
    if NUMI_LAB_ROOT="$root" NUMI_BUILD_DIR="$build" \
      NUMI_MLX_PYTHON="$mlx" NUMI_RUN_DIR="$run" \
      NUMI_SELECTION_ENVS="$selection_environments" \
      NUMI_SELECTION_SEED="$selection_seed" \
      "$root/tools/numi" crow "${common[@]}"; then
      :
    elif [ ! -s "$selection" ]; then
      echo "Crow band $band failed without a durable selection" >&2
      exit 5
    fi
  fi

  if "$mlx" -c \
    'import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get("candidate_advanced_deployment", False) else 1)' \
    "$selection"; then
    parent_policy="$run/deployment.policypack"
    parent_state="$run/learner.safetensors"
    [ -s "$parent_policy" ] || {
      echo "selected Crow deployment is missing: $parent_policy" >&2; exit 5;
    }
    printf '{"schema":"numi.crow-curriculum-progress.v1","variant":"%s","course":"%s","completed_band":%s,"milestone":"%s","deployment":"%s","selection":"%s"}\n' \
      "$journey_variant" "$course" "$band" "$milestone" "$parent_policy" "$selection" \
      > "$runs/progress.json"
    band=$((band + 1))
    retry=0
  else
    retry=$((retry + 1))
    [ "$retry" -le "$maximum_retries" ] || {
      echo "Crow band $band retained its incumbent after $maximum_retries retries" >&2
      exit 6
    }
    echo "Crow band $band retained incumbent; retrying with seed $((seed_base + band + retry * 1000))"
  fi
done

echo "neural Crow $course curriculum bands $start_band-$maximum_band completed"

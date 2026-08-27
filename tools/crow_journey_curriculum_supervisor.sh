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
parent_mode=${NUMI_CROW_PARENT_MODE:-auto}
start_band=${NUMI_CROW_START_BAND:-0}
maximum_band=${NUMI_CROW_MAXIMUM_BAND:-10}
course=${NUMI_CROW_COURSE:-state}
environments=${NUMI_CROW_ENVIRONMENTS:-2048}
steps=${NUMI_CROW_STEPS:-32}
updates=${NUMI_CROW_UPDATES:-250}
chunk=${NUMI_CROW_CHUNK:-16}
checkpoint_interval=${NUMI_CROW_CHECKPOINT_INTERVAL:-50}
selection_environments=${NUMI_CROW_SELECTION_ENVIRONMENTS:-512}
selection_seed=${NUMI_CROW_SELECTION_SEED:-2650443581}
seed_base=${NUMI_CROW_SEED_BASE:-2650445000}
maximum_retries=${NUMI_CROW_MAXIMUM_RETRIES:-3}
teacher_distillation=${NUMI_CROW_TEACHER_DISTILLATION:-1}
teacher_student_authority=${NUMI_CROW_TEACHER_STUDENT_AUTHORITY:-0.25}
rehearsal_depth=${NUMI_CROW_REHEARSAL_DEPTH:-0}
rehearsal_minimum_band=${NUMI_CROW_REHEARSAL_MINIMUM_BAND:-}
difficulty_sampling_exponent=${NUMI_CROW_DIFFICULTY_SAMPLING_EXPONENT:-}
learning_rate=${NUMI_CROW_LEARNING_RATE:-}
initial_log_standard_deviation=${NUMI_CROW_INITIAL_LOG_STANDARD_DEVIATION:-}
retention_policy=${NUMI_CROW_RETENTION_POLICY:-}
retention_coefficient=${NUMI_CROW_RETENTION_COEFFICIENT:-1.0}
retention_maximum_band=${NUMI_CROW_RETENTION_MAXIMUM_BAND:-}
retention_protected_actor_only=${NUMI_CROW_RETENTION_PROTECTED_ACTOR_ONLY:-0}

case "$course" in
  state)
    journey_variant=v8-neural
    visual_config=
    ;;
  sensor-fast)
    journey_variant=v9-visual-neural
    visual_config="$root/assets/crow_journey_window/crow-journey.sensor-fast.visual-observation.json"
    ;;
  *) echo "NUMI_CROW_COURSE must be state or sensor-fast" >&2; exit 2 ;;
esac
case "$parent_mode" in
  auto)
    if [ "$course" = sensor-fast ] && [ -n "$parent_policy" ] && \
      [ -z "$parent_state" ]; then
      parent_mode=actor-transfer
    else
      parent_mode=resume
    fi
    ;;
  actor-transfer|resume) ;;
  *) echo "NUMI_CROW_PARENT_MODE must be auto, actor-transfer, or resume" >&2; exit 2 ;;
esac
case "$teacher_distillation" in
  0|1) ;;
  *) echo "NUMI_CROW_TEACHER_DISTILLATION must be 0 or 1" >&2; exit 2 ;;
esac
case "$retention_protected_actor_only" in
  0|1) ;;
  *) echo "NUMI_CROW_RETENTION_PROTECTED_ACTOR_ONLY must be 0 or 1" >&2; exit 2 ;;
esac
[[ "$start_band" =~ ^([0-9]|10)$ && "$maximum_band" =~ ^([0-9]|10)$ ]] || {
  echo "Crow curriculum bands must be integers in 0...10" >&2; exit 2;
}
[ "$start_band" -le "$maximum_band" ] || {
  echo "Crow start band exceeds maximum band" >&2; exit 2;
}
[[ "$checkpoint_interval" =~ ^[1-9][0-9]*$ ]] || {
  echo "NUMI_CROW_CHECKPOINT_INTERVAL must be a positive integer" >&2; exit 2;
}
[[ "$rehearsal_depth" =~ ^([0-9]|10)$ ]] || {
  echo "NUMI_CROW_REHEARSAL_DEPTH must be an integer in 0...10" >&2; exit 2;
}
if [ -n "$rehearsal_minimum_band" ]; then
  [[ "$rehearsal_minimum_band" =~ ^([0-9]|10)$ ]] || {
    echo "NUMI_CROW_REHEARSAL_MINIMUM_BAND must be an integer in 0...10" >&2
    exit 2
  }
fi
[ -x "$build/bin/metalrobo_task_train" ] && [ -x "$mlx" ] || {
  echo "Crow curriculum requires a built trainer and MLX Python" >&2; exit 2;
}
if ! "$mlx" -c \
  'import math,sys; value=float(sys.argv[1]); enabled=int(sys.argv[2]); sys.exit(0 if math.isfinite(value) and 0.0 <= value <= 1.0 and (enabled or value == 0.0) else 1)' \
  "$teacher_student_authority" "$teacher_distillation"; then
  echo "Crow teacher student authority must be in [0,1] and zero when teacher distillation is disabled" >&2
  exit 2
fi
if [ -n "$difficulty_sampling_exponent" ] && ! "$mlx" -c \
  'import math,sys; value=float(sys.argv[1]); sys.exit(0 if math.isfinite(value) and value > 0.0 else 1)' \
  "$difficulty_sampling_exponent"; then
  echo "NUMI_CROW_DIFFICULTY_SAMPLING_EXPONENT must be finite and positive" >&2
  exit 2
fi
if [ -n "$learning_rate" ] && ! "$mlx" -c \
  'import math,sys; value=float(sys.argv[1]); sys.exit(0 if math.isfinite(value) and value > 0.0 else 1)' \
  "$learning_rate"; then
  echo "NUMI_CROW_LEARNING_RATE must be finite and positive" >&2
  exit 2
fi
if [ -n "$initial_log_standard_deviation" ] && ! "$mlx" -c \
  'import math,sys; value=float(sys.argv[1]); sys.exit(0 if math.isfinite(value) else 1)' \
  "$initial_log_standard_deviation"; then
  echo "NUMI_CROW_INITIAL_LOG_STANDARD_DEVIATION must be finite" >&2
  exit 2
fi
if [ -n "$retention_policy" ]; then
  [ -s "$retention_policy" ] || {
    echo "NUMI_CROW_RETENTION_POLICY must name a readable PolicyPack" >&2
    exit 2
  }
  [ "$teacher_distillation" -eq 0 ] || {
    echo "Crow retention policy cannot be combined with teacher distillation" >&2
    exit 2
  }
  if ! "$mlx" -c \
    'import math,sys; value=float(sys.argv[1]); sys.exit(0 if math.isfinite(value) and value > 0.0 else 1)' \
    "$retention_coefficient"; then
    echo "NUMI_CROW_RETENTION_COEFFICIENT must be finite and positive" >&2
    exit 2
  fi
  if [ -n "$retention_maximum_band" ] && \
     ! [[ "$retention_maximum_band" =~ ^([0-9]|10)$ ]]; then
    echo "NUMI_CROW_RETENTION_MAXIMUM_BAND must be an integer in 0...10" >&2
    exit 2
  fi
  if [ "$retention_protected_actor_only" -eq 1 ] && \
     [ -z "$retention_maximum_band" ]; then
    echo "protected actor-only retention requires a maximum band" >&2
    exit 2
  fi
elif [ -n "$retention_maximum_band" ]; then
  echo "NUMI_CROW_RETENTION_MAXIMUM_BAND requires a retention policy" >&2
  exit 2
elif [ "$retention_protected_actor_only" -eq 1 ]; then
  echo "protected actor-only retention requires a retention policy" >&2
  exit 2
fi
if [ "$parent_mode" = actor-transfer ]; then
  [ -n "$parent_policy" ] && [ -s "$parent_policy" ] && \
    [ -z "$parent_state" ] || {
    echo "Crow actor transfer requires a source policy and no learner state" >&2
    exit 2
  }
  if [ "$course" = sensor-fast ] && [ "$start_band" -ne 0 ]; then
    echo "sensor-fast actor transfer requires start band 0" >&2
    exit 2
  fi
fi
if [ "$parent_mode" = resume ]; then
  if [ -n "$parent_policy" ] || [ -n "$parent_state" ]; then
    [ -n "$parent_policy" ] && [ -s "$parent_policy" ] && \
      [ -n "$parent_state" ] && [ -s "$parent_state" ] || {
      echo "Crow PPO resume requires both a full PolicyPack and learner state" >&2
      exit 2
    }
  fi
fi
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
    if [ "$parent_mode" = resume ] && [ -n "$parent_state" ]; then
      cp "$parent_state" "$run/learner.safetensors"
    fi
    common=(
      --envs "$environments" --steps "$steps" --updates "$updates"
      --chunk "$chunk" --seed "$seed" --learner-seed "$seed"
      --checkpoint-directory "$run/checkpoints"
      --checkpoint-interval "$checkpoint_interval"
      --verbose
    )
    if [ -n "$difficulty_sampling_exponent" ]; then
      common+=(--difficulty-sampling-exponent "$difficulty_sampling_exponent")
    fi
    if [ -n "$learning_rate" ]; then
      common+=(
        --learning-rate "$learning_rate"
        --minimum-learning-rate "$learning_rate"
        --maximum-learning-rate "$learning_rate"
        --fixed-learning-rate
      )
    fi
    if [ -n "$initial_log_standard_deviation" ]; then
      common+=(
        --initial-log-standard-deviation
        "$initial_log_standard_deviation"
      )
    fi
    if [ -n "$retention_policy" ]; then
      common+=(
        --retention-policy-pack "$retention_policy"
        --imagination-distillation-coefficient "$retention_coefficient"
      )
      if [ -n "$retention_maximum_band" ]; then
        common+=(
          --retention-maximum-difficulty-band "$retention_maximum_band"
        )
      fi
      if [ "$retention_protected_actor_only" -eq 1 ]; then
        common+=(--retention-protected-actor-only)
      fi
    fi
    if [ -n "$visual_config" ]; then
      common+=(--visual-observation-config "$visual_config")
    fi
    if [ "$teacher_distillation" -eq 1 ]; then
      common+=(
        --birdflow-journey-teacher
        --birdflow-journey-student-authority "$teacher_student_authority"
      )
    fi
    if [ "$parent_mode" = actor-transfer ]; then
      common+=(
        --initialize-actor-policy-pack "$parent_policy"
        --initialize-actor-fresh-critic
      )
    elif [ -n "$parent_policy" ]; then
      common+=(--policy-pack "$parent_policy")
    fi
    echo "launching neural Crow $course band $band ($milestone), retry $retry"
    if [ -n "$rehearsal_minimum_band" ]; then
      training_minimum_band=$rehearsal_minimum_band
      [ "$training_minimum_band" -le "$band" ] || \
        training_minimum_band=$band
    else
      training_minimum_band=$((band - rehearsal_depth))
      [ "$training_minimum_band" -ge 0 ] || training_minimum_band=0
    fi
    explicit_train=0
    if [ "$parent_mode" = actor-transfer ] || \
       [ "$teacher_distillation" -eq 0 ] || \
       { [ "$training_minimum_band" -lt "$band" ] && \
         [ "$parent_mode" = resume ] && [ -n "$parent_policy" ]; }; then
      explicit_train=1
    fi
    if [ "$explicit_train" -eq 1 ]; then
      launch=(
        "$root/tools/numi" train --birdflow-american-crow-journey
        --birdflow-journey-variant "$journey_variant"
        --minimum-difficulty-band "$training_minimum_band"
        --maximum-difficulty-band "$band"
      )
      if [ "$training_minimum_band" -lt "$band" ]; then
        echo "rehearsing protected Crow bands $training_minimum_band-$band"
      fi
    else
      launch=(
        "$root/tools/numi" crow journey train
        --variant "$journey_variant" --milestone "$milestone"
      )
    fi
    if NUMI_LAB_ROOT="$root" NUMI_BUILD_DIR="$build" \
      NUMI_MLX_PYTHON="$mlx" NUMI_RUN_DIR="$run" \
      NUMI_SELECTION_ENVS="$selection_environments" \
      NUMI_SELECTION_SEED="$selection_seed" \
      "${launch[@]}" "${common[@]}"; then
      :
    elif [ ! -s "$selection" ]; then
      echo "Crow band $band failed without a durable selection" >&2
      exit 5
    fi
  fi

  if ! "$mlx" -c \
    'import json,sys; data=json.load(open(sys.argv[1])); sys.exit(0 if not data.get("selection_error") else 1)' \
    "$selection"; then
    echo "Crow band $band selection/runtime failed; retained evidence: $selection" >&2
    exit 5
  fi

  if "$mlx" -c \
    'import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get("candidate_advanced_deployment", False) else 1)' \
    "$selection"; then
    selected_label=$("$mlx" -c \
      'import json,sys; print(json.load(open(sys.argv[1])).get("selected_candidate_label") or "candidate")' \
      "$selection")
    selected_revision=${selected_label#candidate-}
    checkpoint_policy="$run/checkpoints/revision-$selected_revision.policypack"
    checkpoint_state="$run/checkpoints/revision-$selected_revision.safetensors"
    if [ "$selected_revision" != "$selected_label" ] && \
      [ -s "$checkpoint_policy" ] && [ -s "$checkpoint_state" ]; then
      parent_policy="$checkpoint_policy"
      parent_state="$checkpoint_state"
    else
      parent_policy="$run/candidate.policypack"
      parent_state="$run/learner.safetensors"
    fi
    parent_mode=resume
    deployment_policy="$run/deployment.policypack"
    [ -s "$deployment_policy" ] && [ -s "$parent_policy" ] && \
      [ -s "$parent_state" ] || {
      echo "selected Crow deployment/training parent is incomplete for $selected_label" >&2
      exit 5
    }
    printf '{"schema":"numi.crow-curriculum-progress.v1","variant":"%s","course":"%s","completed_band":%s,"milestone":"%s","deployment":"%s","training_parent":"%s","learner_state":"%s","selection":"%s"}\n' \
      "$journey_variant" "$course" "$band" "$milestone" \
      "$deployment_policy" "$parent_policy" "$parent_state" "$selection" \
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

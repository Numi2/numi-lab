#!/bin/bash

# Durable, low-frequency supervisor for the authored G1 adult curriculum.
#
# The first adult band may be an actor-only migration from the developmental
# curriculum. Once an adult run has produced a full stochastic PolicyPack and
# learner state, later promoted bands continue from both artifacts. The adult
# task keeps one observation/action contract across difficulty bands, so this
# preserves the critic and Adam state instead of repeatedly relearning value
# estimates from scratch.

set -eu

numi_curriculum_root=${NUMI_CURRICULUM_ROOT:?set NUMI_CURRICULUM_ROOT}
numi_curriculum_build=${NUMI_CURRICULUM_BUILD:?set NUMI_CURRICULUM_BUILD}
numi_curriculum_mlx=${NUMI_CURRICULUM_MLX:?set NUMI_CURRICULUM_MLX}
numi_curriculum_runs_root=${NUMI_CURRICULUM_RUNS_ROOT:?set NUMI_CURRICULUM_RUNS_ROOT}
numi_curriculum_start_run=${NUMI_CURRICULUM_START_RUN:?set NUMI_CURRICULUM_START_RUN}
numi_curriculum_parent_policy=${NUMI_CURRICULUM_PARENT_POLICY:?set NUMI_CURRICULUM_PARENT_POLICY}
numi_curriculum_parent_state=${NUMI_CURRICULUM_PARENT_STATE:-}
numi_curriculum_start_band=${NUMI_CURRICULUM_START_BAND:-2}
numi_curriculum_max_band=${NUMI_CURRICULUM_MAX_BAND:-10}
numi_curriculum_seed_base=${NUMI_CURRICULUM_SEED_BASE:-2650444043}
numi_curriculum_selection_envs=${NUMI_CURRICULUM_SELECTION_ENVS:-512}
numi_curriculum_selection_seed=${NUMI_CURRICULUM_SELECTION_SEED:-2650443581}
numi_curriculum_learning_rate=${NUMI_CURRICULUM_LEARNING_RATE:-1e-5}
numi_curriculum_selection_timeout_seconds=${NUMI_CURRICULUM_SELECTION_TIMEOUT_SECONDS:-7200}

numi_curriculum_band=$numi_curriculum_start_band
numi_curriculum_retry=0

numi_curriculum_run_for() {
    numi_curriculum_previous=$((numi_curriculum_band - 1))
    numi_curriculum_seed=$((
        numi_curriculum_seed_base + numi_curriculum_band +
        numi_curriculum_retry * 1000
    ))
    if [ "$numi_curriculum_band" -eq "$numi_curriculum_start_band" ] &&
       [ "$numi_curriculum_retry" -eq 0 ]; then
        numi_curriculum_run=$numi_curriculum_start_run
    elif [ "$numi_curriculum_retry" -eq 0 ]; then
        numi_curriculum_run="${numi_curriculum_runs_root}/g1-adult-band${numi_curriculum_band}-after-band${numi_curriculum_previous}-1e-5-seed${numi_curriculum_seed}-r1"
    else
        numi_curriculum_run="${numi_curriculum_runs_root}/g1-adult-band${numi_curriculum_band}-after-band${numi_curriculum_previous}-1e-5-seed${numi_curriculum_seed}-retry${numi_curriculum_retry}"
    fi
}

numi_curriculum_launch() {
    numi_curriculum_previous=$((numi_curriculum_band - 1))
    numi_curriculum_seed=$((
        numi_curriculum_seed_base + numi_curriculum_band +
        numi_curriculum_retry * 1000
    ))

    # A copied learner state is only paired with its exact full PolicyPack.
    # Rejected candidates never become parents: retries use the run's initial
    # full pack but deliberately start a fresh optimizer state.
    if [ -n "$numi_curriculum_parent_state" ] &&
       [ -s "$numi_curriculum_parent_state" ] &&
       [ -s "$numi_curriculum_parent_policy" ]; then
        mkdir -p "$numi_curriculum_run"
        cp "$numi_curriculum_parent_state" \
            "$numi_curriculum_run/learner.safetensors"
        NUMI_LAB_ROOT="$numi_curriculum_root" \
        NUMI_BUILD_DIR="$numi_curriculum_build" \
        NUMI_MLX_PYTHON="$numi_curriculum_mlx" \
        NUMI_SELECTION_ENVS="$numi_curriculum_selection_envs" \
        NUMI_SELECTION_SEED="$numi_curriculum_selection_seed" \
        NUMI_RUN_DIR="$numi_curriculum_run" \
        "$numi_curriculum_root/tools/numi" train \
            --scene ground \
            --task adult-locomotion \
            --minimum-difficulty-band "$numi_curriculum_previous" \
            --maximum-difficulty-band "$numi_curriculum_band" \
            --envs 4096 \
            --steps 32 \
            --updates 250 \
            --chunk 16 \
            --policy-pack "$numi_curriculum_parent_policy" \
            --learning-rate "$numi_curriculum_learning_rate" \
            --minimum-learning-rate "$numi_curriculum_learning_rate" \
            --maximum-learning-rate "$numi_curriculum_learning_rate" \
            --fixed-learning-rate \
            --initial-log-standard-deviation -1.6094379 \
            --seed "$numi_curriculum_seed" \
            --learner-seed "$numi_curriculum_seed" \
            --checkpoint-directory "$numi_curriculum_run/checkpoints" \
            --checkpoint-interval 20 \
            --verbose
    else
        NUMI_LAB_ROOT="$numi_curriculum_root" \
        NUMI_BUILD_DIR="$numi_curriculum_build" \
        NUMI_MLX_PYTHON="$numi_curriculum_mlx" \
        NUMI_SELECTION_ENVS="$numi_curriculum_selection_envs" \
        NUMI_SELECTION_SEED="$numi_curriculum_selection_seed" \
        NUMI_RUN_DIR="$numi_curriculum_run" \
        "$numi_curriculum_root/tools/numi" train \
            --scene ground \
            --task adult-locomotion \
            --minimum-difficulty-band "$numi_curriculum_previous" \
            --maximum-difficulty-band "$numi_curriculum_band" \
            --envs 4096 \
            --steps 32 \
            --updates 250 \
            --chunk 16 \
            --initialize-policy \
                "g1-adult-band${numi_curriculum_band}-parent" \
            --initialize-actor-policy-pack "$numi_curriculum_parent_policy" \
            --initialize-actor-fresh-critic \
            --learning-rate "$numi_curriculum_learning_rate" \
            --minimum-learning-rate "$numi_curriculum_learning_rate" \
            --maximum-learning-rate "$numi_curriculum_learning_rate" \
            --fixed-learning-rate \
            --initial-log-standard-deviation -1.6094379 \
            --seed "$numi_curriculum_seed" \
            --learner-seed "$numi_curriculum_seed" \
            --checkpoint-directory "$numi_curriculum_run/checkpoints" \
            --checkpoint-interval 20 \
            --verbose
    fi
}

while [ "$numi_curriculum_band" -le "$numi_curriculum_max_band" ]; do
    numi_curriculum_run_for
    numi_curriculum_selection="$numi_curriculum_run/selection/selection.json"

    if [ ! -s "$numi_curriculum_selection" ]; then
        if [ -e "$numi_curriculum_run" ]; then
            echo "waiting for existing run selection: $numi_curriculum_run"
            numi_curriculum_wait_started=$(date +%s)
            while [ ! -s "$numi_curriculum_selection" ]; do
                if [ "$numi_curriculum_selection_timeout_seconds" -gt 0 ]; then
                    numi_curriculum_wait_now=$(date +%s)
                    if [ $((numi_curriculum_wait_now - numi_curriculum_wait_started)) -ge "$numi_curriculum_selection_timeout_seconds" ]; then
                        echo "timed out waiting for adult band $numi_curriculum_band selection: $numi_curriculum_run" >&2
                        exit 7
                    fi
                fi
                sleep 30
            done
        else
            echo "launching adult band $numi_curriculum_band retry $numi_curriculum_retry from parent=$numi_curriculum_parent_policy at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
            # The train overlay durably records an incumbent-retaining
            # selection when the selector fails. Run the launch in an if
            # context so set -e cannot discard that decision before the
            # curriculum retry logic consumes it.
            if numi_curriculum_launch; then
                :
            else
                numi_curriculum_launch_status=$?
                if [ ! -s "$numi_curriculum_selection" ]; then
                    echo "adult band $numi_curriculum_band retry $numi_curriculum_retry failed without a selection decision" >&2
                    exit "$numi_curriculum_launch_status"
                fi
                echo "adult band $numi_curriculum_band retry $numi_curriculum_retry produced an incumbent fallback selection"
            fi
            echo "adult band $numi_curriculum_band retry $numi_curriculum_retry exited at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
        fi
    fi

    [ -s "$numi_curriculum_selection" ]
    if "$numi_curriculum_mlx" -c \
        'import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get("candidate_advanced_deployment", False) else 1)' \
        "$numi_curriculum_selection"; then
        echo "adult band $numi_curriculum_band promoted candidate from $numi_curriculum_run"
        numi_curriculum_parent_policy="$numi_curriculum_run/candidate.policypack"
        numi_curriculum_parent_state="$numi_curriculum_run/learner.safetensors"
        numi_curriculum_band=$((numi_curriculum_band + 1))
        numi_curriculum_retry=0
    else
        numi_curriculum_retry=$((numi_curriculum_retry + 1))
        numi_curriculum_parent_policy="$numi_curriculum_run/initial.policypack"
        numi_curriculum_parent_state=
        [ "$numi_curriculum_retry" -le 3 ] || {
            echo "adult band $numi_curriculum_band failed promotion after three retries" >&2
            exit 6
        }
        echo "adult band $numi_curriculum_band retained incumbent; retrying same band with seed $((numi_curriculum_seed_base + numi_curriculum_band + numi_curriculum_retry * 1000))"
    fi
done

echo "adult bands $numi_curriculum_start_band-$numi_curriculum_max_band completed at $(date -u +%Y-%m-%dT%H:%M:%SZ)"

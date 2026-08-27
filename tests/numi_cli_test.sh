#!/bin/sh

set -eu

numi_repo=$(CDPATH= cd "$(dirname "$0")/.." && pwd -P)
numi_temp=$(mktemp -d "${TMPDIR:-/tmp}/numi-cli-test.XXXXXX")
trap 'rm -rf "$numi_temp"' EXIT HUP INT TERM

mkdir -p "$numi_temp/workspace/.numi/commands" "$numi_temp/extra"
mkdir -p \
    "$numi_temp/fake-build/bin" \
    "$numi_temp/fake-build/lib" \
    "$numi_temp/fake-build/shaders" \
    "$numi_temp/runs"

printf '%s\n' '#!/bin/sh' \
    'if [ "${1:-}" = "--numi-describe" ]; then printf "Workspace override.\n"; exit 0; fi' \
    'printf "workspace:%s\n" "$*"' \
    > "$numi_temp/workspace/.numi/commands/train"
chmod +x "$numi_temp/workspace/.numi/commands/train"

printf '%s\n' '#!/bin/sh' \
    'if [ "${1:-}" = "--numi-describe" ]; then printf "Extra capability.\n"; exit 0; fi' \
    'printf "extra:%s\n" "$*"' \
    > "$numi_temp/extra/custom"
chmod +x "$numi_temp/extra/custom"

printf '%s\n' '#!/bin/sh' \
    'incumbent=' \
    'candidate=' \
    'take=' \
    'for argument do' \
    '  if [ "$take" = incumbent ]; then incumbent=$argument; take=; continue; fi' \
    '  if [ "$take" = candidate ]; then candidate=$argument; take=; continue; fi' \
    '  [ "$argument" = --incumbent-policy-pack ] && take=incumbent' \
    '  [ "$argument" = --deployment-policy-pack ] && take=candidate' \
    'done' \
    'mkdir -p "$(dirname "$incumbent")"' \
    'printf "incumbent\n" > "$incumbent"' \
    'printf "candidate\n" > "$candidate"' \
    'printf "{\"status\":\"trained\"}\n"' \
    > "$numi_temp/fake-build/bin/metalrobo_task_train"
chmod +x "$numi_temp/fake-build/bin/metalrobo_task_train"

printf '%s\n' '#!/bin/sh' \
    'policy=' \
    'take=0' \
    'for argument do' \
    '  if [ "$take" -eq 1 ]; then policy=$argument; take=0; continue; fi' \
    '  [ "$argument" = --policy-pack ] && take=1' \
    'done' \
    'case "$policy" in' \
    '  *candidate*) printf "{\"task\":\"velocity\",\"termination_count\":0,\"termination_count_by_environment\":[0],\"failed_environment_steps\":0,\"mean_tracking_score\":0.5,\"mean_tilt\":0.1}\n" ;;' \
    '  *incumbent*) printf "{\"task\":\"velocity\",\"termination_count\":1,\"termination_count_by_environment\":[1],\"failed_environment_steps\":0,\"mean_tracking_score\":0.4,\"mean_tilt\":0.2}\n" ;;' \
    '  *) printf "fake-evaluate\n" ;;' \
    'esac' \
    > "$numi_temp/fake-build/bin/metalrobo_task_rollout"
chmod +x "$numi_temp/fake-build/bin/metalrobo_task_rollout"
printf 'fake native library\n' > "$numi_temp/fake-build/lib/libmetalrobo.dylib"
printf 'fake metal library\n' > "$numi_temp/fake-build/shaders/MetalRobo.metallib"

numi_version=$($numi_repo/tools/numi version)
[ "$numi_version" = "0.4.0" ]

ln -s "$numi_repo/tools/numi" "$numi_temp/numi"
numi_linked_version=$($numi_temp/numi version)
[ "$numi_linked_version" = "0.4.0" ]

numi_codex_description=$($numi_repo/tools/numi codex --numi-describe)
[ "$numi_codex_description" = "Install or inspect Numi Lab inside Codex." ]

numi_foundation_description=$($numi_repo/tools/numi foundation --numi-describe)
[ "$numi_foundation_description" = \
    "Inspect or run a foundation model as a fingerprinted action-chunk proposer." ]

numi_context=$(
    cd "$numi_temp/workspace"
    NUMI_LAB_ROOT=$numi_repo \
    NUMI_COMMAND_PATH=$numi_temp/extra \
        "$numi_repo/tools/numi" context
)
printf '%s\n' "$numi_context" | grep 'Workspace override.' >/dev/null
printf '%s\n' "$numi_context" | grep 'Extra capability.' >/dev/null

numi_train=$(
    cd "$numi_temp/workspace"
    NUMI_LAB_ROOT=$numi_repo \
        "$numi_repo/tools/numi" train alpha beta
)
[ "$numi_train" = "workspace:alpha beta" ]

numi_custom=$(
    cd "$numi_temp/workspace"
    NUMI_LAB_ROOT=$numi_repo \
    NUMI_COMMAND_PATH=$numi_temp/extra \
        "$numi_repo/tools/numi" run custom gamma
)
[ "$numi_custom" = "extra:gamma" ]

numi_train_run=$numi_temp/runs/train
numi_train_output=$(
    cd "$numi_repo"
    NUMI_BUILD_DIR=$numi_temp/fake-build \
    NUMI_RUN_DIR=$numi_train_run \
        "$numi_repo/tools/numi" train \
            --interaction-pack sample.interactionpack \
            --interaction-clip sample-clip \
            --updates 1
)
printf '%s\n' "$numi_train_output" | grep '"status":"trained"' >/dev/null
printf '%s\n' "$numi_train_output" | grep '"selected": "candidate"' >/dev/null
grep -- '--initialize-policy' "$numi_train_run/arguments.txt" >/dev/null
grep -- '--updated-policy-pack' "$numi_train_run/arguments.txt" >/dev/null
grep -- '--mlx-python' "$numi_train_run/arguments.txt" >/dev/null
grep -- 'sample.interactionpack' "$numi_train_run/arguments.txt" >/dev/null
grep -- 'sample-clip' "$numi_train_run/arguments.txt" >/dev/null
test -s "$numi_train_run/revision.txt"
test -s "$numi_train_run/runtime.sha256"
test -s "$numi_train_run/artifacts.sha256"
test -s "$numi_train_run/candidate.deployment.policypack"
test -s "$numi_train_run/incumbent.deployment.policypack"
test -s "$numi_train_run/deployment.policypack"
cmp "$numi_train_run/candidate.deployment.policypack" \
    "$numi_train_run/deployment.policypack"
test -s "$numi_train_run/selection/incumbent.evidence.json"
test -s "$numi_train_run/selection/candidate.evidence.json"
test -s "$numi_train_run/selection.log"
grep '"candidate_retained": true' \
    "$numi_train_run/selection/selection.json" >/dev/null

# A remote supervisor may capture the command's output directly into the
# durable run logs. That must retain native records exactly once rather than
# feeding a live log back into `cat` until the training volume is full.
numi_self_capture_run=$numi_temp/runs/self-capture
mkdir -p "$numi_self_capture_run"
(
    cd "$numi_repo"
    NUMI_BUILD_DIR=$numi_temp/fake-build \
    NUMI_RUN_DIR=$numi_self_capture_run \
        "$numi_repo/tools/numi" train --updates 1 \
            > "$numi_self_capture_run/stdout.log" \
            2> "$numi_self_capture_run/stderr.log"
)
grep '"status":"trained"' "$numi_self_capture_run/stdout.log" >/dev/null
test "$(wc -c < "$numi_self_capture_run/stdout.log" | tr -d ' ')" -lt 65536
test "$(wc -c < "$numi_self_capture_run/stderr.log" | tr -d ' ')" -lt 65536

crow_actor_run=$numi_temp/runs/crow-actor-transfer
crow_actor_source=$numi_temp/crow-stage-one.policypack
printf 'selected stage-one actor\n' > "$crow_actor_source"
crow_actor_output=$(
    cd "$numi_repo"
    NUMI_BUILD_DIR=$numi_temp/fake-build \
    NUMI_RUN_DIR=$crow_actor_run \
        "$numi_repo/tools/numi" crow train \
            --initialize-actor-policy-pack "$crow_actor_source" \
            --initialize-actor-fresh-critic \
            --updates 1
)
printf '%s\n' "$crow_actor_output" | grep '"status":"trained"' >/dev/null
grep -- '--birdflow-american-crow' "$crow_actor_run/arguments.txt" >/dev/null
grep -- '--initialize-policy' "$crow_actor_run/arguments.txt" >/dev/null
grep -- '--initialize-actor-policy-pack' "$crow_actor_run/arguments.txt" >/dev/null
if grep -- '--zero-actor-output' "$crow_actor_run/arguments.txt" >/dev/null; then
    printf '%s\n' 'crow actor transfer was unexpectedly zeroed' >&2
    exit 1
fi

if (
    cd "$numi_repo"
    NUMI_BUILD_DIR=$numi_temp/fake-build \
        "$numi_repo/tools/numi" crow train \
            --initialize-actor-policy-pack "$crow_actor_source" \
            --zero-actor-output \
            --updates 1
) > "$numi_temp/crow-invalid-transfer.log" 2>&1; then
    printf '%s\n' 'crow accepted a contradictory actor-transfer request' >&2
    exit 1
fi
grep -- \
    '--zero-actor-output cannot be combined with --initialize-actor-policy-pack' \
    "$numi_temp/crow-invalid-transfer.log" >/dev/null

crow_journey_evaluate_run=$numi_temp/runs/crow-journey-evaluate
crow_journey_output=$(
    cd "$numi_repo"
    NUMI_BUILD_DIR=$numi_temp/fake-build \
    NUMI_RUN_DIR=$crow_journey_evaluate_run \
        "$numi_repo/tools/numi" crow journey evaluate --zero-actions
)
[ "$crow_journey_output" = "fake-evaluate" ]
grep -- '--birdflow-american-crow-journey' \
    "$crow_journey_evaluate_run/arguments.txt" >/dev/null
grep -- '--birdflow-journey-variant' \
    "$crow_journey_evaluate_run/arguments.txt" >/dev/null
grep -- '^v7-hierarchical$' \
    "$crow_journey_evaluate_run/arguments.txt" >/dev/null
grep -- '--minimum-difficulty-band' \
    "$crow_journey_evaluate_run/arguments.txt" >/dev/null
grep -- '^4$' "$crow_journey_evaluate_run/arguments.txt" >/dev/null

crow_journey_train_run=$numi_temp/runs/crow-journey-train
crow_journey_train_output=$(
    cd "$numi_repo"
    NUMI_BUILD_DIR=$numi_temp/fake-build \
    NUMI_RUN_DIR=$crow_journey_train_run \
        "$numi_repo/tools/numi" crow journey train --updates 1
)
printf '%s\n' "$crow_journey_train_output" | grep '"status":"trained"' >/dev/null
grep -- '--birdflow-american-crow-journey' \
    "$crow_journey_train_run/arguments.txt" >/dev/null
grep -- '--birdflow-journey-teacher' \
    "$crow_journey_train_run/arguments.txt" >/dev/null
grep -- '--maximum-difficulty-band' \
    "$crow_journey_train_run/arguments.txt" >/dev/null
grep -- '^4$' "$crow_journey_train_run/arguments.txt" >/dev/null

crow_journey_transfer_run=$numi_temp/runs/crow-journey-transfer
(
    cd "$numi_repo"
    NUMI_BUILD_DIR=$numi_temp/fake-build \
    NUMI_RUN_DIR=$crow_journey_transfer_run \
        "$numi_repo/tools/numi" crow journey train \
            --milestone walking \
            --initialize-actor-policy-pack "$crow_actor_source" \
            --updates 1 >/dev/null
)
grep -- '--initialize-actor-policy-pack' \
    "$crow_journey_transfer_run/arguments.txt" >/dev/null
grep -- '^1$' "$crow_journey_transfer_run/arguments.txt" >/dev/null
if grep -- '--zero-actor-output' \
    "$crow_journey_transfer_run/arguments.txt" >/dev/null; then
    printf '%s\n' 'crow journey actor transfer was unexpectedly zeroed' >&2
    exit 1
fi

crow_full_journey_run=$numi_temp/runs/crow-full-journey-evaluate
crow_full_journey_output=$(
    cd "$numi_repo"
    NUMI_BUILD_DIR=$numi_temp/fake-build \
    NUMI_RUN_DIR=$crow_full_journey_run \
        "$numi_repo/tools/numi" crow journey evaluate \
            --milestone full-journey --zero-actions
)
[ "$crow_full_journey_output" = "fake-evaluate" ]
grep -- '--minimum-difficulty-band' \
    "$crow_full_journey_run/arguments.txt" >/dev/null
grep -- '^10$' "$crow_full_journey_run/arguments.txt" >/dev/null

crow_neural_journey_run=$numi_temp/runs/crow-neural-journey-evaluate
crow_neural_journey_output=$(
    cd "$numi_repo"
    NUMI_BUILD_DIR=$numi_temp/fake-build \
    NUMI_RUN_DIR=$crow_neural_journey_run \
        "$numi_repo/tools/numi" crow journey evaluate \
            --variant v8-neural --milestone approach --zero-actions
)
[ "$crow_neural_journey_output" = "fake-evaluate" ]
grep -- '^v8-neural$' \
    "$crow_neural_journey_run/arguments.txt" >/dev/null
grep -- '^7$' "$crow_neural_journey_run/arguments.txt" >/dev/null

crow_visual_journey_run=$numi_temp/runs/crow-visual-journey-evaluate
crow_visual_journey_output=$(
    cd "$numi_repo"
    NUMI_BUILD_DIR=$numi_temp/fake-build \
    NUMI_RUN_DIR=$crow_visual_journey_run \
        "$numi_repo/tools/numi" crow journey evaluate \
            --variant v9-visual-neural --milestone approach --zero-actions
)
[ "$crow_visual_journey_output" = "fake-evaluate" ]
grep -- '^v9-visual-neural$' \
    "$crow_visual_journey_run/arguments.txt" >/dev/null
grep -- 'crow-journey.sensor-fast.visual-observation.json$' \
    "$crow_visual_journey_run/arguments.txt" >/dev/null

if (
    cd "$numi_repo"
    NUMI_BUILD_DIR=$numi_temp/fake-build \
        "$numi_repo/tools/numi" crow journey evaluate \
            --variant unknown --zero-actions
) > "$numi_temp/crow-invalid-variant.log" 2>&1; then
    printf '%s\n' 'crow accepted an unknown journey variant' >&2
    exit 1
fi
grep -- 'journey variant requires v7-hierarchical, v8-neural, or v9-visual-neural' \
    "$numi_temp/crow-invalid-variant.log" >/dev/null

if "$numi_repo/tools/numi" crow journey window > /dev/null 2>&1; then
    printf '%s\n' 'crow journey window accepted a missing policy' >&2
    exit 1
fi

# The durable sensor-fast supervisor crosses the v8/v9 observation ABI only
# through actor initialization. Treating the v8 pack as a PPO resume would
# either fail late after launch or silently weaken the transfer contract.
crow_curriculum_root=$numi_temp/crow-curriculum-root
crow_curriculum_build=$numi_temp/crow-curriculum-build
crow_curriculum_runs=$numi_temp/runs/crow-sensor-fast-curriculum
mkdir -p "$crow_curriculum_root/tools" "$crow_curriculum_build/bin"
printf '%s\n' '#!/bin/sh' 'exit 0' \
    > "$crow_curriculum_build/bin/metalrobo_task_train"
chmod +x "$crow_curriculum_build/bin/metalrobo_task_train"
printf '%s\n' '#!/bin/sh' \
    'run=${NUMI_RUN_DIR:?}' \
    'mkdir -p "$run/selection"' \
    'printf "%s\n" "$@" > "$run/arguments.txt"' \
    'printf "initial visual policy\n" > "$run/initial.policypack"' \
    'printf "candidate training policy\n" > "$run/candidate.policypack"' \
    'printf "selected deployment\n" > "$run/deployment.policypack"' \
    'printf "learner state\n" > "$run/learner.safetensors"' \
    'printf "{\"candidate_advanced_deployment\":true,\"selected_candidate_label\":\"candidate\"}\n" > "$run/selection/selection.json"' \
    > "$crow_curriculum_root/tools/numi"
chmod +x "$crow_curriculum_root/tools/numi"
NUMI_CROW_CURRICULUM_ROOT=$crow_curriculum_root \
NUMI_CROW_CURRICULUM_BUILD=$crow_curriculum_build \
NUMI_CROW_CURRICULUM_MLX=/usr/bin/python3 \
NUMI_CROW_CURRICULUM_RUNS=$crow_curriculum_runs \
NUMI_CROW_COURSE=sensor-fast \
NUMI_CROW_PARENT_POLICY=$crow_actor_source \
NUMI_CROW_START_BAND=0 \
NUMI_CROW_MAXIMUM_BAND=0 \
    "$numi_repo/tools/crow_journey_curriculum_supervisor.sh" >/dev/null
crow_sensor_run=$(find "$crow_curriculum_runs" -maxdepth 1 -type d \
    -name 'v9-visual-neural-band0-*' -print | head -1)
test -n "$crow_sensor_run"
grep -- '--initialize-actor-policy-pack' \
    "$crow_sensor_run/arguments.txt" >/dev/null
grep -- '--initialize-actor-fresh-critic' \
    "$crow_sensor_run/arguments.txt" >/dev/null
grep -- '--birdflow-journey-student-authority' \
    "$crow_sensor_run/arguments.txt" >/dev/null
grep -- '^0.25$' "$crow_sensor_run/arguments.txt" >/dev/null
grep -- '--checkpoint-interval' "$crow_sensor_run/arguments.txt" >/dev/null
grep -- '^50$' "$crow_sensor_run/arguments.txt" >/dev/null
if grep -- '--policy-pack' "$crow_sensor_run/arguments.txt" >/dev/null; then
    printf '%s\n' 'sensor-fast supervisor attempted cross-ABI PPO resume' >&2
    exit 1
fi
test -s "$crow_curriculum_runs/progress.json"

crow_resume_runs=$numi_temp/runs/crow-state-resume-curriculum
NUMI_CROW_CURRICULUM_ROOT=$crow_curriculum_root \
NUMI_CROW_CURRICULUM_BUILD=$crow_curriculum_build \
NUMI_CROW_CURRICULUM_MLX=/usr/bin/python3 \
NUMI_CROW_CURRICULUM_RUNS=$crow_resume_runs \
NUMI_CROW_COURSE=state \
NUMI_CROW_PARENT_POLICY=$crow_sensor_run/candidate.policypack \
NUMI_CROW_PARENT_STATE=$crow_sensor_run/learner.safetensors \
NUMI_CROW_START_BAND=1 \
NUMI_CROW_MAXIMUM_BAND=1 \
NUMI_CROW_REHEARSAL_MINIMUM_BAND=0 \
NUMI_CROW_DIFFICULTY_SAMPLING_EXPONENT=0.25 \
    "$numi_repo/tools/crow_journey_curriculum_supervisor.sh" >/dev/null
crow_resume_run=$(find "$crow_resume_runs" -maxdepth 1 -type d \
    -name 'v8-neural-band1-*' -print | head -1)
test -n "$crow_resume_run"
grep -- '--policy-pack' "$crow_resume_run/arguments.txt" >/dev/null
grep -- '--birdflow-journey-teacher' \
    "$crow_resume_run/arguments.txt" >/dev/null
grep -- '--birdflow-journey-student-authority' \
    "$crow_resume_run/arguments.txt" >/dev/null
grep -- '--minimum-difficulty-band' "$crow_resume_run/arguments.txt" >/dev/null
grep -- '^0$' "$crow_resume_run/arguments.txt" >/dev/null
grep -- '--maximum-difficulty-band' "$crow_resume_run/arguments.txt" >/dev/null
grep -- '^1$' "$crow_resume_run/arguments.txt" >/dev/null
grep -- '--difficulty-sampling-exponent' \
    "$crow_resume_run/arguments.txt" >/dev/null
grep -- '^0.25$' "$crow_resume_run/arguments.txt" >/dev/null
test -s "$crow_resume_run/learner.safetensors"

if (
    NUMI_CROW_CURRICULUM_ROOT=$crow_curriculum_root \
    NUMI_CROW_CURRICULUM_BUILD=$crow_curriculum_build \
    NUMI_CROW_CURRICULUM_MLX=/usr/bin/python3 \
    NUMI_CROW_CURRICULUM_RUNS=$numi_temp/runs/crow-invalid-transfer \
    NUMI_CROW_COURSE=sensor-fast \
    NUMI_CROW_PARENT_MODE=actor-transfer \
    NUMI_CROW_PARENT_POLICY=$crow_actor_source \
    NUMI_CROW_PARENT_STATE=$crow_sensor_run/learner.safetensors \
    NUMI_CROW_START_BAND=0 \
    NUMI_CROW_MAXIMUM_BAND=0 \
        "$numi_repo/tools/crow_journey_curriculum_supervisor.sh"
) > "$numi_temp/crow-invalid-curriculum-transfer.log" 2>&1; then
    printf '%s\n' 'sensor-fast supervisor accepted actor transfer with learner state' >&2
    exit 1
fi
grep -- 'sensor-fast actor transfer requires a source policy, no learner state' \
    "$numi_temp/crow-invalid-curriculum-transfer.log" >/dev/null

numi_evaluate_run=$numi_temp/runs/evaluate
numi_evaluate_output=$(
    cd "$numi_repo"
    NUMI_BUILD_DIR=$numi_temp/fake-build \
    NUMI_RUN_DIR=$numi_evaluate_run \
        "$numi_repo/tools/numi" evaluate \
            --interaction-pack sample.interactionpack \
            --interaction-clip sample-clip \
            --repeats 1
)
[ "$numi_evaluate_output" = "fake-evaluate" ]
grep -- '--native-policy' "$numi_evaluate_run/arguments.txt" >/dev/null
grep -- '--metallib' "$numi_evaluate_run/arguments.txt" >/dev/null
grep -- 'sample.interactionpack' "$numi_evaluate_run/arguments.txt" >/dev/null
grep -- 'sample-clip' "$numi_evaluate_run/arguments.txt" >/dev/null
test -s "$numi_evaluate_run/revision.txt"
test -s "$numi_evaluate_run/runtime.sha256"
test -s "$numi_evaluate_run/artifacts.sha256"

printf 'numi CLI discovery and override checks passed\n'

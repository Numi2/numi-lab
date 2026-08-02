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

printf '%s\n' '#!/bin/sh' 'printf "fake-train\n"' \
    > "$numi_temp/fake-build/bin/metalrobo_task_train"
chmod +x "$numi_temp/fake-build/bin/metalrobo_task_train"

printf '%s\n' '#!/bin/sh' 'printf "fake-evaluate\n"' \
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
[ "$numi_train_output" = "fake-train" ]
grep -- '--initialize-policy' "$numi_train_run/arguments.txt" >/dev/null
grep -- '--updated-policy-pack' "$numi_train_run/arguments.txt" >/dev/null
grep -- '--mlx-python' "$numi_train_run/arguments.txt" >/dev/null
grep -- 'sample.interactionpack' "$numi_train_run/arguments.txt" >/dev/null
grep -- 'sample-clip' "$numi_train_run/arguments.txt" >/dev/null
test -s "$numi_train_run/revision.txt"
test -s "$numi_train_run/runtime.sha256"
test -s "$numi_train_run/artifacts.sha256"

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

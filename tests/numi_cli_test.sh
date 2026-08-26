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
    'state_trace=' \
    'steps=1' \
    'take=' \
    'for argument do' \
    '  if [ "$take" = policy ]; then policy=$argument; take=; continue; fi' \
    '  if [ "$take" = state_trace ]; then state_trace=$argument; take=; continue; fi' \
    '  if [ "$take" = steps ]; then steps=$argument; take=; continue; fi' \
    '  [ "$argument" = --policy-pack ] && take=policy' \
    '  [ "$argument" = --state-trace ] && take=state_trace' \
    '  [ "$argument" = --steps ] && take=steps' \
    'done' \
    'if [ -n "$state_trace" ]; then' \
    '  {' \
    '    printf "# step nq=1 scene_bodies=0 scene_stride=13 timestep=0.02 environment=0\n"' \
    '    step=1' \
    '    while [ "$step" -le "$steps" ]; do' \
    '      printf "%s\t0\n" "$step"' \
    '      step=$((step + 1))' \
    '    done' \
    '  } > "$state_trace"' \
    'fi' \
    'case "$policy" in' \
    '  *candidate*) printf "{\"task\":\"velocity\",\"termination_count\":0,\"termination_count_by_environment\":[0],\"failed_environment_steps\":0,\"mean_tracking_score\":0.5,\"mean_tilt\":0.1}\n" ;;' \
    '  *incumbent*) printf "{\"task\":\"velocity\",\"termination_count\":1,\"termination_count_by_environment\":[1],\"failed_environment_steps\":0,\"mean_tracking_score\":0.4,\"mean_tilt\":0.2}\n" ;;' \
    '  *) printf "fake-evaluate\n" ;;' \
    'esac' \
    > "$numi_temp/fake-build/bin/metalrobo_task_rollout"
chmod +x "$numi_temp/fake-build/bin/metalrobo_task_rollout"

printf '%s\n' '#!/bin/sh' \
    'case "${1:-}" in' \
    '  --self-check) printf "{\"schema\":\"numi.robot-catalog-check.v1\",\"robots\":1,\"status\":\"ok\"}\\n" ;;' \
    '  "") printf "{\"schema\":\"numi.robot-catalog.v1\",\"robots\":[{\"id\":\"unitree_g1\"}]}\\n" ;;' \
    '  unitree_g1) printf "{\"schema\":\"numi.robot-pack.v1\",\"robot\":{\"id\":\"unitree_g1\"}}\\n" ;;' \
    '  *) printf "unknown robot: %s\\n" "$1" >&2; exit 2 ;;' \
    'esac' \
    > "$numi_temp/fake-build/bin/metalrobo_robot_catalog"
chmod +x "$numi_temp/fake-build/bin/metalrobo_robot_catalog"
printf 'fake native library\n' > "$numi_temp/fake-build/lib/libmetalrobo.dylib"
printf 'fake metal library\n' > "$numi_temp/fake-build/shaders/MetalRobo.metallib"

numi_version=$($numi_repo/tools/numi version)
[ "$numi_version" = "0.4.0" ]

numi_doctor_help=$($numi_repo/tools/numi doctor --help)
printf '%s\n' "$numi_doctor_help" | grep 'Usage: numi doctor' >/dev/null
[ "$($numi_repo/tools/numi help doctor)" = "$numi_doctor_help" ]
numi_context_help=$($numi_repo/tools/numi context --help)
printf '%s\n' "$numi_context_help" | grep 'Usage: numi context \[--paths\]' >/dev/null
[ "$($numi_repo/tools/numi help context)" = "$numi_context_help" ]
numi_run_help=$($numi_repo/tools/numi run --help)
[ "$numi_run_help" = "Usage: numi run CAPABILITY [arguments...]" ]

numi_help_foundation=$($numi_repo/tools/numi help foundation)
numi_direct_foundation_help=$($numi_repo/tools/numi foundation --help)
[ "$numi_help_foundation" = "$numi_direct_foundation_help" ]

if $numi_repo/tools/numi version unexpected >/dev/null 2>&1; then
    printf 'numi version accepted an unexpected argument\n' >&2
    exit 1
fi
if $numi_repo/tools/numi context unexpected >/dev/null 2>&1; then
    printf 'numi context accepted an unexpected argument\n' >&2
    exit 1
fi
numi_typo=$($numi_repo/tools/numi trian 2>&1 || true)
printf '%s\n' "$numi_typo" | grep "did you mean 'train'" >/dev/null

ln -s "$numi_repo/tools/numi" "$numi_temp/numi"
numi_linked_version=$($numi_temp/numi version)
[ "$numi_linked_version" = "0.4.0" ]

numi_codex_description=$($numi_repo/tools/numi codex --numi-describe)
[ "$numi_codex_description" = "Install or inspect Numi Lab inside Codex." ]

numi_foundation_description=$($numi_repo/tools/numi foundation --numi-describe)
[ "$numi_foundation_description" = \
    "Inspect or run a foundation model as a fingerprinted action-chunk proposer." ]

for numi_command in "$numi_repo"/numi/commands/*; do
    [ -x "$numi_command" ] || continue
    numi_command_description=$(NUMI_LAB_ROOT=$numi_repo \
        "$numi_command" --numi-describe)
    [ -n "$numi_command_description" ]
    [ "$(printf '%s\n' "$numi_command_description" | wc -l | tr -d ' ')" = 1 ]
    NUMI_LAB_ROOT=$numi_repo "$numi_command" --help >/dev/null
done

numi_context=$(
    cd "$numi_temp/workspace"
    NUMI_LAB_ROOT=$numi_repo \
    NUMI_COMMAND_PATH=$numi_temp/extra \
        "$numi_repo/tools/numi" context
)
printf '%s\n' "$numi_context" | grep 'Workspace override.' >/dev/null
printf '%s\n' "$numi_context" | grep 'Extra capability.' >/dev/null
printf '%s\n' "$numi_context" | grep '^  residual-teacher  ' >/dev/null

numi_context_paths=$(
    cd "$numi_temp/workspace"
    NUMI_LAB_ROOT=$numi_repo \
    NUMI_COMMAND_PATH=$numi_temp/extra \
        "$numi_repo/tools/numi" context --paths
)
numi_workspace_physical=$(CDPATH= cd "$numi_temp/workspace" && pwd -P)
printf '%s\n' "$numi_context_paths" | \
    grep "source: $numi_workspace_physical/.numi/commands/train" >/dev/null
printf '%s\n' "$numi_context_paths" | \
    grep "source: $numi_temp/extra/custom" >/dev/null

numi_plugin_version=$(sed -n \
    's/^[[:space:]]*"version":[[:space:]]*"\([^"]*\)".*/\1/p' \
    "$numi_repo/plugins/numi-lab/.codex-plugin/plugin.json")
numi_codex_home=$numi_temp/codex-home
numi_codex_cache=$numi_codex_home/plugins/cache/numi-lab/numi-lab/$numi_plugin_version
mkdir -p "$(dirname "$numi_codex_cache")" "$numi_temp/fake-path" "$numi_temp/user-bin"
cp -R "$numi_repo/plugins/numi-lab" "$numi_codex_cache"
numi_fake_codex_listing=$numi_temp/codex-listing.json
printf '%s\n' \
    '{"installed":[{"pluginId":"numi-lab@numi-lab","version":"'"$numi_plugin_version"'","installed":true,"enabled":true,"source":{"path":"'"$numi_repo"'/plugins/numi-lab"}}]}' \
    > "$numi_fake_codex_listing"
printf '%s\n' '#!/bin/sh' \
    'if [ "${1:-}" = plugin ] && [ "${2:-}" = list ]; then' \
    '  cat "$NUMI_FAKE_CODEX_LISTING"' \
    'fi' \
    'exit 0' \
    > "$numi_temp/fake-path/codex"
chmod +x "$numi_temp/fake-path/codex"

numi_codex_status=$(
    PATH=$numi_temp/fake-path:$PATH \
    CODEX_HOME=$numi_codex_home \
    NUMI_FAKE_CODEX_LISTING=$numi_fake_codex_listing \
        "$numi_repo/tools/numi" codex status
)
printf '%s\n' "$numi_codex_status" | \
    grep 'installed, enabled, and current' >/dev/null

numi_codex_install=$(
    PATH=$numi_temp/fake-path:$PATH \
    CODEX_HOME=$numi_codex_home \
    XDG_BIN_HOME=$numi_temp/user-bin \
    NUMI_FAKE_CODEX_LISTING=$numi_fake_codex_listing \
        "$numi_repo/tools/numi" codex install
)
printf '%s\n' "$numi_codex_install" | grep '^Numi Lab installed and verified.$' >/dev/null
printf '%s\n' "$numi_codex_install" | grep '^Dispatcher: ' >/dev/null
printf '%s\n' "$numi_codex_install" | grep '^Next: start a new Codex task' >/dev/null
if printf '%s\n' "$numi_codex_install" | grep '[{}]' >/dev/null; then
    printf 'numi codex install leaked implementation JSON\n' >&2
    exit 1
fi
test -L "$numi_temp/user-bin/numi"

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

numi_robot=$(
    cd "$numi_repo"
    NUMI_BUILD_DIR=$numi_temp/fake-build \
        "$numi_repo/tools/numi" robots inspect unitree_g1
)
printf '%s\n' "$numi_robot" | grep '"id":"unitree_g1"' >/dev/null

numi_unknown_robot=$(
    cd "$numi_repo"
    NUMI_BUILD_DIR=$numi_temp/fake-build \
        "$numi_repo/tools/numi" robots inspect missing_robot 2>&1 || true
)
printf '%s\n' "$numi_unknown_robot" | \
    grep 'Run `numi robots list` to see available robot IDs.' >/dev/null

numi_doctor=$(
    cd "$numi_repo"
    NUMI_BUILD_DIR=$numi_temp/fake-build \
        "$numi_repo/tools/numi" doctor || true
)
printf '%s\n' "$numi_doctor" | grep 'robot catalog:.*ok' >/dev/null
printf '%s\n' "$numi_doctor" | grep 'Status: ready.' >/dev/null

printf '%s\n' '#!/bin/sh' 'exit 2' \
    > "$numi_temp/fake-build/bin/metalrobo_robot_catalog"
chmod +x "$numi_temp/fake-build/bin/metalrobo_robot_catalog"
numi_doctor=$(
    cd "$numi_repo"
    NUMI_BUILD_DIR=$numi_temp/fake-build \
        "$numi_repo/tools/numi" doctor || true
)
printf '%s\n' "$numi_doctor" | grep 'robot catalog:.*incompatible' >/dev/null
printf '%s\n' "$numi_doctor" | grep 'Status: action required.' >/dev/null

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

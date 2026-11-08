#!/bin/sh
set -eu
repo=$(CDPATH= cd "$(dirname "$0")/.." && pwd -P)
root=$(mktemp -d)
trap 'rm -rf "$root"' EXIT HUP INT TERM
workspace="$root/project with spaces"
build="$root/build"
mkdir -p "$workspace/.numi/commands" "$build/bin" "$build/shaders" "$root/bin"
touch "$workspace/scene.json" "$build/shaders/MetalRobo.metallib"
cat > "$root/bin/cmake" <<'SH'
#!/bin/sh
printf '%s\n' "$*" >> "$TEST_CMAKE_LOG"
SH
cat > "$build/bin/metalrobo_task_rollout" <<'SH'
#!/bin/sh
printf '%s\n' "$@" > "$TEST_ROLLOUT_LOG"
SH
chmod +x "$root/bin/cmake" "$build/bin/metalrobo_task_rollout"
export NUMI_LAB_ROOT="$repo" NUMI_WINDOW_BUILD_DIR="$build" NUMI_WINDOW_SCENE="$workspace/scene.json"
export TEST_CMAKE_LOG="$root/cmake.log" TEST_ROLLOUT_LOG="$root/rollout.log" PATH="$root/bin:$PATH"
cd "$workspace"
"$repo/tools/numi" window --numi-describe > "$root/description"
[ -s "$root/description" ]
"$repo/tools/numi" window --steps 1 > "$root/run.out" 2>&1
grep -F -- "-S $repo -B $build" "$TEST_CMAKE_LOG" >/dev/null
grep -F -- "$workspace/scene.json" "$TEST_ROLLOUT_LOG" >/dev/null
# Project-owned commands take precedence over source-runtime commands.
printf '%s\n' '#!/bin/sh' 'printf "%s\n" project-window' > "$workspace/.numi/commands/window"
chmod +x "$workspace/.numi/commands/window"
[ "$("$repo/tools/numi" window)" = project-window ]
printf '%s\n' 'source workspace window contract passed (stubbed native execution)'

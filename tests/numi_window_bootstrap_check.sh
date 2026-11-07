#!/bin/sh
set -eu
script=$1
root=$(mktemp -d)
trap 'rm -rf "$root"' EXIT
workspace="$root/workspace"
build="$root/build"
mkdir -p "$workspace" "$build/bin" "$build/shaders" "$root/bin"
touch "$workspace/scene.json" "$build/shaders/MetalRobo.metallib"
cat > "$root/bin/cmake" <<'EOF'
#!/bin/sh
printf '%s\n' "$*" >> "$NUMI_BOOTSTRAP_CMAKE_LOG"
EOF
cat > "$build/bin/metalrobo_task_rollout" <<'EOF'
#!/bin/sh
printf '%s\n' "$@" > "$NUMI_BOOTSTRAP_ROLLOUT_LOG"
EOF
chmod +x "$root/bin/cmake" "$build/bin/metalrobo_task_rollout"
export NUMI_WORKSPACE_ROOT="$workspace" NUMI_WINDOW_BUILD_DIR="$build"
export NUMI_BOOTSTRAP_CMAKE_LOG="$root/cmake.log"
export NUMI_BOOTSTRAP_ROLLOUT_LOG="$root/rollout.log"
export PATH="$root/bin:$PATH"
if sh "$script" > "$root/default.out" 2>&1; then
    echo "missing default assets unexpectedly accepted" >&2; exit 1
fi
[ ! -e "$root/cmake.log" ]
for name in g1_29dof.urdf franka_description px4-x500-source; do
    grep -F "$name" "$root/default.out" > /dev/null
done
sh "$script" --window-scene "$workspace/scene.json" > "$root/explicit.out" 2>&1
grep -F -- "$workspace/scene.json" "$root/rollout.log" > /dev/null
if grep -F 'metalrobo_numi_window_catalog_compile' "$root/cmake.log" >/dev/null; then
    echo "explicit scene unnecessarily compiled the built-in catalog" >&2; exit 1
fi
: > "$root/cmake.log"
NUMI_WINDOW_SCENE="$workspace/scene.json" sh "$script" > "$root/env.out" 2>&1
grep -F -- "$workspace/scene.json" "$root/rollout.log" > /dev/null
if grep -F 'metalrobo_numi_window_catalog_compile' "$root/cmake.log" >/dev/null; then
    echo "environment scene unnecessarily compiled the built-in catalog" >&2; exit 1
fi
echo "window bootstrap contract passed (stubbed cmake and rollout)"

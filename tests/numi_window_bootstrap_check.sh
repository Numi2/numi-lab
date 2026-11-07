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
unset NUMI_LAB_ROOT
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
# Installed mode must use the relocated binaries and metallib without CMake.
installed="$root/installed"
mkdir -p "$installed/bin" "$installed/lib/metalrobo"
cp "$build/bin/metalrobo_task_rollout" "$installed/bin/metalrobo_task_rollout"
touch "$installed/lib/metalrobo/MetalRobo.metallib"
cat > "$installed/bin/metalrobo_booster_k1_compile" <<'EOF'
#!/bin/sh
printf '%s\n' "$@" > "$NUMI_BOOTSTRAP_BOOSTER_LOG"
EOF
chmod +x "$installed/bin/metalrobo_booster_k1_compile"
export NUMI_LAB_ROOT="$installed" NUMI_BOOTSTRAP_BOOSTER_LOG="$root/booster.log"
: > "$root/cmake.log"
sh "$script" --window-scene "$workspace/scene.json" > "$root/installed.out" 2>&1
grep -F -- "$workspace/scene.json" "$root/rollout.log" > /dev/null
grep -F -- "$installed/lib/metalrobo/MetalRobo.metallib" "$root/rollout.log" > /dev/null
[ ! -s "$root/cmake.log" ]
sh "$script" add booster-k1 --source "$workspace" > "$root/installed-add.out" 2>&1
grep -F -- "$workspace" "$root/booster.log" > /dev/null
[ ! -s "$root/cmake.log" ]
if sh "$script" > "$root/installed-default.out" 2>&1; then
    echo "installed default unexpectedly accepted missing assets" >&2; exit 1
fi
for name in g1_29dof.urdf franka_description px4-x500-source; do
    grep -F "$name" "$root/installed-default.out" > /dev/null
done
[ ! -s "$root/cmake.log" ]
echo "window bootstrap contract passed (stubbed cmake and rollout)"

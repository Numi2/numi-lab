#!/bin/sh
set -eu
script=$1
dispatcher=$2
root=$(mktemp -d)
trap 'rm -rf "$root"' EXIT
mkdir -p "$root/bin" "$root/build/bin" "$root/build/lib" "$root/build/shaders" "$root/python/metalrobo" "$root/numi/commands" "$root/tools"
cp "$script" "$root/numi/commands/train"
cp "$dispatcher" "$root/tools/numi"
chmod +x "$root/numi/commands/train" "$root/tools/numi"
touch "$root/build/lib/libmetalrobo.dylib" "$root/build/shaders/MetalRobo.metallib" "$root/python/metalrobo/mlx_policy_worker.py" "$root/python/metalrobo/policy_selection.py"
cat > "$root/bin/python" <<'EOF'
#!/bin/sh
case "$*" in
  *policy_selection.py*)
    echo $$ > "$TEST_ROOT/selection.pid"
    sleep 300 &
    echo $! > "$TEST_ROOT/selection-child.pid"
    wait $!
    ;;
  *'import json,sys'*) echo 1 ;;
  *) exit 0 ;;
esac
EOF
cat > "$root/build/bin/metalrobo_task_train" <<'EOF'
#!/bin/sh
echo $$ > "$TEST_ROOT/trainer.pid"
if [ "$TEST_PHASE" = selection ]; then
    echo '{"maximum_episode_steps":1}'
    exit 0
fi
sleep 300 &
echo $! > "$TEST_ROOT/trainer-child.pid"
wait $!
EOF
cat > "$root/build/bin/metalrobo_task_rollout" <<'EOF'
#!/bin/sh
exit 0
EOF
chmod +x "$root/bin/python" "$root/build/bin/metalrobo_task_train" "$root/build/bin/metalrobo_task_rollout"
export TEST_ROOT="$root" NUMI_LAB_ROOT="$root" NUMI_WORKSPACE_ROOT="$root"
export NUMI_BUILD_DIR="$root/build" NUMI_MLX_PYTHON="$root/bin/python"
for phase in trainer selection; do
    export TEST_PHASE="$phase" NUMI_RUN_DIR="$root/run-$phase"
    (cd "$root" && exec "$root/tools/numi" train --steps 1) > "$root/$phase.log" 2>&1 &
    parent=$!
    started=0
    for unused in $(seq 1 100); do
        if [ -s "$root/$phase-child.pid" ]; then started=1; break; fi
        sleep 0.05
    done
    if [ "$started" -ne 1 ]; then
        kill "$parent" 2>/dev/null || true
        wait "$parent" 2>/dev/null || true
        cat "$root/$phase.log" >&2
        echo "$phase did not start" >&2
        exit 1
    fi
    owner=$(cat "$root/$phase.pid")
    child=$(cat "$root/$phase-child.pid")
    kill -TERM "$parent"
    status=0
    wait "$parent" || status=$?
    [ "$status" -eq 143 ] || { echo "$phase cancellation status $status" >&2; exit 1; }
    for unused in $(seq 1 100); do
        if ! kill -0 "$owner" 2>/dev/null && ! kill -0 "$child" 2>/dev/null; then break; fi
        sleep 0.05
    done
    if kill -0 "$owner" 2>/dev/null || kill -0 "$child" 2>/dev/null; then
        echo "$phase descendants survived cancellation" >&2
        kill -TERM "$owner" "$child" 2>/dev/null || true
        exit 1
    fi
    [ ! -d "$NUMI_RUN_DIR/.train.lock" ] || { echo "training lock retained" >&2; exit 1; }
done
echo "dispatcher, trainer and selector cancellation contract passed (stubbed processes)"

#!/bin/sh

# Numi user-command overlay. Keep the lab train implementation authoritative,
# but make a missing selector decision durable so a curriculum supervisor can
# retry instead of waiting forever on an absent selection.json.

set -u

numi_guard_root=${NUMI_LAB_ROOT:?Numi must set NUMI_LAB_ROOT before dispatch}
numi_guard_train="$numi_guard_root/numi/commands/train"

"$numi_guard_train" "$@"
numi_guard_status=$?

# The bundled train command records its own path, but this user overlay is
# part of the effective selector behavior too. Preserve its exact hash in
# future run provenance without modifying an existing run's artifact.
if [ -n "${NUMI_RUN_DIR:-}" ] &&
   [ -f "$NUMI_RUN_DIR/runtime.sha256" ]; then
    numi_guard_hash=""
    if command -v shasum >/dev/null 2>&1; then
        numi_guard_hash=$(shasum -a 256 "$0" | awk '{print $1}')
    elif command -v sha256sum >/dev/null 2>&1; then
        numi_guard_hash=$(sha256sum "$0" | awk '{print $1}')
    fi
    if [ -n "$numi_guard_hash" ]; then
        printf '%s  %s\n' "$numi_guard_hash" "$0" >> "$NUMI_RUN_DIR/runtime.sha256"
        # The bundled command hashes the run before this overlay records its
        # own executable. Refresh the enclosing artifact manifest only after
        # runtime provenance is complete, otherwise runtime.sha256 fails its
        # own retained verification despite a successful selector run.
        find "$NUMI_RUN_DIR" -type f ! -name artifacts.sha256 \
            -exec shasum -a 256 {} \; > "$NUMI_RUN_DIR/artifacts.sha256"
    fi
fi

if [ -n "${NUMI_RUN_DIR:-}" ] &&
   [ ! -s "$NUMI_RUN_DIR/selection/selection.json" ]; then
    mkdir -p "$NUMI_RUN_DIR/selection"
    numi_guard_error="selector exited with status $numi_guard_status"
    if [ "$numi_guard_status" -eq 0 ]; then
        numi_guard_error="selector exited without writing selection.json"
    fi
    numi_guard_python=${NUMI_MLX_PYTHON:-python3}
    if "$numi_guard_python" -c \
        'import json,sys,tempfile,os; path=sys.argv[1]; payload={"schema":"numi.policy-selection.v1","selected":"incumbent","candidate_advanced_deployment":False,"candidate_retained":True,"selection_error":sys.argv[2]}; fd,tmp=tempfile.mkstemp(prefix="selection.",suffix=".tmp",dir=os.path.dirname(path)); os.close(fd); open(tmp,"w").write(json.dumps(payload,indent=2,sort_keys=True)+"\n"); os.replace(tmp,path)' \
        "$NUMI_RUN_DIR/selection/selection.json" "$numi_guard_error"; then
        # The durable incumbent decision is the supervisor's retry signal.
        # Return success so its set -e launch boundary can consume it.
        numi_guard_status=0
    else
        numi_guard_status=1
    fi
fi

exit "$numi_guard_status"

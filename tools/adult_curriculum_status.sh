#!/bin/bash

# Bounded, single-instance status snapshot for the durable adult curriculum.
#
# This command is intentionally not a recursive monitor. It reads one run,
# one selection directory, and a small set of known log files. The global lock
# prevents duplicate polling sessions from competing for the host.

set -eu

usage() {
    cat >&2 <<'EOF'
usage: adult_curriculum_status.sh --run RUN_DIRECTORY [--interval SECONDS]

With no interval, emit one bounded snapshot and exit. Watch mode requires an
interval of at least 30 seconds and is protected by a global single-instance
lock.
EOF
}

numi_status_run=
numi_status_interval=0

while [ "$#" -gt 0 ]; do
    case "$1" in
        --run)
            [ "$#" -ge 2 ] || { usage; exit 2; }
            numi_status_run=$2
            shift 2
            ;;
        --interval)
            [ "$#" -ge 2 ] || { usage; exit 2; }
            numi_status_interval=$2
            shift 2
            ;;
        --help|-h)
            usage >&1
            exit 0
            ;;
        *)
            usage
            exit 2
            ;;
    esac
done

case "$numi_status_run" in
    /*) ;;
    *)
        echo "--run must be an absolute path" >&2
        exit 2
        ;;
esac

[ -d "$numi_status_run" ] || {
    echo "run directory does not exist: $numi_status_run" >&2
    exit 2
}

case "$numi_status_interval" in
    ''|*[!0-9]*)
        echo "--interval must be a non-negative integer" >&2
        exit 2
        ;;
esac

if [ "$numi_status_interval" -gt 0 ] && [ "$numi_status_interval" -lt 30 ]; then
    echo "--interval must be 0 or at least 30 seconds" >&2
    exit 2
fi

numi_status_lock_root=${NUMI_STATUS_LOCK_ROOT:-${TMPDIR:-/tmp}}
numi_status_lock_dir="$numi_status_lock_root/numi-adult-curriculum-status.lock"

numi_status_acquire_lock() {
    if mkdir "$numi_status_lock_dir" 2>/dev/null; then
        printf '%s\n' "$$" > "$numi_status_lock_dir/pid"
        return
    fi

    numi_status_existing_pid=
    if [ -s "$numi_status_lock_dir/pid" ]; then
        numi_status_existing_pid=$(awk 'NR == 1 { print $1; exit }' "$numi_status_lock_dir/pid")
    fi
    if [ -n "$numi_status_existing_pid" ] &&
       kill -0 "$numi_status_existing_pid" 2>/dev/null; then
        echo "another adult curriculum status monitor is already running (pid $numi_status_existing_pid)" >&2
        exit 75
    fi

    rm -f "$numi_status_lock_dir/pid"
    rmdir "$numi_status_lock_dir" 2>/dev/null || {
        echo "could not reclaim adult curriculum status lock: $numi_status_lock_dir" >&2
        exit 75
    }
    mkdir "$numi_status_lock_dir"
    printf '%s\n' "$$" > "$numi_status_lock_dir/pid"
}

numi_status_release_lock() {
    rm -f "$numi_status_lock_dir/pid"
    rmdir "$numi_status_lock_dir" 2>/dev/null || true
}

numi_status_acquire_lock
numi_status_stop() {
    exit 130
}
trap numi_status_release_lock EXIT
trap numi_status_stop INT TERM

numi_status_snapshot() {
    numi_status_selection="$numi_status_run/selection/selection.json"
    numi_status_supervisor_log=${NUMI_STATUS_SUPERVISOR_LOG:-}

    printf 'TIME '
    date
    printf 'HOST '
    hostname
    printf 'RUN %s\n' "$numi_status_run"

    printf 'PROCESS\n'
    ps -axo pid=,ppid=,pcpu=,pmem=,state=,etime=,command= |
        awk '$0 !~ /zsh -c/ && $0 !~ /awk / && $0 ~ /policy_selection\.py|metalrobo_task_rollout|metalrobo_task_train|adult_curriculum_supervisor\.sh/ { print }'

    printf 'SELECTION\n'
    if [ -s "$numi_status_selection" ]; then
        python3 -c '
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    decision = json.load(handle)
keys = (
    "selected",
    "candidate_advanced_deployment",
    "selected_candidate_label",
    "adult_current_band",
    "adult_previous_band",
    "selection_method",
    "selection_score",
    "regressions",
    "improvements",
    "selection_error",
)
print(json.dumps({key: decision.get(key) for key in keys}, sort_keys=True))
' "$numi_status_selection"
    else
        echo pending
    fi

    printf 'COUNTS '
    if [ -d "$numi_status_run/selection" ]; then
        numi_status_current_count=$(find "$numi_status_run/selection" -maxdepth 1 -type f -name '*.evidence.json' ! -name '*.previous-band.evidence.json' | wc -l | tr -d ' ')
        numi_status_previous_count=$(find "$numi_status_run/selection" -maxdepth 1 -type f -name '*.previous-band.evidence.json' | wc -l | tr -d ' ')
        printf 'current=%s previous=%s\n' "$numi_status_current_count" "$numi_status_previous_count"
    else
        echo 'current=0 previous=0'
    fi

    printf 'LATEST_EVIDENCE\n'
    if [ -d "$numi_status_run/selection" ]; then
        find "$numi_status_run/selection" -maxdepth 1 -type f \
            \( -name '*.evidence.json' -o -name '*.previous-band.evidence.json' \) \
            -exec stat -f '%m %N' {} + 2>/dev/null |
            sort -nr |
            head -n 6
    fi

    if [ -n "$numi_status_supervisor_log" ] && [ -f "$numi_status_supervisor_log" ]; then
        printf 'SUPERVISOR_LOG_TAIL\n'
        tail -n 12 "$numi_status_supervisor_log"
    fi

    printf 'THERMAL\n'
    pmset -g therm 2>&1 || true
    printf 'SWAP\n'
    sysctl vm.swapusage 2>&1 || true

    printf 'KNOWN_LOG_ERRORS\n'
    for numi_status_log in \
        "$numi_status_run/stdout.log" \
        "$numi_status_run/stderr.log" \
        "$numi_status_run/train.log" \
        "$numi_status_run/selection/selection.log"; do
        if [ -f "$numi_status_log" ]; then
            grep -Ein -m 10 \
                'gpu error|nonfinite|fatal|broken pipe|selection_error|oom|out of memory|timed out|failed_environment_steps[^0-9]*[1-9][0-9]*|failed_steps[^0-9]*[1-9][0-9]*|timeout_count[^0-9]*[1-9][0-9]*|pending_timeout_episode_count[^0-9]*[1-9][0-9]*' \
                "$numi_status_log" || true
        fi
    done
}

while :; do
    numi_status_snapshot
    [ "$numi_status_interval" -gt 0 ] || break
    sleep "$numi_status_interval"
done

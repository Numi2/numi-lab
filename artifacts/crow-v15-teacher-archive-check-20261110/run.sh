#!/bin/bash
set -euo pipefail
root="$PWD"
work=$(mktemp -d /tmp/crow-verify-XXXXXXXX)
trap 'rm -rf "$work"' EXIT
git clone --quiet --filter=blob:none --no-checkout --branch agent/american-crow-rl-20261107 git@github.com:Numi2/numi-lab.git "$work/repo"
git -C "$work/repo" sparse-checkout init --cone
dirs=(crow-v15-perch-toe-contact-gate-20261110 crow-v15-perch-toe-debounce-20261110 crow-v15-perch-damped-hip-20261110 crow-v15-route-teacher-diagnostic-20261110 crow-v15-route-teacher-signal-fix-20261110 crow-v15-route-teacher-batch-20261110 crow-v15-route-teacher-earlier-turn-20261110 crow-v15-route-teacher-recovery-20261110 crow-v15-teacher-label-signal-fix-20261110 crow-v15-flight-low-speed-20261110 crow-v15-flight-encoded-speed-20261110 crow-v15-flight-reverse-transition-20261110)
paths=(); for d in "${dirs[@]}"; do paths+=("artifacts/$d"); done
git -C "$work/repo" sparse-checkout set "${paths[@]}"
git -C "$work/repo" checkout --quiet f60c6a2b0
printf "restored commit %s\n" "$(git -C "$work/repo" rev-parse HEAD)"
python3 "$root/artifacts/crow-v15-archive-verification-20261110/verify.py" "$work/repo" "${dirs[@]}"

#!/bin/bash
set -euo pipefail
root="$PWD"
work=$(mktemp -d /tmp/crow-verify-XXXXXXXX)
trap 'rm -rf "$work"' EXIT
git clone --quiet --filter=blob:none --no-checkout --branch agent/american-crow-rl-20261107 git@github.com:Numi2/numi-lab.git "$work/repo"
git -C "$work/repo" sparse-checkout init --cone
dirs=(crow-v15-gait-kinematic-jacobian-20261110 crow-v15-gait-smooth-liftoff-20261110 crow-v15-gait-smooth-clearance-20261110 crow-v15-toe-grip-redamping-20261110 crow-v15-gait-jacobian-candidate-20261110 crow-v15-gait-slower-cycle-20261110 crow-v15-gait-support-extension-20261110 crow-v15-grip-slow-close-20261110 crow-v15-grip-soft-release-20261110 crow-v15-gait-coordinated-stance-20261110 crow-v15-gait-coordinated-confirmation-20261110 crow-v15-integrated-route-baseline-20261110 crow-v15-landing-toe-ramp-20261110)
paths=(); for d in "${dirs[@]}"; do paths+=("artifacts/$d"); done
git -C "$work/repo" sparse-checkout set "${paths[@]}"
git -C "$work/repo" checkout --quiet 25bb75802
printf "restored commit %s\n" "$(git -C "$work/repo" rev-parse HEAD)"
python3 "$root/artifacts/crow-v15-archive-verification-20261110/verify.py" "$work/repo" "${dirs[@]}"

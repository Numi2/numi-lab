#!/bin/bash
set -euo pipefail
root="$PWD"
work=$(mktemp -d /tmp/crow-verify-XXXXXXXX)
trap 'rm -rf "$work"' EXIT
git clone --quiet --filter=blob:none --no-checkout --branch agent/american-crow-rl-20261107 git@github.com:Numi2/numi-lab.git "$work/repo"
git -C "$work/repo" sparse-checkout init --cone
dirs=(crow-v15-leg-push-sweep-20261110 crow-v15-leg-push-timing-20261110 crow-v15-launch-com-difference-20261110 crow-v15-landing-support-alignment-20261110 crow-v15-landing-forward-offset-20261110)
paths=(); for d in "${dirs[@]}"; do paths+=("artifacts/$d"); done
git -C "$work/repo" sparse-checkout set "${paths[@]}"
git -C "$work/repo" checkout --quiet 008195cd7
printf "restored commit %s\n" "$(git -C "$work/repo" rev-parse HEAD)"
python3 "$root/artifacts/crow-v15-archive-verification-20261110/verify.py" "$work/repo" "${dirs[@]}"

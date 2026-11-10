#!/bin/bash
set -euo pipefail
root="$PWD"
work=$(mktemp -d /tmp/crow-verify-XXXXXXXX)
trap 'rm -rf "$work"' EXIT
git clone --quiet --filter=blob:none --no-checkout --branch agent/american-crow-rl-20261107 git@github.com:Numi2/numi-lab.git "$work/repo"
git -C "$work/repo" sparse-checkout init --cone
dirs=(crow-v15-loaded-balance-20261110 crow-v15-gentle-balance-20261110 crow-v15-landing-joint-identification-20261110 crow-v15-pitch-balance-20261110 crow-v15-staged-descent-20261110 crow-v15-staged-descent-height-gate-20261110 crow-v15-descent-retry-20261110 crow-v15-contact-wing-fade-20261110)
paths=(); for d in "${dirs[@]}"; do paths+=("artifacts/$d"); done
git -C "$work/repo" sparse-checkout set "${paths[@]}"
git -C "$work/repo" checkout --quiet 3cbdc4d2e
printf "restored commit %s\n" "$(git -C "$work/repo" rev-parse HEAD)"
python3 "$root/artifacts/crow-v15-archive-verification-20261110/verify.py" "$work/repo" "${dirs[@]}"

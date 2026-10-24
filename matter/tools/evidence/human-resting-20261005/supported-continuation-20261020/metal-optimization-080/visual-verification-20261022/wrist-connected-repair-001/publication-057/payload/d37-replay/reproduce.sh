#!/bin/sh
set -eu
R=${NUMI_REPO_ROOT:-/Users/numi/numi-human-aortic-contact-opt-20261019}
W="$R/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001"
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
CANDIDATE="$W/publication-056/payload/bilateral-hand-functional-rebind-020/run-001/candidate.nhskin"
SOURCE_CSV="$W/native-candidate020-mechanical-source-024/source-preflight-001/registered-skin.csv"
OUT=${1:-/tmp/d37-registered-skin.csv}
clang++ -std=c++17 -O2 "$HERE/export_pose_lbs_csv.cpp" -o /tmp/export-pose-lbs-csv-057
/tmp/export-pose-lbs-csv-057 "$CANDIDATE" "$HERE/body-frames.json" "$SOURCE_CSV" "$OUT"
echo "b766fd63b13bf87384ec5a0893356dcc139b7f8d079110096972f9d98f3cc4ad  $OUT" | shasum -a 256 -c -

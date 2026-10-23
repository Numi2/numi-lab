#!/bin/bash
set -euo pipefail
ROOT=/Users/n/numi-human-aortic-contact-opt-20261019
W=$ROOT/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001
E=$W/ankle-coupled-021-on-294-001
O=$E/native-compileworld-candidate-001
APP=/Users/n/numi-human-separation-20261018/C/wholebody-integration-295/build-wholebody-016/source/apps
BUILD=/tmp/numi-human-joint-owner-build-033
PY=/Users/n/numi-human-prep-venv-20261005/bin/python
LIB=$BUILD/lib/libmetalrobo.dylib
test "$(shasum -a 256 "$LIB" | awk '{print $1}')" = b4c83d0df4edec0cc00fb40853158499a9d34d90e315a63f9af554b065657149
"$PY" "$O/rebuild_commonmode_layer_candidate.py" --output /tmp/numi-ankle-commonmode-rebuilt.npz --report "$E/layer-rebuild-receipt.json"
"$PY" "$O/audit_donor_transport_residuals.py"
"$PY" "$O/prepare_compileworld_candidate.py" > "$O/preparation.stdout.json"
/usr/bin/clang++ -std=c++23 -O2 -Wall -Wextra -Wpedantic -Werror -arch arm64 \
  -I"$O" -I"$APP" -I"$ROOT/include" -I"$ROOT/matter/include" -I"$ROOT/src" -I"$BUILD/generated" \
  "$O/check_commonmode_compileworld.cpp" -L"$BUILD/lib" -lmetalrobo -Wl,-rpath,"$BUILD/lib" \
  -o "$O/check_commonmode_compileworld.rebuilt"
if "$O/check_commonmode_compileworld.rebuilt" "$O/human_skin_candidate.nmatter" "$O/human_fat_candidate.nmatter" > "$O/run-final.log" 2> "$O/run-final.stderr"; then
  status=0
else
  status=$?
fi
printf '{"schema":"numi.human.ankle-294-cpu-check.v1","exit_code":%s,"gpu":false,"simulation_steps":0,"layers_admitted":false}\n' "$status" > "$O/run-final-receipt.json"
cat "$O/run-final.log"
cat "$O/run-final.stderr"
test "$status" -eq 0

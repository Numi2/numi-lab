#!/bin/bash
set -euo pipefail
repo="${NUMI_REPO:-$(cd "$(dirname "$0")/../../../.." && pwd)}"
base_commit="${MPM_ABI_BASE_COMMIT:-2a677b3d2634d9cad27b6c83d3ca49053ea8ec5e}"
clang="${CXX:-/usr/bin/clang++}"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
mkdir -p "$tmp/legacy/include/numi/matter" "$tmp/extended/include/numi/matter"
git -C "$repo" show "$base_commit:matter/include/numi/matter/matter.hpp" > "$tmp/legacy/include/numi/matter/matter.hpp"
if [[ -n "${MPM_HEADER_OVERLAY:-}" ]]; then
  cp "$MPM_HEADER_OVERLAY/numi/matter/matter.hpp" "$tmp/extended/include/numi/matter/matter.hpp"
else
  cp "$repo/matter/include/numi/matter/matter.hpp" "$tmp/extended/include/numi/matter/matter.hpp"
fi
src="$(dirname "$0")/view_layout_probe.cpp"
"$clang" -std=c++23 -I"$tmp/legacy/include" -I"$repo/matter/include" -I"$repo/include" "$src" -o "$tmp/legacy-probe"
"$clang" -std=c++23 -DMPM_EXTENSION_LAYOUT=1 -I"$tmp/extended/include" -I"$repo/matter/include" -I"$repo/include" "$src" -o "$tmp/extended-probe"
"$tmp/legacy-probe" > "$tmp/legacy.txt"
"$tmp/extended-probe" > "$tmp/extended.txt"
python3 - "$tmp/legacy.txt" "$tmp/extended.txt" <<'PY'
import sys
def read(path):
    out={}
    for line in open(path):
        k,v=line.rstrip().split('=',1)
        out[k]=int(v)
    return out
old,new=map(read,sys.argv[1:])
for name,value in old.items():
    if name == 'sizeof':
        continue
    if new.get(name)!=value:
        raise SystemExit(f"legacy prefix changed: {name}: {value} -> {new.get(name)}")
legacy_end = old['microstepTimestepSeconds'] + 4  # final legacy field is float
if new.get('particleAccepted') != legacy_end:
    raise SystemExit(f"first MPM field does not follow legacy fields: {new.get('particleAccepted')} != {legacy_end}")
append=['particleAccepted','particleCandidate','particleCheckpoint','particleMaterialStateDefaults','particleMaterialStateAccepted','particleMaterialStateCandidate','particleMaterialStateCheckpoint','particleCount','materialStateStride']
if any(new[a] >= new[b] for a,b in zip(append,append[1:])):
    raise SystemExit('appended MPM fields are not monotone')
print(f"PASS legacy_size={old['sizeof']} extended_size={new['sizeof']} first_mpm_offset={new['particleAccepted']} legacy_field_end={legacy_end}")
print(f"PASS common_prefix_fields={len(old)-1} append_fields={len(append)}")
PY

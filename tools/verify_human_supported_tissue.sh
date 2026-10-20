#!/bin/sh
# One entrypoint over the existing preparation tests, geometry auditors and native replay.
set -eu
case "$(hostname)" in numi|numi.local) ;; *) echo "Run this check on SSH Numi." >&2; exit 2;; esac
VERIFY_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
VERIFY_OUT=${1:?usage: verify_human_supported_tissue.sh /absolute/new-output [--native|--rollback]}
VERIFY_MODE=${2:-}
if test "$#" -gt 2; then echo "Too many arguments." >&2; exit 2; fi
case "$VERIFY_OUT" in /*) ;; *) echo "Output must be an absolute fresh directory." >&2; exit 2;; esac
case "$VERIFY_MODE" in ""|--native|--rollback) ;; *) echo "Unknown verification mode." >&2; exit 2;; esac
test ! -e "$VERIFY_OUT"
mkdir -p "$VERIFY_OUT"
cd "$VERIFY_ROOT"
. /Users/numi/.config/numi/metal-toolchain.env
VERIFY_BUILD="$VERIFY_ROOT/build-integrated-validation-20261019/profile-build-039"
VERIFY_EVIDENCE="$VERIFY_ROOT/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020"
case "${NUMI_MATTER_VASCULAR_BLOCK_PRECONDITIONER:-0}" in
  0) VERIFY_TEMPLATE="$VERIFY_EVIDENCE/elastic-attachment-022/template.json" ;;
  1) VERIFY_TEMPLATE="$VERIFY_EVIDENCE/vascular-principal-block-033/template.json" ;;
  *) echo "Vascular block policy must be exactly 0 or 1." >&2; exit 2 ;;
esac
python3 - "$VERIFY_ROOT" >"$VERIFY_OUT/input-identities.json" <<'PY'
import pathlib,hashlib,re,sys,json
r=pathlib.Path(sys.argv[1])
p=r/"matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/mesh-quality-018/candidate-layers.csv"
actual=hashlib.file_digest(p.open("rb"),"sha256").hexdigest()
h=(r/"apps/NumiHumanWholeBodySkinReduction.hpp").read_text()
expected=re.search(r'kWholeBodySkinLayerCoordinatesSHA256="([a-f0-9]{64})"',h).group(1)
assert actual==expected,"audited CSV differs from the compiled header identity"
print(json.dumps(dict(coordinates_sha256=actual,header_sha256=hashlib.sha256(h.encode()).hexdigest()),indent=2))
PY
/opt/homebrew/bin/cmake --build "$VERIFY_BUILD" --target numi-human-native metalrobo_numilab_human_tendon_fem_load_probe numi-matter-cardboard-contact-geometry-check numi-matter-fem-reference-check numi-matter-vascular-cavity-check -j 2 >"$VERIFY_OUT/build.log" 2>&1
for VERIFY_TEST in human_wholebody_skin_layers human_skin_fat_from_layers_precomputed numi_human_registered_ligaments_test human_accepted_fem_skin_transfer_mixed_donor; do
  clang++ -std=c++20 -O2 -Wall -Wextra -Wpedantic -Werror -fno-fast-math -Iapps -Iinclude -Imatter/include \
    "tests/$VERIFY_TEST.cpp" "$VERIFY_BUILD/matter/libnumi_matter_compiler.a" -framework Foundation \
    -o "$VERIFY_OUT/$VERIFY_TEST" >"$VERIFY_OUT/$VERIFY_TEST.compile.log" 2>&1
  "$VERIFY_OUT/$VERIFY_TEST" >"$VERIFY_OUT/$VERIFY_TEST.run.log" 2>&1
done
clang++ -std=c++20 -O2 -Wall -Wextra -Wpedantic -Werror -Iapps -Iinclude -Imatter/include \
  "$VERIFY_EVIDENCE/mesh-quality-018/retained-float32-audit.cpp" -o "$VERIFY_OUT/float32-audit" >"$VERIFY_OUT/float32-audit.compile.log" 2>&1
"$VERIFY_OUT/float32-audit" tools/evidence/human-wholebody-skinfat-20261019/c296/candidate-024-safe.txt \
  "$VERIFY_EVIDENCE/mesh-quality-018/candidate-layers.csv" "$VERIFY_OUT/float32-geometry.json" >"$VERIFY_OUT/float32-audit.run.log" 2>&1
clang++ -std=c++20 -O2 -Wall -Wextra -Wpedantic -Werror -Iapps -Iinclude -Imatter/include \
  tools/evidence/human-wholebody-skinfat-20261019/c300/auditor.cpp -o "$VERIFY_OUT/compact-boundary-audit" >"$VERIFY_OUT/compact-boundary.compile.log" 2>&1
mkdir "$VERIFY_OUT/compact-boundary"
"$VERIFY_OUT/compact-boundary-audit" tools/evidence/human-wholebody-skinfat-20261019/c296/candidate-024-safe.txt \
  "$VERIFY_EVIDENCE/mesh-quality-018/candidate-layers.csv" \
  /Users/numi/numi-human-separation-20261018/C/wholebody-closed-boundary-audit-300/input/registered-skin-mechanical.csv \
  "$VERIFY_OUT/compact-boundary" 909d364267a09340ecd2038ed901efb6732b5d3b5a525f65221dbccc3cd40526 \
  8a82e721cc3800fe2d7293a927ed4d4084e89e4ff260dfc6c44a344bd5b80b2e \
  c48e4d9076d90c6f8f4d308918e1ce6925739d1ee5e50b5affb234e8e54d4eb5 >"$VERIFY_OUT/compact-boundary.run.log" 2>&1
"$VERIFY_BUILD/bin/metalrobo_numilab_human_tendon_fem_load_probe" >"$VERIFY_OUT/fem-body-interface.run.log" 2>&1
"$VERIFY_BUILD/matter/numi-matter-cardboard-contact-geometry-check" >"$VERIFY_OUT/contact-geometry.run.log" 2>&1
"$VERIFY_BUILD/matter/numi-matter-cardboard-contact-geometry-check" --surface-sort-only \
  "$VERIFY_BUILD/matter/shaders/NumiMatter.metallib" >"$VERIFY_OUT/contact-surface-sort.run.log" 2>&1
"$VERIFY_BUILD/matter/numi-matter-fem-reference-check" --backtracking-only >"$VERIFY_OUT/fem-backtracking.run.log" 2>&1
"$VERIFY_BUILD/matter/numi-matter-vascular-cavity-check" --block-preconditioner-only \
  /Users/n/numi-human-resting-evidence-20261005/reference-circulation-001/resting_reference_lv15.native.v3.json >"$VERIFY_OUT/vascular-block.run.log" 2>&1
python3 - "$VERIFY_ROOT" "$VERIFY_OUT" "$VERIFY_BUILD" <<'PYPROVENANCE'
import pathlib,hashlib,json,subprocess,sys,platform
root,out,build=map(pathlib.Path,sys.argv[1:])
sha=lambda p:hashlib.file_digest(p.open("rb"),"sha256").hexdigest()
sources=["apps/NumiHumanAcceptedFEMSkinTransfer.hpp","apps/NumiHumanWholeBodySkinReduction.hpp","apps/NumiHumanCoarseTissuePreparation.hpp","apps/NumiHumanRegisteredLigaments.hpp","apps/numilab_human_myosim_visual_probe.mm","apps/numi_human_tendon_fem_load_probe.mm","matter/materials/human_subcutaneous_adipose_low_strain_effective.nmatter","tools/verify_human_supported_tissue.sh","tests/human_accepted_fem_skin_transfer_mixed_donor.cpp","tests/human_wholebody_skin_layers.cpp","tests/human_skin_fat_from_layers_precomputed.cpp","tests/numi_human_registered_ligaments_test.cpp","matter/src/runtime.mm","matter/src/metal/fem.metalinc","matter/src/metal/vascular.metalinc","matter/tools/vascular_cavity_check.mm","matter/src/metal/contact.metalinc","matter/tools/cardboard_contact_geometry_check.metal","matter/tools/cardboard_contact_geometry_check.mm","matter/tools/fem_reference_check.mm","matter/src/numi_human.mm","matter/src/metal/numi_human.metalinc","matter/include/numi/matter/numi_human_shared.h"]
receipt=dict(base_revision=subprocess.check_output(["git","rev-parse","HEAD"],cwd=root,text=True).strip(),host=platform.node(),source_sha256={x:sha(root/x) for x in sources},executable_sha256=sha(build/"bin/numi-human-native"),metallib_sha256=sha(build/"matter/shaders/NumiMatter.metallib"),interface_probe_sha256=sha(build/"bin/metalrobo_numilab_human_tendon_fem_load_probe"),suite_library_sha256=sha(build/"lib/libmetalrobo.dylib"),contact_geometry_probe_sha256=sha(build/"matter/numi-matter-cardboard-contact-geometry-check"),fem_reference_probe_sha256=sha(build/"matter/numi-matter-fem-reference-check"),vascular_cavity_probe_sha256=sha(build/"matter/numi-matter-vascular-cavity-check"),contact_geometry_metallib_sha256=sha(build/"matter/shaders/cardboard_contact_geometry_check.metallib"),output_sha256={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob("*")) if p.is_file()},scope="CPU preparation, geometry and native Metal interface checks passed; integrated execution, if requested, has its own invocation and result")
(out/"preparation-provenance.json").write_text(json.dumps(receipt,indent=2)+"\n")
PYPROVENANCE
if test -n "$VERIFY_MODE"; then
  if test "$VERIFY_MODE" = --rollback; then
    python3 "$VERIFY_EVIDENCE/../supported-skin-ligaments-20261020/reproduction.py" \
      "$VERIFY_TEMPLATE" "$VERIFY_OUT/native" --steps 2 --rollback \
      --geometry-audit "$VERIFY_EVIDENCE/elbow/actual_pose_catalog_audit.json"
  else
    python3 "$VERIFY_EVIDENCE/../supported-skin-ligaments-20261020/reproduction.py" \
      "$VERIFY_TEMPLATE" "$VERIFY_OUT/native" --steps 32 \
      --geometry-audit "$VERIFY_EVIDENCE/elbow/actual_pose_catalog_audit.json"
  fi
fi
echo "Supported tissue preparation checks passed: $VERIFY_OUT"

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
VERIFY_GEOMETRY_AUDIT="$VERIFY_EVIDENCE/coracoclavicular-046/actual-pose-044-final.json"
case "${NUMI_MATTER_VASCULAR_BLOCK_PRECONDITIONER:-0}" in
  0) VERIFY_TEMPLATE="$VERIFY_EVIDENCE/elastic-attachment-022/template.json" ;;
  1) VERIFY_TEMPLATE="$VERIFY_EVIDENCE/multipoint-support-048/launch-template.json" ;;
  *) echo "Vascular block policy must be exactly 0 or 1." >&2; exit 2 ;;
esac
VERIFY_PREDICTOR=${NUMI_MATTER_VASCULAR_BLOCK_PREDICTOR:-0}
case "$VERIFY_PREDICTOR" in
  0) ;;
  1) test "${NUMI_MATTER_VASCULAR_BLOCK_PRECONDITIONER:-0}" = 1 || { echo "Vascular predictor requires the existing principal block." >&2; exit 2; } ;;
  *) echo "Vascular predictor policy must be exactly 0 or 1." >&2; exit 2 ;;
esac
# Unrelated synthetic fixtures have no vascular graph. The vascular owner
# fixture selects both policies internally; native Human receives the request.
export NUMI_MATTER_VASCULAR_BLOCK_PREDICTOR=0
python3 - "$VERIFY_ROOT" "$VERIFY_TEMPLATE" "$VERIFY_GEOMETRY_AUDIT" >"$VERIFY_OUT/input-identities.json" <<'PY'
import pathlib,hashlib,re,sys,json
r=pathlib.Path(sys.argv[1])
p=r/"matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/mesh-quality-018/candidate-layers.csv"
actual=hashlib.file_digest(p.open("rb"),"sha256").hexdigest()
h=(r/"apps/NumiHumanWholeBodySkinReduction.hpp").read_text()
expected=re.search(r'kWholeBodySkinLayerCoordinatesSHA256="([a-f0-9]{64})"',h).group(1)
assert actual==expected,"audited CSV differs from the compiled header identity"
ligament_header=r/"apps/NumiHumanRegisteredLigaments.hpp"
ligament_header_sha=hashlib.sha256(ligament_header.read_bytes()).hexdigest()
audit_path=pathlib.Path(sys.argv[3])
template_path=pathlib.Path(sys.argv[2])
assert template_path.is_file(),"selected native template is missing"
audit=json.loads(audit_path.read_text())
assert audit["source"]["catalogHeaderSha256"]==ligament_header_sha,"ligament geometry audit is stale"
assert audit["rows"] and all(row["admittedByAudit"] for row in audit["rows"]),"ligament geometry audit contains rejected rows"
print(json.dumps(dict(coordinates_sha256=actual,header_sha256=hashlib.sha256(h.encode()).hexdigest(),
    ligament_header_sha256=ligament_header_sha,ligament_audit_sha256=hashlib.sha256(audit_path.read_bytes()).hexdigest(),
    ligament_geometry_rows=len(audit["rows"]),selected_template_path=str(template_path),
    selected_template_sha256=hashlib.sha256(template_path.read_bytes()).hexdigest()),indent=2))
PY
/opt/homebrew/bin/cmake -S "$VERIFY_ROOT" -B "$VERIFY_BUILD" -DBUILD_TESTING=ON >"$VERIFY_OUT/configure.log" 2>&1
/opt/homebrew/bin/cmake --build "$VERIFY_BUILD" --target numi-human-native metalrobo_numilab_human_tendon_fem_load_probe numi-matter-cardboard-contact-geometry-check numi-matter-fem-reference-check numi-matter-fem-kinematic-release-check numi-matter-fem-mutable-snapshot-check numi-matter-vascular-cavity-check metalrobo_numi_human_static_support_test metalrobo_numi_human_per_contact_plane_metal_test -j 2 >"$VERIFY_OUT/build.log" 2>&1
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
"$VERIFY_BUILD/bin/metalrobo_numi_human_static_support_test" >"$VERIFY_OUT/static-support.run.log" 2>&1
"$VERIFY_BUILD/metalrobo_numi_human_per_contact_plane_metal_test" "$VERIFY_BUILD/shaders/MetalRobo.metallib" >"$VERIFY_OUT/support-capture-metal.run.log" 2>&1
"$VERIFY_BUILD/matter/numi-matter-cardboard-contact-geometry-check" >"$VERIFY_OUT/contact-geometry.run.log" 2>&1
"$VERIFY_BUILD/matter/numi-matter-cardboard-contact-geometry-check" --surface-sort-only \
  "$VERIFY_BUILD/matter/shaders/NumiMatter.metallib" >"$VERIFY_OUT/contact-surface-sort.run.log" 2>&1
NUMI_MATTER_VASCULAR_BLOCK_PRECONDITIONER=0 "$VERIFY_BUILD/matter/numi-matter-fem-reference-check" --backtracking-only >"$VERIFY_OUT/fem-backtracking.run.log" 2>&1
NUMI_MATTER_VASCULAR_BLOCK_PRECONDITIONER=0 "$VERIFY_BUILD/matter/numi-matter-fem-reference-check" matter/materials/rodero2021_ventricular_guccione_passive.nmatter matter/materials/rodero2021_nonventricular_neo_hookean_passive.nmatter >"$VERIFY_OUT/fem-reference.run.log" 2>&1
NUMI_MATTER_VASCULAR_BLOCK_PRECONDITIONER=0 "$VERIFY_BUILD/matter/numi-matter-fem-kinematic-release-check" >"$VERIFY_OUT/fem-kinematic-release.run.log" 2>&1
NUMI_MATTER_VASCULAR_BLOCK_PRECONDITIONER=0 "$VERIFY_BUILD/matter/numi-matter-fem-mutable-snapshot-check" >"$VERIFY_OUT/fem-mutable-snapshot.run.log" 2>&1
"$VERIFY_BUILD/matter/numi-matter-vascular-cavity-check" --block-preconditioner-only \
  /Users/n/numi-human-resting-evidence-20261005/reference-circulation-001/resting_reference_lv15.native.v3.json >"$VERIFY_OUT/vascular-block.run.log" 2>&1
"$VERIFY_BUILD/matter/numi-matter-vascular-cavity-check" --block-predictor-only >"$VERIFY_OUT/vascular-predictor.run.log" 2>&1
python3 - "$VERIFY_ROOT" "$VERIFY_OUT" "$VERIFY_BUILD" <<'PYPROVENANCE'
import pathlib,hashlib,json,subprocess,sys,platform
root,out,build=map(pathlib.Path,sys.argv[1:])
sha=lambda p:hashlib.file_digest(p.open("rb"),"sha256").hexdigest()
sources=["apps/NumiHumanAcceptedFEMSkinTransfer.hpp","apps/NumiHumanWholeBodySkinReduction.hpp","apps/NumiHumanCoarseTissuePreparation.hpp","apps/NumiHumanRegisteredLigaments.hpp","apps/numilab_human_myosim_visual_probe.mm","apps/numi_human_tendon_fem_load_probe.mm","matter/materials/human_subcutaneous_adipose_low_strain_effective.nmatter","tools/verify_human_supported_tissue.sh","tests/human_accepted_fem_skin_transfer_mixed_donor.cpp","tests/human_wholebody_skin_layers.cpp","tests/human_skin_fat_from_layers_precomputed.cpp","tests/numi_human_registered_ligaments_test.cpp","matter/src/runtime.mm","matter/src/metal/fem.metalinc","matter/src/metal/fgmres.metalinc","matter/src/metal/vascular.metalinc","matter/tools/vascular_cavity_check.mm","matter/src/metal/contact.metalinc","matter/tools/cardboard_contact_geometry_check.metal","matter/tools/cardboard_contact_geometry_check.mm","matter/tools/fem_reference_check.mm","matter/src/numi_human.mm","matter/src/metal/numi_human.metalinc","matter/include/numi/matter/numi_human_shared.h","matter/include/numi/matter/shared.h","include/metalrobo/MetalArticulatedOperator.hpp","src/metal/MetalArticulatedOperator.mm","include/metalrobo/NumiHumanMuscleEquilibrium.hpp","src/core/NumiHumanMuscleEquilibrium.cpp","tests/numi_human_static_support_test.cpp","tests/numi_human_per_contact_plane_metal_test.mm","CMakeLists.txt","matter/CMakeLists.txt","matter/include/numi/matter/fem_compensated_deformation.h","matter/src/compiler_impl.cpp","matter/src/validation_impl.cpp","matter/src/fem_reference_geometry.hpp","matter/src/metal/base_cgls_runtime.metalinc","matter/src/metal/common.metalinc","matter/include/numi/matter/paper_compensated_bytecode.h","include/metalrobo/compensated_translation_gpu.h","apps/numanx_matter_attachment_runtime_probe.mm","matter/tools/cardboard_cached_operator_audit.py","matter/tools/cardboard_offline_conditioning_study.py","matter/tools/cardboard_paper_exact_capture.py","matter/tools/cardboard_solver_capture_analysis.py","matter/tools/cardboard_solver_pair_analyzer.py","matter/tools/test_cardboard_cached_operator_audit.py","matter/tools/test_cardboard_offline_conditioning_study.py","matter/tools/fem_kinematic_release_check.mm","matter/tools/fem_mutable_snapshot_check.mm","matter/tools/cardboard_source370_material_point_check.mm"]
receipt=dict(base_revision=subprocess.check_output(["git","rev-parse","HEAD"],cwd=root,text=True).strip(),host=platform.node(),source_sha256={x:sha(root/x) for x in sources},executable_sha256=sha(build/"bin/numi-human-native"),metallib_sha256=sha(build/"matter/shaders/NumiMatter.metallib"),interface_probe_sha256=sha(build/"bin/metalrobo_numilab_human_tendon_fem_load_probe"),suite_library_sha256=sha(build/"lib/libmetalrobo.dylib"),contact_geometry_probe_sha256=sha(build/"matter/numi-matter-cardboard-contact-geometry-check"),static_support_probe_sha256=sha(build/"bin/metalrobo_numi_human_static_support_test"),support_capture_probe_sha256=sha(build/"metalrobo_numi_human_per_contact_plane_metal_test"),support_capture_metallib_sha256=sha(build/"shaders/MetalRobo.metallib"),fem_reference_probe_sha256=sha(build/"matter/numi-matter-fem-reference-check"),vascular_cavity_probe_sha256=sha(build/"matter/numi-matter-vascular-cavity-check"),contact_geometry_metallib_sha256=sha(build/"matter/shaders/cardboard_contact_geometry_check.metallib"),output_sha256={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob("*")) if p.is_file()},scope="CPU preparation, geometry and native Metal interface checks passed; integrated execution, if requested, has its own invocation and result")
(out/"preparation-provenance.json").write_text(json.dumps(receipt,indent=2)+"\n")
PYPROVENANCE
export NUMI_MATTER_VASCULAR_BLOCK_PREDICTOR="$VERIFY_PREDICTOR"
if test -n "$VERIFY_MODE"; then
  if test "$VERIFY_MODE" = --rollback; then
    python3 "$VERIFY_EVIDENCE/../supported-skin-ligaments-20261020/reproduction.py" \
      "$VERIFY_TEMPLATE" "$VERIFY_OUT/native" --steps 2 --rollback \
      --geometry-audit "$VERIFY_GEOMETRY_AUDIT"
  else
    python3 "$VERIFY_EVIDENCE/../supported-skin-ligaments-20261020/reproduction.py" \
      "$VERIFY_TEMPLATE" "$VERIFY_OUT/native" --steps 32 \
      --geometry-audit "$VERIFY_GEOMETRY_AUDIT"
  fi
fi
echo "Supported tissue preparation checks passed: $VERIFY_OUT"

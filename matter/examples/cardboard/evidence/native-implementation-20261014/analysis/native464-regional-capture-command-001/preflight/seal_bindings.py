#!/usr/bin/env python3
import hashlib,json
from pathlib import Path
B=Path('/Users/home/cardboard-implementation-20261014')
R=B/'analysis/native464-regional-capture-command-001'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
source=B/'snapshots/source-cgls-regional-state-capture-151'
build=Path('/Users/home/numi-cardboard-20261006/build-cardboard-provenance-151/source-build.json')
case462=B/'solver/cgls-block-right-disabled-parity-462'
files=[
 'preflight/build_graph_input.py','preflight/graph_layout_check.cpp','preflight/record_layout.cpp',
 'preflight/preflight_capacity.py','preflight/validate_plan.py','preflight/actual-graph-input.bin',
 'preflight/compiled-package-topology-inventory.json','preflight/graph-layout.json',
 'preflight/capture-capacity-preflight.json','preflight/record-layout.log',
 'preflight/reader-cpu-tests.log','preflight/reader-python-syntax.log',
 'preflight/command-validation.json','preflight/command-validation.log',
 'preflight/graph_layout_check','preflight/record_layout','run_native464.py','README.md',
 'validator/read_regional_capture.py','validator/read_regional_capture_v1.py',
 'validator/read_regional_capture_v2.py','validator/test_reader_v1.py','validator/test_reader_v2.py'
]
manifest={
 'schema':'numi.cardboard.native464.source-bindings.v2',
 'scope':'source/build/capture-command preflight; no 464 native execution',
 'source151_snapshot':str(source),
 'source151_frozen_manifest_sha256':sha(source/'frozen-source.json'),
 'source151_runtime_sha256':sha(source/'matter/src/runtime.mm'),
 'source151_regional_reader_overlay_sha256':sha(B/'analysis/base-cgls-regional-state-capture-overlay-151-001/overlay-manifest.json'),
 'source151_regional_reader_source_sha256':sha(B/'analysis/base-cgls-regional-state-capture-overlay-151-001/tools/read_regional_capture.py'),
 'source151_build_receipt':str(build),'source151_build_receipt_sha256':sha(build),
 'source151_executable_sha256':sha('/Users/home/numi-cardboard-20261006/build-cardboard-provenance-151/matter/numi-matter-cardboard-probe'),
 'source151_metallib_sha256':sha('/Users/home/numi-cardboard-20261006/build-cardboard-provenance-151/matter/shaders/NumiMatter.metallib'),
 'source151_changed_sources':json.loads(build.read_text()).get('changed_sources'),
 'source151_source_binding_count':len(json.loads(build.read_text())['source_sha256']),
 'source151_configured_target':json.loads(build.read_text())['targets'],
 'source462_command_sha256':sha(B/'native-462-command.json'),
 'source462_receipt_sha256':sha(case462/'receipt.json'),
 'source462_output_manifest_sha256':sha(case462/'output/manifest.json'),
 'source462_compiled_package_sha256':sha(case462/'output/compiled.nmatterpack'),
 'source462_stderr_sha256':sha(case462/'stderr.log'),
 'source462_complete_history_parity_report':str(B/'analysis/source149-right-scale-disabled462-parity-001/report.json'),
 'source462_complete_history_parity_report_sha256':sha(B/'analysis/source149-right-scale-disabled462-parity-001/report.json'),
 'source151_material_sha256':{
   'liner':sha(source/'matter/materials/hajali2009_liner_hill_bilinear_fit.nmatter'),
   'medium':sha(source/'matter/materials/hajali2009_medium_hill_bilinear_fit.nmatter')},
 'compiled_package_topology_inventory_sha256':sha(R/'preflight/compiled-package-topology-inventory.json'),
 'graph_layout_preflight_sha256':sha(R/'preflight/graph-layout.json'),
 'capture_capacity_preflight_sha256':sha(R/'preflight/capture-capacity-preflight.json'),
 'validator_contract_tests_sha256':sha(R/'preflight/reader-cpu-tests.log'),
 'validator_python_syntax_sha256':sha(R/'preflight/reader-python-syntax.log'),
 'source151_graph_helper_header_sha256':sha(source/'matter/src/regional_graph_schwarz.hpp'),
 'source151_shared_abi_sha256':sha(source/'matter/include/numi/matter/shared.h'),
 'planned_case':str(B/'solver/cgls-regional-state-capture-464'),
 'planned_sidecars':{
    'state':'base-cgls.bin.state.json/.state.bin',
    'regional':'base-cgls.bin.regional.json/.regional.bin'},
 'capture_preflight':json.loads((R/'preflight/capture-capacity-preflight.json').read_text()),
 'regional_capture_currentness_limit':{
    'source151_restart_tag_semantics':'restart_generation.w is producer cycle c; runtime consumes augmented mode at cycle c+1',
    'independent_consumed_cycle_captured':False,
    'reader_v2_augmented_admission':'always false without independent consumed-cycle evidence; do not promote tag match/mode-valid/factor-valid alone'},
 'environment_role_note':'source463 correction: environment 0 is active/eligible; environment 1 is quiet. Actual sidecar selection and fgmres-health words remain authoritative.',
 'preflight_files':{name:sha(R/name) for name in files},
 'qualification_boundary':'planned native diagnostic command only; no source151 464 native execution, no selector-5 action qualification, no solver/physical/assembly qualification',
}
(R/'source151-regional-capture-bindings.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
# Refresh command input digest list without admitting command/self hash loops.
cmd_path=R/'native-464-command.json'; cmd=json.loads(cmd_path.read_text())
separator=cmd.index('--')
current=set(cmd[i+1] for i,v in enumerate(cmd) if v=='--input-file')
add=[str(R/name) for name in [
 'preflight/validate_plan.py','preflight/seal_bindings.py',
 'preflight/command-validation.json','preflight/command-validation.log',
 'preflight/graph_layout_check','preflight/record_layout',
 'run_native464.py','README.md'
] if str(R/name) not in current]
for name in add:
 if Path(name).name in {Path(p).name for p in current}: raise SystemExit(f'duplicate basename {Path(name).name}')
insert=[]
for name in add:insert+=['--input-file',name]
cmd[separator:separator]=insert
cmd_path.write_text(json.dumps(cmd,indent=2)+'\n')
print(json.dumps({'bindings_sha256':sha(R/'source151-regional-capture-bindings.json'),'command_sha256':sha(cmd_path),'preflight_files':len(files)+len(add)},indent=2))

#!/usr/bin/env python3
import hashlib,json,shutil
from pathlib import Path
B=Path('/Users/home/cardboard-implementation-20261014')
root=B/'analysis/native464-regional-capture-command-001'
cmd=json.loads((B/'native-462-command.json').read_text())
oldbase='/Users/home/cardboard-implementation-20261014/snapshots/source-cgls-block-right-preconditioned-149'
newbase='/Users/home/cardboard-implementation-20261014/snapshots/source-cgls-regional-state-capture-151'
oldbuild='/Users/home/numi-cardboard-20261006/build-cardboard-provenance-149'
newbuild='/Users/home/numi-cardboard-20261006/build-cardboard-provenance-151'
oldcase='/Users/home/cardboard-implementation-20261014/solver/cgls-block-right-disabled-parity-462'
newcase='/Users/home/cardboard-implementation-20261014/solver/cgls-regional-state-capture-464'
# Rewrite source/build/case-owned input paths while preserving all other physical and solver arguments.
new=[]
for a in cmd:
    if a == 'Does source149 with the block3 right-scaling option absent preserve full attempted history under the exact source459 protocol?':
        a='At the selected source151 CGLS Newton point, does the bounded regional state observer emit complete joined state and selector-5 graph arrays with their raw augmentation tags? This is a readback diagnostic, not solver or physical qualification.'
    elif a.startswith('Run source149/build149'):
        a='Run source151/build151 with the same physical, solver, material, tolerance, and phase settings as the retained source149 control, adding only the opt-in regional state observer and omitting the incompatible coupled observer. The reader must keep augmented RAP unadmitted unless the consumed restart cycle is independently known.'
    else:
        a=a.replace(oldbase,newbase).replace(oldbuild,newbuild).replace(oldcase,newcase)
    new.append(a)
# Change exact same output destinations to the new, unique case.
# Remove the coupled observer because the source151 bounded reader rejects overlapping selected observers.
filtered=[]
i=0
while i<len(new):
    if new[i]=='--env' and i+1<len(new) and new[i+1].startswith('NM_FEM_COUPLED_CAPTURE_SELECTOR='):
        i+=2; continue
    if new[i]=='--env' and i+1<len(new) and new[i+1].startswith('NM_FEM_COUPLED_CAPTURE_PATH='):
        i+=2; continue
    filtered.append(new[i]); i+=1
new=filtered
# Explicitly enable the source151 regional sidecar alongside the already-selected CGLS state capture.
ix=new.index('NM_FEM_BASE_CGLS_STATE_CAPTURE=1')
new[ix:ix+1]=['NM_FEM_BASE_CGLS_STATE_CAPTURE=1','--env','NM_FEM_BASE_CGLS_STATE_CAPTURE_REGIONAL=1']
# Add one uniquely named bound input; the combined document binds all source/build/preflight materials.
new += ['--input-file',str(root/'source151-regional-capture-bindings.json')]
(root/'native-464-command.json').write_text(json.dumps(new,indent=2)+'\n')
# Keep a source-bound copy of the runner logic but adapt only manifest/case/log names.
runner=(B/'run_native462.py').read_text().replace('native-462-command.json','analysis/native464-regional-capture-command-001/native-464-command.json')
runner=runner.replace('native462-launch-guard-001','analysis/native464-regional-capture-command-001/launch-guard')
runner=runner.replace('native462-runner.log','native464-runner.log')
runner=runner.replace('native462','native464')
(root/'run_native464.py').write_text(runner)
# Build bound input document after helper outputs exist.
source151=B/'snapshots/source-cgls-regional-state-capture-151'
build151=Path(newbuild)/'source-build.json'
base462=B/'native-462-command.json'
case462=B/'solver/cgls-block-right-disabled-parity-462'
manifest={
 'schema':'numi.cardboard.native464.source-bindings.v1',
 'scope':'command preflight only; no native execution is claimed',
 'source151_snapshot':str(source151),
 'source151_frozen_manifest_sha256':hashlib.sha256((source151/'frozen-source.json').read_bytes()).hexdigest(),
 'source151_runtime_sha256':hashlib.sha256((source151/'matter/src/runtime.mm').read_bytes()).hexdigest(),
 'source151_regional_reader_overlay_sha256':hashlib.sha256((B/'analysis/base-cgls-regional-state-capture-overlay-151-001/overlay-manifest.json').read_bytes()).hexdigest(),
 'source151_build_receipt':str(build151),
 'source151_build_receipt_sha256':hashlib.sha256(build151.read_bytes()).hexdigest(),
 'source151_executable_sha256':hashlib.sha256(Path(newbuild+'/matter/numi-matter-cardboard-probe').read_bytes()).hexdigest(),
 'source151_metallib_sha256':hashlib.sha256(Path(newbuild+'/matter/shaders/NumiMatter.metallib').read_bytes()).hexdigest(),
 'source151_changed_sources':json.loads(build151.read_text()).get('changed_sources'),
 'source462_command':str(base462),
 'source462_command_sha256':hashlib.sha256(base462.read_bytes()).hexdigest(),
 'source462_case_receipt_sha256':hashlib.sha256((case462/'receipt.json').read_bytes()).hexdigest(),
 'source462_complete_history_parity_report':str(B/'analysis/source149-right-scale-disabled462-parity-001/report.json'),
 'compiled_mesh_package':str(case462/'output/compiled.nmatterpack'),
 'compiled_mesh_sha256':hashlib.sha256((case462/'output/compiled.nmatterpack').read_bytes()).hexdigest(),
 'compiled_package_topology_inventory_sha256':hashlib.sha256((root/'preflight/compiled-package-topology-inventory.json').read_bytes()).hexdigest(),
 'graph_layout_preflight_sha256':hashlib.sha256((root/'preflight/graph-layout.json').read_bytes()).hexdigest(),
 'capture_capacity_preflight_sha256':hashlib.sha256((root/'preflight/capture-capacity-preflight.json').read_bytes()).hexdigest(),
 'planned_case':newcase,
 'planned_regional_manifest':newcase+'/base-cgls.bin.regional.json',
 'planned_regional_payload':newcase+'/base-cgls.bin.regional.bin',
 'selection_expectation':'actual selection/health markers in capture are authoritative; based on corrected source463 audit, environment0 is active/eligible and environment1 quiet',
 'augmentation_admission':'always false in strict reader v2 unless consuming restart cycle is captured independently; source151 stores producer-cycle tag c, the consumer uses c+1, and current schema has no selected consumer-cycle field',
 'physical_settings':'all production args in source462 from --preset through --release-steps are retained exactly; source151 materials are byte-identical to source149; only source/build/case/output paths and diagnostic environment options differ',
}
(root/'source151-regional-capture-bindings.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')

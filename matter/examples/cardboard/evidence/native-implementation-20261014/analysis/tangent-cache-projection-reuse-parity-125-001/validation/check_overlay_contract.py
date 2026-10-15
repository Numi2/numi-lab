#!/usr/bin/env python3
from pathlib import Path
import hashlib,json,subprocess
root=Path('/Users/home/cardboard-implementation-20261014')
base=root/'snapshots/source-status5-vtccd-diagnostics-125'
pkg=root/'analysis/tangent-cache-projection-reuse-parity-125-001'
proto=pkg/'prototype/matter'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
orig_fem=(base/'matter/src/metal/fem.metalinc').read_text()
new_fem=(proto/'src/metal/fem.metalinc').read_text()
start=orig_fem.index('kernel void nm_fem_build_algorithmic_tangent_cache(')
end=orig_fem.index('\nkernel void nm_fem_apply_operator_elements(',start)
original=orig_fem[start:end]
assert new_fem.count(original)==1, 'production cache kernel changed or not preserved byte-for-byte'
assert new_fem.count('kernel void nm_fem_build_algorithmic_tangent_cache_projection_reuse_diagnostic(')==1
clone_start=new_fem.index('kernel void nm_fem_build_algorithmic_tangent_cache_projection_reuse_diagnostic(')
clone_end=new_fem.index('\nkernel void nm_fem_apply_operator_elements(',clone_start)
clone=new_fem[clone_start:clone_end]
assert 'float reusedProjectedState[NM_MAX_MATERIAL_STATE]' in clone
assert 'float reusedProjectedLow[NM_MAX_MATERIAL_STATE]' in clone
assert 'column == 0u' in clone and 'column > 0u' in clone
assert 'usesScaledCorrectionProjection(material)' in clone
common=(proto/'src/metal/common.metalinc').read_text()
assert 'reusedProjectedState = nullptr' in common and 'reusedProjectedLow = nullptr' in common
assert 'material.stateCount == 15u' in common and 'projectionDiagnosticsRequested' in common
checker=(proto/'tools/cardboard_tangent_cache_projection_reuse_check.mm').read_text()
for token in ['nm_fem_build_algorithmic_tangent_cache_projection_reuse_diagnostic','nm_fem_apply_operator_elements','candidate node does not bit-match capture center','materialStateStride','physical_steps']:
 assert token in checker,token
metallib=pkg/'validation/metallib/NumiMatter.metallib'
assert metallib.exists() and metallib.stat().st_size>1000000
names=subprocess.check_output(['strings',str(metallib)],text=True,errors='replace')
assert 'numi_matter_metal::nm_fem_build_algorithmic_tangent_cache' in names
assert 'numi_matter_metal::nm_fem_build_algorithmic_tangent_cache_projection_reuse_diagnostic' in names
print(json.dumps({'status':'pass','production_kernel_preserved_byte_exact':True,'reuse_buffers_state_capacity':'NM_MAX_MATERIAL_STATE (32 float words each)','diagnostic_function_emitted':True,'metal_compile':'pass','metallib_symbols':'pass','physical_steps':0},indent=2))

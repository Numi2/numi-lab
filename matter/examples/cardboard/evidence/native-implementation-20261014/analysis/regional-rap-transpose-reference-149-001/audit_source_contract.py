#!/usr/bin/env python3
"""Source-bound inventory of the exact source149 regional preconditioner path."""
import hashlib, json
from pathlib import Path
ROOT=Path('/Users/home/cardboard-implementation-20261014/snapshots/source-cgls-block-right-preconditioned-149')
OUT=Path('/Users/home/cardboard-implementation-20261014/analysis/regional-rap-transpose-reference-149-001')
FG=ROOT/'matter/src/metal/fgmres.metalinc'
RT=ROOT/'matter/src/runtime.mm'
BC=ROOT/'matter/src/metal/base_cgls.metalinc'
EXPECTED={
 str(FG):'8e0b9da3e28e7997eea9ee9db37e89a0bf3b9753d98967c80221e328e5da2af8',
 str(RT):'f859b1ee45390dd844fcbd883356b931ab71d2c36260f53a4b03600b04fb3959',
 str(BC):'127d05d6323085ad6abd160ff560b95008981e9421541e9677dddbb3fa253aec',
}
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def line_of(text, needle):
    for i,line in enumerate(text.splitlines(),1):
        if needle in line: return i
    raise AssertionError('source anchor missing: '+needle)
fg=FG.read_text(); rt=RT.read_text(); bc=BC.read_text()
for p,want in EXPECTED.items():
    assert sha(Path(p))==want,(p,sha(Path(p)),want)
anchors={
 'regional_build_rap':line_of(fg,'kernel void nm_fgmres_regional_build_rap('),
 'regional_factor_rap':line_of(fg,'kernel void nm_fgmres_regional_factor_rap('),
 'regional_project_rhs':line_of(fg,'kernel void nm_fgmres_regional_project_rhs('),
 'regional_solve_rap':line_of(fg,'kernel void nm_fgmres_regional_solve_rap('),
 'regional_apply_rap':line_of(fg,'kernel void nm_fgmres_regional_apply_rap('),
 'regional_apply_block_rap':line_of(fg,'kernel void nm_fgmres_regional_apply_block_rap('),
 'graph_build_patch_matrices':line_of(fg,'kernel void nm_fgmres_graph_build_patch_matrices('),
 'graph_validate_mask':line_of(fg,'kernel void nm_fgmres_graph_validate_mask('),
 'graph_factor_patches':line_of(fg,'kernel void nm_fgmres_graph_factor_patches('),
 'graph_solve_patches':line_of(fg,'kernel void nm_fgmres_graph_solve_patches('),
 'graph_apply_patches':line_of(fg,'kernel void nm_fgmres_graph_apply_patches('),
 'restart_form_mode':line_of(fg,'kernel void nm_fgmres_regional_restart_form_mode('),
 'restart_measure':line_of(fg,'kernel void nm_fgmres_regional_restart_measure('),
 'runtime_augmentation_selector':line_of(rt,'const bool useRestartAugmentedRegional ='),
 'runtime_right_scale_transpose':line_of(bc,'kernel void nm_fgmres_base_cgls_apply_right_scale('),
}
assert 'coarseCorrection += coefficient * p.xyz;' in fg
assert 'coarseResidual += coefficient * ap.xyz;' in fg
assert 'inverse.xyz * (residual.xyz - coarseResidual) + coarseCorrection' in fg
assert 'const float3 residual = basis[global].xyz - coarseResidual;' in fg
assert 'row0.w > 0.5f' in fg and 'fineCorrection + coarseCorrection' in fg
assert 'state.femRegionalRestartAugmentationEnabled &&' in rt
assert 'restartCycle > 0u' in rt
assert 'useRestartAugmentedRegional' in rt and 'regionalModeStride' in rt
assert 'const float3 result = transpose == 0u' in bc
assert 'value -= coefficients[coefficientBase + mode] *' in fg
assert 'originalMatrices[originalMatrixBase + row * n + column]' in fg
assert 'graphCorrection /= float(coverage);' in fg
assert 'environment * patchNodeCount + patchNode].xyz' in fg
report={
 'schema':'numi.matter.source149_regional_rap_source_contract.v1',
 'source_snapshot':str(ROOT),
 'files':{p:{'sha256':sha(Path(p)),'expected_sha256':v} for p,v in EXPECTED.items()},
 'anchors':anchors,
 'findings':{
  'regional_action':'source selector 3/4 block-RAP action is z = P c + B (r - AP c), c solves (P^T AP)c=P^T r; B is marked per-node 3x3 inverse block, falling back to componentwise inverse diagonal when row0.w <= 0.5.',
  'regional_transpose':'No regional RAP transpose kernel is present in source149. The independent algebraic transpose must reverse the composition and solve RAP^T; do not reuse RAP solve or assume symmetry.',
  'cgls_scope':'CGLS source149 uses a separate nodal block3 S: initialization applies S^T to A^T b, each forward product applies A(S p), and each residual gradient applies S^T A^T r. It does not invoke the regional RAP map inside CGLS.',
  'base_variant':'Runtime selects base mode count at restartCycle=0, but if restart augmentation storage is enabled, modeStride is augmentedModeCount even for the base view. A decoder must slice only modeCount columns while honoring that stride.',
  'augmented_variant':'At restartCycle>0 with augmentation enabled, runtime selects augmented P/AP, augmented RAP/LU/pivots/valid and modeCount. The extra mode carries (control, physicsSubstep, Newton, restartCycle) generation; mismatched generation is inadmissible.',
  'selector5_graph_action':'The graph path forms c=R^-1 P^T r, patch RHS E(r-AP c), and patch correction D_p times that RHS. D_p is the production one-refinement map from stored LU solve F_p and original patch matrix A_p: D_p b=F_p b+F_p(b-A_p F_p b). The apply kernel gathers each patch-node correction and divides each global node by nodeCoverage, then adds P c. Thus H=G D E and S=H+(P-H AP)R^-1 P^T. Its algebraic transpose is H^T+P R^-T(P^T-(AP)^T H^T), with H^T=E^T D^T G^T and D_p^T=2F_p^T-F_p^T A_p^T F_p^T in exact arithmetic. This graph map is source-audited but no selector5 same-generation capture is present in this package, so it is not implemented or qualified here.',
  'selector5_required_capture':'A rigorous selector5 replay needs one same-generation sidecar with graph patch descriptors [patchNodeOffset,nodeCount,matrixOffset,pivotOffset], sorted global patch-node IDs, graph row map, node-to-patch CSR offsets/values, per-node coverage, original pre-LU patch matrices, in-place LU, final row pivots, factor-valid words, effective P/AP, selected base or augmented RAP LU/pivots/validity, modeCount/modeStride, candidate node fixed/free tags, selected/healthy status, and source/world/program/device plus control/substep/microtick/micro-generation/Newton/dt and candidate-history digests. Patch RHS/coefficient dumps are input-specific and cannot stand in for this map. An arbitrary-vector native output would be needed to verify native forward action; source149 has no selector5 transpose kernel, so transpose parity remains CPU algebraic only.',
  'unsupported':'This package implements CPU references and tests only for selector 3/4 block-RAP and diagonal fallback. Selector 5 is explicitly unsupported pending the source-bound same-generation capture described above; no generic S test or another generation is used as selector5 evidence.'
 }
}
(OUT/'source-contract.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
print(json.dumps(report,indent=2,sort_keys=True))

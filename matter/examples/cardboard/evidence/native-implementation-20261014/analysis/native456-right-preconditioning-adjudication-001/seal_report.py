#!/usr/bin/env python3
"""Write source bindings and human-readable report for the 456/457 audit."""
from pathlib import Path
import json, hashlib, math
B=Path('/Users/home/cardboard-implementation-20261014')
R=Path('/Users/home/numi-cardboard-20261006')
OUT=B/'analysis/native456-right-preconditioning-adjudication-001'
PKGS={'456':B/'analysis/native-right-preconditioned-cgls-residual-replay-456',
      '457':B/'analysis/native-block-inverse-cgls-residual-replay-457'}
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
summary=json.loads((OUT/'summary.json').read_text())
helper_dir=B/'analysis/native-residual-direction-replay-419-001'
helper_source=helper_dir/'native_residual_direction.mm'
helper_program=helper_dir/'compiled-program.json'
source_snapshot=B/'snapshots/source-bounded-native-cgls-mask-repair-142'
fem=source_snapshot/'matter/src/metal/fem.metalinc'
shared=source_snapshot/'matter/include/numi/matter/shared.h'
frozen=source_snapshot/'frozen-source.json'
source_build=R/'build-cardboard-provenance-125/source-build.json'
source_bindings={
 'source_snapshot':str(source_snapshot),'frozen_source_manifest_sha256':sha(frozen),
 'native_helper_source':{'path':str(helper_source),'sha256':sha(helper_source)},
 'native_helper_program_manifest':{'path':str(helper_program),'sha256':sha(helper_program)},
 'native_helper_executable':{'path':str(helper_dir/'build/native_residual_direction'),'sha256':sha(helper_dir/'build/native_residual_direction')},
 'source125_build_receipt':{'path':str(source_build),'sha256':sha(source_build),'matches_prediction':sha(source_build)==summary['packages']['456']['source_build_sha256']},
 'compiled_world':{'path':'/Users/home/cardboard-implementation-20261014/solver/two-stage-first-failure-rich-fd-419/output/compiled.nmatterpack','sha256':sha('/Users/home/cardboard-implementation-20261014/solver/two-stage-first-failure-rich-fd-419/output/compiled.nmatterpack')},
 'matter_sources':{
  'fem.metalinc':{'path':str(fem),'sha256':sha(fem),'lines':{'armijo_prepare_candidate':'4469-4512','mechanical_residual':'2730-2824'}},
  'shared.h':{'path':str(shared),'sha256':sha(shared),'lines':{'NMFEMNodeStateGPU':'1139-1147'}}},
 'capture_packages':{},
 'CGLS_direction_study_inputs':{
  '146-001':{'path':str(B/'analysis/cgls-right-preconditioning-study-146-001'),'report_sha256':sha(B/'analysis/cgls-right-preconditioning-study-146-001/report.json'),'manifest_sha256':sha(B/'analysis/cgls-right-preconditioning-study-146-001/manifest.json')},
  '146-002':{'path':str(B/'analysis/cgls-right-preconditioning-study-146-002'),'report_sha256':sha(B/'analysis/cgls-right-preconditioning-study-146-002/report.json'),'manifest_sha256':sha(B/'analysis/cgls-right-preconditioning-study-146-002/manifest.json')}
 },
 'native_helper_source_semantics':{
  'evaluateResidual_and_signed_direction':'native_residual_direction.mm:473-490',
  'geometry_force_residual_pipeline':'native_residual_direction.mm:526-548',
  'accepted_history_and_nodes_immutable_checks':'native_residual_direction.mm:547-557',
  'constraint_mask_metric_definition':'native_residual_direction.mm:593-627',
  'alpha_grid_and_side_logging':'native_residual_direction.mm:760-771',
  'payload_layout':'native_residual_direction.mm:114-150'},
 'residual_definition':'nm_fem_build_mechanical_residual computes free-node impulse residual dt*(internal force + configured gravity + external force) - mass*(current velocity - accepted velocity); this test sets hasExternalForces=0 and passes no contact residual term. xyz components are normed; fixed constraint values 1 or 3 excluded.',
 'direction_sign':'sample side 0 uses the exported direction unchanged; side 1 negates xyz; positive alpha is added to candidate velocity, then displacement is reconstructed from accepted displacement and dt times the new velocity.',
 'fixed_mask':'candidate/sample restAndFixed.w exactly 1.0 or 3.0 is fixed; all other legal 0.0 rows are included. Captured state has 846 free and 36 fixed nodes per environment (36 tag1, zero tag3).'
}
for label,base in PKGS.items():
 p=json.loads((base/'prediction.json').read_text())
 r=json.loads((base/'receipt.json').read_text())
 files=json.loads((base/'manifest.json').read_text())['files']
 inputs={}
 for run,cmd in p['commands'].items(): inputs[run]={'path':cmd[2],'sha256':sha(cmd[2])}
 source_bindings['capture_packages'][label]={
  'path':str(base),'prediction_sha256':sha(base/'prediction.json'),'receipt_sha256':sha(base/'receipt.json'),'manifest_sha256':sha(base/'manifest.json'),
  'native_commands':len([k for k in r['commands'] if k.endswith('.native')]),'preflight_commands':len([k for k in r['commands'] if k.endswith('.preflight')]),
  'physical_transactions':r['physical_transactions'],'source_and_helper_inputs_unchanged':r['source_and_helper_inputs_unchanged'],
  'output_file_count':len(files),'output_payloads':inputs,'helper_program_sha256':p['helper_program_sha256'],'source_build_sha256':p['source125_build_sha256']}
(OUT/'source_bindings.json').write_text(json.dumps(source_bindings,indent=2,allow_nan=False)+'\n')
# Selected comparison rows. Values in exact report remain linked to full CSV.
choices=[
 ('456','none_fp64_iter064','None, 64'),('456','column_fp64_iter064','Column-norm, 64'),('456','column_fp64_iter128','Column-norm, 128'),
 ('456','block3_fp64_iter064','Gram-block inverse square root, 64'),('456','block3_fp64_iter128','Gram-block inverse square root, 128'),
 ('457','diag_Aii_fp64_iter064','Diagonal inverse Aii, 64'),('457','diag_Aii_fp64_iter128','Diagonal inverse Aii, 128'),
 ('457','block_Aii_fp64_iter064','Nonsymmetric 3x3 Aii inverse, 64'),('457','block_Aii_fp64_iter128','Nonsymmetric 3x3 Aii inverse, 128')]
lines=['# Native 456/457 nonlinear residual adjudication','',
'This CPU-only analysis recomputes all 300 environment/sample residual norms from the immutable native outputs. The GPU helper evaluated candidate directions through production FEM preparation, geometry validation, internal-force projection, and mechanical-residual kernels. No physical timestep was advanced: both receipts record zero physical transactions.',
'', '## Binding and computation','',
'456 and 457 each contain 10 successful native residual calls plus 10 successful payload preflights. There were no timeouts, all receipt command exit codes are zero, and both runners record that source/helper inputs were unchanged. Every output hash and byte count matches its capture manifest. The two packages use the same source125 build receipt, compiled world (SHA-256 `'+source_bindings['compiled_world']['sha256']+'`), native helper executable (SHA-256 `'+source_bindings['native_helper_executable']['sha256']+'`), dimensions `2×882 nodes ×1920 tets`, and `dt=9.999999747378752e-5 s`.',
'', 'Every input payload has bitwise-identical candidate, RHS, accepted material history, cached tangent and validity flags, parameters, captured fresh action, tet audit, and accepted nodes across all 20 run payloads. Only the exported correction-direction segment differs. Alpha-zero residual bytes and candidate-node bytes are identical across all runs. The source and binary identities, all payload paths/hashes, and exact helper/source lines are in `source_bindings.json`.',
'', 'The helper sets `metrics.z=epsilon`. It negates the exported direction only for `side=1`; therefore `side=0` is the positive correction direction. The production candidate kernel adds `alpha*correction` to velocity, then uses accepted displacement plus `dt*new_velocity` for reference-displacement FEM. It runs geometry validation, material/internal-force evaluation, and the production mechanical-residual kernel. The helper verifies that accepted nodes and accepted material history remain byte-identical after each sample.',
'', 'The reported residual is the xyz norm of the FEM mechanical impulse residual (Ns): production computes `dt*(internal force + gravity + external force) - mass*(candidate velocity - accepted velocity)`. The helper sets `hasExternalForces=0` and this residual kernel does not include contact contributions. This is an isolated FEM residual comparison, not the full-contact Armijo merit. The fixed-row mask is taken from `restAndFixed.w`: exact values 1 or 3 are fixed and excluded; all other legal zero-tag rows are included. Each environment has 846 free rows and 36 fixed rows. In this captured payload all fixed rows are tag 1; none is tag 3.',
'', 'For every one of the 300 samples, the two environment status codes are `[0,0]`, and the helper’s accepted-history-unchanged marker is true. All output floats are finite. Environment 1 has an all-zero direction segment and its node/residual buffers are bitwise unchanged across every sample; the measured change is in environment 0.',
'', '## Residual results','',
'Norms are recomputed from the stored FP32 residual vectors using `math.fsum` for squared terms. The baseline alpha-zero free residual is `1.8444380407784322e-4 Ns` in env0 and `6.780005479517216e-8 Ns` in env1. Table entries are env0 free residuals in Ns; the complete alpha/sign table for both environments, status/history markers and output hashes is in `residuals.csv`.',
'', '| Direction source | alpha 1 | alpha 0.5 | alpha 0.25 | alpha 2 |', '|---|---:|---:|---:|---:|']
for label,name,title in choices:
 row=summary['runs'][label][name]
 v=row['positive_l2_Ns_by_alpha']
 lines.append(f"| {title} (`{label}/{name}`) | {v['1']} | {v['0.5']} | {v['0.25']} | {v['2']} |")
lines += ['',
'At alpha 1, unpreconditioned CGLS64 gives the smallest env0 residual, `4.8387813e-5 Ns`. Column-norm64 gives `5.6735871e-5`, Gram-block64 `5.8870630e-5`, diagonal inverse Aii64 `6.1015640e-5`, and nonsymmetric 3×3 Aii inverse64 `5.8870550e-5`. All reduce the center residual at that tested positive alpha, but right scaling does not beat the unpreconditioned direction at alpha 1.',
'',
'At alpha 0.5, right-scaled directions are modestly better than the unpreconditioned direction (`9.25–9.64e-5 Ns` versus `9.99125e-5 Ns`). At alpha 0.25, the block and column variants are also modestly better (`1.379–1.388e-4 Ns` versus `1.41243e-4 Ns`). At alpha 2, every positive direction is at or above the center residual; the block directions are substantially worse than the unpreconditioned direction. Every negative-side sample increases the residual above the center. The alpha grid’s lowest residual for each direction is alpha 1, among tested points.',
'',
'Increasing CGLS from 64 to 128 does not improve the tested alpha-1 nonlinear residual for any compared family: column-norm goes from `5.67359e-5` to `6.07912e-5 Ns`, Gram-block from `5.88706e-5` to `6.00012e-5 Ns`, diagonal Aii inverse from `6.10156e-5` to `6.04056e-5 Ns`, and 3×3 Aii inverse from `5.88705e-5` to `6.00009e-5 Ns`. Some alpha 0.5/2 comparisons vary, so there is no uniform 128-iteration benefit. The 128-iteration candidates were generated offline and passed through the native residual helper; they are not production CGLS runs, since the source-bound native CGLS cap is 64.',
'',
'## Recommendation and limits','',
'These results do not justify enabling diagonal or 3×3 Aii right scaling by default. They show a valid FEM residual decrease at tested positive alphas, but the unpreconditioned direction is stronger at alpha 1 and lower-alphas only slightly favor some scaled candidates. Crucially, this helper omits contact force/residual terms and does not perform the full-contact selector, rollback, or nonlinear terminal certificate. Treat this as candidate-direction evidence for one captured state only. If pursued, evaluate the same candidates through full production Armijo with unchanged physical operator, history/rollback, and strict final-certificate gates against the unpreconditioned control.',
'',
'The 146 studies also bind the direction generation: `column` is `S[j,j]=1/||A[:,j]||₂`; `block3` is per-free-node SPD `Sₙ=(AₙₙᵀAₙₙ)^(-1/2)`; 457 `diag_Aii` is signed `1/A[j,j]`; 457 `block_Aii` is the nonsymmetric principal-block inverse `Sₙ=(Aₙₙ)^(-1)` with adjoint recurrence applying `Sₙᵀ`. No operator shift or regularization was applied. The prior 146-001 report’s operator sign/unit typo is corrected in 146-002; this analysis consumes its exported direction bytes without changing the captured matrix.',
'', 'This does not establish a complete coupon, physical calibration, or production convergence. It is a zero-transaction FEM residual replay at one captured state; 128 is exploratory; no full contact or rigid-body residual is present.']
(OUT/'report.md').write_text('\n'.join(lines)+'\n')
readme='''# Native 456/457 residual adjudication\n\nCPU-only independent analysis of native residual replay packages.\n\nRun `python3 adjudicate.py` to verify capture manifests and receipts, check input payload equivalence, and regenerate `summary.json` and `residuals.csv`. Run `python3 seal_report.py` afterward to regenerate this report, source bindings, and the local artifact manifest. Both scripts read immutable 456/457 capture artifacts; neither launches a native/GPU command.\n\n`residuals.csv` contains all 300 observations. `summary.json` records per-run summaries and input identity. `source_bindings.json` identifies the exact native helper, source snapshot, compiled world, 146 direction studies, and every payload. `report.md` states the bounded result and its limits.\n'''
(OUT/'README.md').write_text(readme)
files={str(p.relative_to(OUT)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='manifest.json'}
(OUT/'manifest.json').write_text(json.dumps({'schema':'numi.cardboard.analysis_manifest.v1','files':files},indent=2,allow_nan=False)+'\n')
print('report sha',sha(OUT/'report.md'))
print('summary sha',sha(OUT/'summary.json'))
print('csv sha',sha(OUT/'residuals.csv'))
print('source bindings sha',sha(OUT/'source_bindings.json'))
print('manifest sha',sha(OUT/'manifest.json'))
print('manifest file count',len(files))

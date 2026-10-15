#!/usr/bin/env python3
"""Independent source-bound recheck of root373v3 dynamic correction scaling.

CPU-only. Replays one captured rejected Newton direction; it does not alter
source, run a physical transaction, or qualify the general solver.
"""
from pathlib import Path
import hashlib, importlib.util, json, sys, datetime
import numpy as np

B=Path('/Users/home/cardboard-implementation-20261014')
CAP=B/'solver/reset-augmentation-first-failure-capture-360/operator-capture'
SNAP=B/'snapshots/source-boundary-reset-order-101'
COMMON=SNAP/'matter/src/metal/common.metalinc'
FROZEN=SNAP/'frozen-source.json'
LINER=SNAP/'matter/materials/hajali2009_liner_hill_bilinear_fit.nmatter'
MEDIUM=SNAP/'matter/materials/hajali2009_medium_hill_bilinear_fit.nmatter'
REF=B/'analysis/bilinear15-fp64-replay-267/bilinear15_reference.py'
FD=B/'analysis/source101-capture360-directional-fd-001/compare_directional_fd.py'
AUDIT=B/'analysis/source101-independent-assembled-action-373-003'
REPORT_IN=AUDIT/'report.json'
VECTORS=AUDIT/'reference-vectors.npz'
OUT=Path(__file__).resolve().parent
CAP_SHA='548dce37fbe5a28a609dd6bc45318926072c878774ea0caffa3237d925cebb33'
LIMIT=2.0e-8
RESIDUAL_SCALE=np.array([1e4]*6+[1.0]*6+[1e4]*3,dtype=np.float64)

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''): h.update(block)
    return h.hexdigest()

def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    if not spec or not spec.loader: raise RuntimeError(f'cannot load {path}')
    mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod)
    return mod

def main():
    ref=load(REF,'v3review_ref');fd=load(FD,'v3review_fd')
    frozen=json.loads(FROZEN.read_text())
    assert frozen['source_sha256']['matter/src/metal/common.metalinc']==sha(COMMON)
    common_text=COMMON.read_text()
    required=(
        'stateScale[column] = max(max(abs(acceptedState[column]),',
        'abs(candidateState[column])), scaleFloor);',
        'scaledCorrectionNorm = max(scaledCorrectionNorm, abs(scaledCorrection));',
        'parameters[material.parameterOffset + 26u]',
        'scaledCorrectionNorm <= 2.0e-8f',
    )
    if not all(token in common_text for token in required):
        raise RuntimeError('production dynamic state-scale/threshold source check failed')
    assert sha(CAP)==CAP_SHA
    records=[json.loads(line,parse_int=fd.parse_int) for line in CAP.read_text().splitlines() if line.strip()]
    header=records[0]
    assert header['record']=='header' and (header['root'],header['newton_iteration'])==(2,23)
    assert records[-1].get('record')=='integrity' and records[-1].get('production_inputs_bitwise_unchanged') is True
    tets=sorted([r for r in records if r.get('record')=='tetrahedron' and int(r['environment'])==0],key=lambda r:int(r['tet_id']))
    assert len(tets)==1920
    paper=[r for r in tets if int(r['material_index'])<2]
    assert len(paper)==1728
    flat=fd.f32(header['parameters_by_environment'][0])
    p=np.asarray([flat[int(r['material_index'])*28:(int(r['material_index'])+1)*28] for r in paper],dtype=np.float64)
    old=np.asarray([fd.f32(r['center']['accepted_state']) for r in paper],dtype=np.float64)
    F=np.asarray([fd.pair(r['center'],'F_local') for r in paper],dtype=np.float64)
    z,iterations,residual,ok,fail=fd.project_resilient(ref,p,F,old)
    if not bool(np.all(ok)): raise RuntimeError(f'initial independent projections failed: {fail}')

    refinement=[]
    for iteration in range(5):
        r=np.asarray(ref.residual(p,F,old,z),dtype=np.float64)
        J=np.asarray(ref.jacobian(p,F,old,z),dtype=np.float64)
        direct=np.linalg.solve(J,-r[...,None])[...,0]
        # This is the production coordinate scale: accepted state, current
        # candidate state, and parameter 26 (FB epsilon) as the floor.
        state_scale=np.maximum(np.maximum(np.abs(old),np.abs(z)),p[:,26,None])
        direct_y=direct/state_scale
        direct_norm=np.max(np.abs(direct_y),axis=1)

        # Reproduce solveScaledLocalMaterialSystem's algebra in FP64: scale
        # columns by stateScale, each row by its weighted L1 norm, then solve.
        row_scale=np.sum(np.abs(J)*state_scale[:,None,:],axis=2)
        scaled_J=J*state_scale[:,None,:]/row_scale[:,:,None]
        scaled_rhs=-r/row_scale
        equilibrated_y=np.linalg.solve(scaled_J,scaled_rhs[...,None])[...,0]
        equilibrated_norm=np.max(np.abs(equilibrated_y),axis=1)
        max_y_difference=float(np.max(np.abs(direct_y-equilibrated_y)))
        selected=np.flatnonzero(equilibrated_norm>LIMIT)
        refinement.append({
            'iteration':iteration,
            'rows_above_2e_minus8':int(len(selected)),
            'max_dynamic_correction_direct_solve':float(np.max(direct_norm)),
            'max_dynamic_correction_equilibrated_solve':float(np.max(equilibrated_norm)),
            'direct_max_tet_id':int(paper[int(np.argmax(direct_norm))]['tet_id']),
            'equilibrated_max_tet_id':int(paper[int(np.argmax(equilibrated_norm))]['tet_id']),
            'max_componentwise_delta_over_scale_difference':max_y_difference,
        })
        if not len(selected): break
        if iteration>=4: raise RuntimeError('bounded one-Newton continuation exceeded its cap')
        baseline=np.max(np.abs(r[selected]*RESIDUAL_SCALE),axis=1)
        alpha=np.ones(len(selected),dtype=np.float64)
        pending=np.ones(len(selected),dtype=bool)
        next_z=z[selected].copy()
        for backtrack in range(24):
            candidate=z[selected]+alpha[:,None]*direct[selected]
            trial=np.asarray(ref.residual(p[selected],F[selected],old[selected],candidate),dtype=np.float64)
            trial_norm=np.max(np.abs(trial*RESIDUAL_SCALE),axis=1)
            accept=pending&np.isfinite(trial_norm)&(trial_norm<baseline)
            next_z[accept]=candidate[accept];pending[accept]=False
            if not pending.any(): break
            alpha[pending]*=0.5
        if pending.any(): raise RuntimeError('Newton continuation failed residual descent')
        z[selected]=next_z

    r=np.asarray(ref.residual(p,F,old,z),dtype=np.float64)
    J=np.asarray(ref.jacobian(p,F,old,z),dtype=np.float64)
    direct=np.linalg.solve(J,-r[...,None])[...,0]
    state_scale=np.maximum(np.maximum(np.abs(old),np.abs(z)),p[:,26,None])
    direct_norm=np.max(np.abs(direct)/state_scale,axis=1)
    row_scale=np.sum(np.abs(J)*state_scale[:,None,:],axis=2)
    scaled_J=J*state_scale[:,None,:]/row_scale[:,:,None]
    scaled_rhs=-r/row_scale
    equilibrated_y=np.linalg.solve(scaled_J,scaled_rhs[...,None])[...,0]
    equilibrated_norm=np.max(np.abs(equilibrated_y),axis=1)
    scaled_residual=np.max(np.abs(r*RESIDUAL_SCALE),axis=1)
    vectors=np.load(VECTORS)
    stored_correction=vectors['paper_final_scaled_correction']
    stored_residual=vectors['paper_projection_residual']
    correction_vector_max_diff=float(np.max(np.abs(direct_norm-stored_correction)))
    residual_vector_max_diff=float(np.max(np.abs(scaled_residual-stored_residual)))
    if correction_vector_max_diff>1e-20 or residual_vector_max_diff>1e-20:
        raise RuntimeError('replayed vectors do not match sealed source373v3 vectors')
    maximum_residual=float(np.max(scaled_residual))
    maximum_native_raw_residual_upper_bound=10.0*maximum_residual
    report={
        'schema':'numi.cardboard.source101-dynamic-correction-review.v1',
        'classification':'independent CPU recheck of an FP64 reference continuation; not a native projection pass or solver qualification',
        'bindings':{
            'capture':{'path':str(CAP),'sha256':sha(CAP)},
            'common_metalinc':{'path':str(COMMON),'sha256':sha(COMMON)},
            'frozen_source_manifest':{'path':str(FROZEN),'sha256':sha(FROZEN)},
            'liner_material':{'path':str(LINER),'sha256':sha(LINER)},
            'medium_material':{'path':str(MEDIUM),'sha256':sha(MEDIUM)},
            'reference':{'path':str(REF),'sha256':sha(REF)},
            'capture_helper':{'path':str(FD),'sha256':sha(FD)},
            'source373v3_report':{'path':str(REPORT_IN),'sha256':sha(REPORT_IN)},
            'source373v3_vectors':{'path':str(VECTORS),'sha256':sha(VECTORS)},
        },
        'source_check':{
            'production_scale_formula':'max(abs(acceptedState[column]),abs(candidateState[column]),scaleFloor)',
            'production_scale_floor':'material parameter index 26 (FB epsilon)',
            'production_scaled_correction':'max absolute solution in dimensionless state coordinates',
            'production_threshold':LIMIT,
            'dynamic_formula_equivalence':'J*delta=-r; delta=stateScale*y. Production row equilibration divides each equation by sum(abs(J[row,column])*stateScale[column]); this leaves y and its infinity norm unchanged.',
            'common_source_hash_is_in_frozen_source_manifest':True,
        },
        'independent_replay':{
            'paper_rows':len(paper),
            'center_reference_projection_rows':int(np.all(ok)),
            'center_refinement':refinement,
            'final_max_direct_dynamic_correction':float(np.max(direct_norm)),
            'final_max_equilibrated_dynamic_correction':float(np.max(equilibrated_norm)),
            'direct_and_equilibrated_pass_count':int(np.sum((direct_norm<=LIMIT)&(equilibrated_norm<=LIMIT))),
            'rows_above_2e_minus8':int(np.sum(equilibrated_norm>LIMIT)),
            'max_correction_tet_id':int(paper[int(np.argmax(equilibrated_norm))]['tet_id']),
            'maximum_scaled_reference_residual':maximum_residual,
            'scaled_reference_residual_rows_above_2e_minus10':int(np.sum(scaled_residual>2e-10)),
            'stored_final_correction_vector_max_abs_difference':correction_vector_max_diff,
            'stored_residual_vector_max_abs_difference':residual_vector_max_diff,
            'uniform_residual_scale':10.0,
            'conservative_production_raw_residual_upper_bound':maximum_native_raw_residual_upper_bound,
            'minimum_compensated_production_residual_limit':2e-8,
            'residual_bound_note':'All authored paper implicit residual equations multiply the independent residual by the same residual_scale=10. Since the reference weighted norm uses weights >=1, 10*max_weighted_reference_residual conservatively bounds production unweighted max residual. This bound is below the minimum 2e-8 compensated projection tolerance.',
        },
        'original_report_adjudication':{
            'dynamic_correction_formula_matches_production_coordinates':True,
            'numerical_gate_passed_in_independent_fp64_reference':bool(np.max(equilibrated_norm)<=LIMIT),
            'native_projection_gate_was_executed':False,
            'assembled_action_is_comparison_only':True,
            'full_solver_or_physical_qualification':False,
            'known_limitations':['FP64 reference continuation is not a native float32 scaled solve.','No absolute noise floor was measured.','The 0.742% action match is one rejected Newton direction only and is not whole-solver qualification.'],
        },
        'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    (OUT/'review.json').write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps({'review':str(OUT/'review.json'),'report_sha256':sha(OUT/'review.json'),
                      'max_direct':float(np.max(direct_norm)),'max_equilibrated':float(np.max(equilibrated_norm)),
                      'max_residual':maximum_residual,'max_vector_diffs':[correction_vector_max_diff,residual_vector_max_diff]},indent=2))

if __name__=='__main__': main()

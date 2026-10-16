#!/usr/bin/env python3
"""Bounded all-nine local-F basis finite-difference audit for native531."""
from pathlib import Path
import hashlib,json,os,sys,time,datetime
import numpy as np
Q=Path('/Users/n/cardboard-implementation-20261014-host-macmini')
BASE=Q/'analysis/native531-strict-fp64-replay-001/layers'
REFINE=Q/'analysis/native531-fp64-fd-refinement-002'
PREP=Q/'analysis/source198-strict-layer-replay-preparation-001'
CASE=Q/'solver/control17-root17-newton34-replay-531'
OUT=Q/'analysis/native531-fp64-ninebasis-003'
S=Q/'snapshots/source-contact-damped-memory-198-001/matter/tools'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
GRID=list(range(10,25))
GATE=1e-6

def row_metrics(actual, expected):
    a=np.asarray(actual,dtype=np.float64).reshape(len(actual),-1)
    b=np.asarray(expected,dtype=np.float64).reshape(len(expected),-1)
    finite=np.isfinite(a).all(axis=1)&np.isfinite(b).all(axis=1)
    err=np.full(len(a),np.nan); scale=np.full(len(a),np.nan); rel=np.full(len(a),np.nan)
    err[finite]=np.linalg.norm(a[finite]-b[finite],axis=1)
    scale[finite]=np.linalg.norm(b[finite],axis=1)
    nz=finite&(scale>0.0)
    rel[nz]=err[nz]/scale[nz]
    return rel,err,scale,finite

def stats(values,mask):
    v=np.asarray(values)[np.asarray(mask,dtype=bool)&np.isfinite(values)]
    return {'count':int(v.size),'max':float(v.max()) if v.size else None,'median':float(np.median(v)) if v.size else None}

def main():
    for key,expected in json.loads((REFINE/'manifest.json').read_text())['files_sha256'].items():
        assert sha(REFINE/key)==expected,key
    old_manifest=json.loads((BASE/'manifest.json').read_text())
    for key,expected in old_manifest['files_sha256'].items(): assert sha(BASE/key)==expected,key
    assert sha(BASE/'manifest.json')=='1db4342cdf9b866cd8280209a86cdc7cb70cd712ddb8ea68b2d42b0eb89253da'
    assert sha(REFINE/'manifest.json')=='f949f99199d542dc4b4ba78d6ae85a53aa663383d6037827f2487af242b6fc52'
    pred=json.loads((CASE/'prediction.json').read_text())
    capture=Path(pred['environment']['NM_FEM_OPERATOR_FD_CAPTURE_PATH'])
    with np.load(BASE/'arrays.npz') as b:
        material=b['material'].copy()
        Fcenter=b['F_center'].copy()
        old=b['accepted_paper_history'].copy()
        zcenter=b['s0_reference_state'].copy()
        center_valid=b['s0_valid'].copy()
        center_residual=b['s0_scaled_residual'].copy()
        center_correction=b['s0_dynamic_correction'].copy()
        center_C=b['reference_C'].copy()
        center_stress=b['s0_reference_stress'].copy()
    paper=np.flatnonzero(material!=2)
    assert len(paper)==1728 and zcenter.shape==(1728,15) and center_valid.all()
    sys.path.insert(0,str(PREP))
    os.environ['PYTHONPATH']=str(S)
    os.environ['PYTHONDONTWRITEBYTECODE']='1'
    for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
    import strict_layers
    import cardboard_bilinear_fp64_reference as ref
    import cardboard_cached_operator_audit as geometry
    header,nodes,tets,integrity=geometry.capture_records(capture)
    bindings=geometry.bind_inputs(capture,CASE/'prediction.json',CASE/'receipt.json',CASE/'stderr.log',0)
    geometry.validate_and_assemble(header,nodes,tets,integrity,0)
    strict_layers.validate_identity(header,nodes,tets,bindings)
    nodes=sorted((r for r in nodes if r['environment']==0),key=lambda r:r['node_id'])
    tets=sorted((r for r in tets if r['environment']==0),key=lambda r:r['tet_id'])
    specs={m['index']:m for m in header['materials']}
    pt=strict_layers.f32(header['parameters_by_environment'][0])
    expected={0:'hajali2009_liner_hill_bilinear_fit',1:'hajali2009_medium_hill_bilinear_fit',2:'starch2007_finite_glue_elastic'}
    assert set(specs)==set(expected) and all(specs[i]['name']==v for i,v in expected.items())
    assert all(tuple(specs[i]['parameter_names'])==strict_layers.PAPER_NAMES for i in (0,1))
    mats=np.array([t['material_index'] for t in tets],dtype=np.int32)
    paper_from_capture=np.flatnonzero(mats!=2)
    assert np.array_equal(paper,paper_from_capture)
    p=np.stack([pt[specs[mats[i]]['parameter_offset']:specs[mats[i]]['parameter_offset']+28] for i in paper])
    old_capture=np.stack([strict_layers.f32(tets[i]['center']['accepted_state']) for i in paper])
    F_capture=np.stack([strict_layers.pair(tets[i]['center'],'F_local') for i in paper])
    assert np.array_equal(old,old_capture), 'accepted history words changed vs parent layer'
    assert np.array_equal(Fcenter[paper],F_capture), 'captured F words changed vs parent layer'
    dt=float(strict_layers.f32(header['dt_seconds']))
    # Confirm parent strict center projections and their gates from same exact inputs.
    z,it,rn,cn,valid,reasons,passes=strict_layers.project(p,F_capture,old)
    assert valid.all() and rn.max()<=2e-12 and cn.max()<=2e-11
    center_state_delta=np.max(np.abs(z-zcenter))
    if not np.array_equal(z,zcenter):
        raise AssertionError('deterministic strict center projection differs from parent reference state')
    center_branch=strict_layers.branch(p,old,z)
    center_branch=center_branch.astype(np.int8)
    n=len(paper)
    basis=np.zeros((9,n,3,3),dtype=np.float64)
    tangent=np.empty((9,n,3,3),dtype=np.float64)
    tangent_parent_error=np.empty((9,n),dtype=np.float64)
    tangent_norm=np.empty((9,n),dtype=np.float64)
    for col in range(9):
        h=np.zeros((n,3,3),dtype=np.float64);h.reshape(n,9)[:,col]=1.0
        basis[col]=h
        tangent[col]=strict_layers.strict.tangent(ref,p,F_capture,old,z,h)
        tangent_parent_error[col]=np.linalg.norm((tangent[col].reshape(n,9)-center_C[paper,:,col]),axis=1)
        tangent_norm[col]=np.linalg.norm(tangent[col].reshape(n,9),axis=1)
    assert np.isfinite(tangent).all()
    # Arrays are indexed [power_index, basis_column, tet].
    shape=(len(GRID),9,n)
    centered_rel=np.full(shape,np.nan);centered_abs=np.full(shape,np.nan);reference_norm=np.full(shape,np.nan)
    plus_rel=np.full(shape,np.nan);minus_rel=np.full(shape,np.nan)
    plus_valid=np.zeros(shape,dtype=bool);minus_valid=np.zeros(shape,dtype=bool)
    plus_iters=np.full(shape,-1,dtype=np.int16);minus_iters=np.full(shape,-1,dtype=np.int16)
    plus_residual=np.full(shape,np.nan);minus_residual=np.full(shape,np.nan)
    plus_correction=np.full(shape,np.nan);minus_correction=np.full(shape,np.nan)
    plus_branch=np.zeros(shape+(3,),dtype=np.int8);minus_branch=np.zeros(shape+(3,),dtype=np.int8)
    smooth=np.zeros(shape,dtype=bool);plus_keep=np.zeros(shape,dtype=bool);minus_keep=np.zeros(shape,dtype=bool)
    branch_cross=np.zeros(shape,dtype=bool);zero_action=np.zeros((9,n),dtype=bool)
    progress=OUT/'progress.jsonl';OUT.mkdir(parents=True, exist_ok=True)
    with progress.open('x') as log:
        total_start=time.monotonic()
        for pi,power in enumerate(GRID):
            eps=2.0**(-power)
            for col in range(9):
                H=basis[col]
                ends=[]
                for sign in (1.0,-1.0):
                    F=F_capture+sign*eps*H
                    zz,ii,rr,cc,ok,why,provenance=strict_layers.project(p,F,old)
                    pp=np.full((n,3,3),np.nan,dtype=np.float64)
                    if np.any(ok): pp[ok]=ref.stress(p[ok],F[ok],zz[ok])
                    bc=np.zeros((n,3),dtype=np.int8)
                    if np.any(ok): bc[ok]=strict_layers.branch(p[ok],old[ok],zz[ok]).astype(np.int8)
                    ends.append((zz,ii,rr,cc,ok,why,pp,bc))
                zp,ip,rp,cp,vp,wp,Pp,bp=ends[0]
                zm,im,rm,cm,vm,wm,Pm,bm=ends[1]
                plus_valid[pi,col]=vp;minus_valid[pi,col]=vm
                plus_iters[pi,col]=ip;minus_iters[pi,col]=im
                plus_residual[pi,col]=rp;minus_residual[pi,col]=rm
                plus_correction[pi,col]=cp;minus_correction[pi,col]=cm
                plus_branch[pi,col]=bp;minus_branch[pi,col]=bm
                pkeep=vp&np.all(bp==center_branch,axis=1)
                mkeep=vm&np.all(bm==center_branch,axis=1)
                plus_keep[pi,col]=pkeep;minus_keep[pi,col]=mkeep
                both_valid=vp&vm
                branch_cross[pi,col]=both_valid&~(pkeep&mkeep)
                sm=center_valid&pkeep&mkeep
                smooth[pi,col]=sm
                zero_action[col]=tangent_norm[col]==0.0
                sec=(Pp-Pm)/(2.0*eps)
                rel,ab,norm,fin=row_metrics(sec,tangent[col])
                good=sm&fin&(norm>0.0)
                centered_rel[pi,col,good]=rel[good]
                centered_abs[pi,col,good]=ab[good]
                reference_norm[pi,col,good]=norm[good]
                onp=(Pp-center_stress[paper])/(eps)
                onm=(center_stress[paper]-Pm)/(eps)
                rplus,_,_,fplus=row_metrics(onp,tangent[col])
                rminus,_,_,fminus=row_metrics(onm,tangent[col])
                goodp=pkeep&center_valid&fplus&(tangent_norm[col]>0)
                goodm=mkeep&center_valid&fminus&(tangent_norm[col]>0)
                plus_rel[pi,col,goodp]=rplus[goodp]
                minus_rel[pi,col,goodm]=rminus[goodm]
            percol=[]
            for col in range(9):
                smooth_rows=smooth[pi,col]
                goodrel=centered_rel[pi,col]
                finite=np.isfinite(goodrel)
                percol.append({'basis_column':col,'smooth_same_branch_rows':int(smooth_rows.sum()),
                    'branch_cross_rows':int(branch_cross[pi,col].sum()),
                    'invalid_plus_rows':int((~plus_valid[pi,col]).sum()),
                    'invalid_minus_rows':int((~minus_valid[pi,col]).sum()),
                    'relative_at_1e-6_rows':int(np.sum(finite&(goodrel<=GATE))),
                    'max_relative_on_smooth_nonzero':float(np.max(goodrel[finite])) if finite.any() else None})
            log.write(json.dumps({'power':power,'epsilon':eps,'completed_basis_columns':9,'directions':percol,'elapsed_seconds':time.monotonic()-total_start},allow_nan=False)+'\n')
            log.flush()
    qualifying=smooth&np.isfinite(centered_rel)&(centered_rel<=GATE)&(reference_norm>0.0)
    adjacent=np.any(qualifying[:-1]&qualifying[1:],axis=0) if len(GRID)>1 else np.zeros((9,n),bool)
    # A no-pair result is classified conservatively. Rising small-epsilon error
    # is descriptive of cancellation sensitivity, not a measured noise floor.
    per_basis=[]
    unresolved_rows=[]
    for col in range(9):
        best=np.full(n,np.nan);best_power=np.full(n,-1,dtype=np.int16)
        for pi,power in enumerate(GRID):
            vals=centered_rel[pi,col]
            take=np.isfinite(vals)&((~np.isfinite(best))|(vals<best))
            best[take]=vals[take];best_power[take]=power
        zero=zero_action[col]
        row_status=[]
        for j in range(n):
            if not center_valid[j]: status='center_projection_invalid'
            elif zero[j]: status='zero_reference_direction_inconclusive'
            elif adjacent[col,j]: status='resolved_adjacent_same_branch_pair'
            elif not np.any(smooth[:,col,j]): status='branch_or_endpoint_limited_inconclusive'
            else: status='no_adjacent_1e-6_pair_in_declared_grid'
            row_status.append(status)
            if status!='resolved_adjacent_same_branch_pair':
                unresolved_rows.append({'basis_column':col,'tet':int(paper[j]),'status':status,'best_relative_error':float(best[j]) if np.isfinite(best[j]) else None,'best_power':int(best_power[j]),'smooth_grid_points':int(smooth[:,col,j].sum()),'branch_cross_grid_points':int(branch_cross[:,col,j].sum()),'valid_plus_points':int(plus_valid[:,col,j].sum()),'valid_minus_points':int(minus_valid[:,col,j].sum())})
        row_counts={s:row_status.count(s) for s in sorted(set(row_status))}
        per_basis.append({'basis_column':col,'resolved_rows':int(adjacent[col].sum()),'unresolved_rows':n-int(adjacent[col].sum()),'zero_reference_rows':int(zero.sum()),'status_counts':row_counts,'best_relative_error_stats_over_resolvable_values':stats(best,np.isfinite(best)),'best_power_histogram':{str(v):int(np.sum(best_power==v)) for v in GRID}})
    eps_reports=[]
    for pi,power in enumerate(GRID):
        cols=[]
        for col in range(9):
            rel=centered_rel[pi,col]
            mask=np.isfinite(rel)
            cols.append({'basis_column':col,'same_branch_rows':int(smooth[pi,col].sum()),'branch_cross_rows':int(branch_cross[pi,col].sum()),'endpoint_invalid_rows':int(np.sum(~(plus_valid[pi,col]&minus_valid[pi,col]))),'centered_rows_at_1e-6':int(np.sum(mask&(rel<=GATE))),'max_relative_on_resolved_same_branch_rows':float(rel[mask].max()) if mask.any() else None,'plus_one_sided_same_branch_rows':int(plus_keep[pi,col].sum()),'minus_one_sided_same_branch_rows':int(minus_keep[pi,col].sum()),'plus_one_sided_at_1e-6':int(np.sum(np.isfinite(plus_rel[pi,col])&(plus_rel[pi,col]<=GATE))),'minus_one_sided_at_1e-6':int(np.sum(np.isfinite(minus_rel[pi,col])&(minus_rel[pi,col]<=GATE)))})
        eps_reports.append({'power':power,'epsilon':2.0**(-power),'columns':cols})
    report={
        'schema':'numi.cardboard.native531-ninebasis-fp64-fd.v1',
        'classification':'CPU-only compensated-FP64 local material derivative self-consistency sweep at captured centers; no native FP32 noise floor, native matrix qualification, or physical solver qualification',
        'parent_replay_report_sha256':sha(BASE/'report.json'),'parent_replay_arrays_sha256':sha(BASE/'arrays.npz'),'refinement_manifest_sha256':sha(REFINE/'manifest.json'),
        'native_capture_sha256':sha(capture),'prediction_sha256':sha(CASE/'prediction.json'),'reference_reader_sha256':sha(PREP/'strict_layers.py'),
        'scope':{'environment':0,'control':17,'newton':34,'paper_tets':n,'basis_columns':9,'perturbation':'F +/- epsilon * E_j in captured local F coordinates; rate-independent captured law; accepted history held fixed','epsilon_powers':GRID,'reference_tolerance_residual':2e-12,'reference_tolerance_dynamic_correction':2e-11,'relative_l2_gate':GATE},
        'input_identity':{'all_parent_center_projection_valid':bool(center_valid.all()),'parent_center_history_bitwise_match':True,'parent_center_F_bitwise_match':True,'recomputed_center_state_bitwise_match':True,'max_center_state_abs_difference':float(center_state_delta),'center_state_max_scaled_residual':float(center_residual.max()),'center_state_max_dynamic_correction':float(center_correction.max()),'basis_tangents_match_parent_reference_C':bool(np.array_equal(tangent.reshape(9,n,9),np.stack([center_C[paper,:,j] for j in range(9)]))),'max_basis_tangent_vs_parent_C_abs':float(tangent_parent_error.max())},
        'resolved_rows_total':int(adjacent.sum()),'row_direction_pairs_total':int(9*n),'all_row_direction_pairs_resolved':bool(adjacent.all()),
        'per_basis':per_basis,'per_epsilon':eps_reports,
        'unresolved_row_directions':unresolved_rows,
        'limits':['This is local constitutive FP64 self-consistency for nine basis directions at one captured center and fixed accepted history.','The grid is ideal FP64 F perturbations; it does not realize FP32 endpoints or measure native arithmetic noise.','Rows without a resolved pair remain inconclusive; zero reference directions cannot be judged by relative error.','One-sided statistics are retained but do not substitute for the centered same-branch gate.','No native cached tangent comparison is performed by this audit; see parent native layers for its own captured-direction comparison.','No full solver, mesh/time convergence, material calibration, or physical qualification.'],
        'wall_seconds':time.monotonic()-total_start,
        'finished_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()
    }
    arrays={'paper_tet_ids':paper.astype(np.int32),'material_index':material[paper].astype(np.int8),'center_F':F_capture,'accepted_history':old,'center_state':z,'center_branch':center_branch,'basis_tangent':tangent,'basis_tangent_parent_C_abs_error':tangent_parent_error,'basis_tangent_norm':tangent_norm,'center_projection_iterations':it,'center_projection_residual':rn,'center_projection_correction':cn,'plus_valid':plus_valid,'minus_valid':minus_valid,'plus_iterations':plus_iters,'minus_iterations':minus_iters,'plus_scaled_residual':plus_residual,'minus_scaled_residual':minus_residual,'plus_dynamic_correction':plus_correction,'minus_dynamic_correction':minus_correction,'plus_branch':plus_branch,'minus_branch':minus_branch,'smooth_same_branch':smooth,'plus_branch_preserving':plus_keep,'minus_branch_preserving':minus_keep,'branch_cross':branch_cross,'centered_relative_error':centered_rel,'centered_absolute_error':centered_abs,'reference_action_norm':reference_norm,'plus_one_sided_relative_error':plus_rel,'minus_one_sided_relative_error':minus_rel,'qualifying':qualifying,'adjacent_resolved':adjacent}
    np.savez_compressed(OUT/'arrays.npz',**arrays)
    (OUT/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    command={'schema':'numi.cardboard.native531-ninebasis-command.v1','argv':[str(Q/'venvs/cardboard-reference-185/bin/python3'),'-B',str(OUT/'audit9basis.py')],'cwd':str(Q),'environment':{'PYTHONPATH':'snapshots/source-contact-damped-memory-198-001/matter/tools','PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','VECLIB_MAXIMUM_THREADS':'1','MKL_NUM_THREADS':'1'},'execution':'CPU-only, no native transactions','wall_seconds':report['wall_seconds']}
    (OUT/'run.command.json').write_text(json.dumps(command,indent=2)+'\n')
    manifest={'schema':'numi.cardboard.native531-ninebasis-files.v1','qualification':False,'parent_inputs_sha256':{str(BASE/nm):sha(BASE/nm) for nm in ('manifest.json','report.json','arrays.npz')},'refinement_manifest_sha256':sha(REFINE/'manifest.json'),'native_case_inputs_sha256':{str(CASE/nm):sha(CASE/nm) for nm in ('prediction.json','receipt.json','stderr.log','output/manifest.json')},'capture_sha256':sha(capture),'reference_source_sha256':{str(S/nm):sha(S/nm) for nm in ('cardboard_bilinear_fp64_reference.py','cardboard_compensated_fp64_reference.py','cardboard_compensated_fp64_base.py','cardboard_dd_fp64.py','cardboard_cached_operator_audit.py')},'reader_sha256':sha(PREP/'strict_layers.py'),'files_sha256':{}}
    manifest['files_sha256']={str(q.relative_to(OUT)):sha(q) for q in sorted(OUT.rglob('*')) if q.is_file() and q.name!='manifest.json'}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'output':str(OUT),'report_sha256':sha(OUT/'report.json'),'arrays_sha256':sha(OUT/'arrays.npz'),'manifest_sha256':sha(OUT/'manifest.json'),'resolved':int(adjacent.sum()),'total':int(adjacent.size),'unresolved':len(unresolved_rows),'wall_seconds':report['wall_seconds']},indent=2))
if __name__=='__main__':main()

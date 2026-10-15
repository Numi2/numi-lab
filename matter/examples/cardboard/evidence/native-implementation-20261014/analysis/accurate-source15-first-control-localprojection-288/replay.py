#!/usr/bin/env python3
"""CPU-only FP64 local-projection replay for the first-control 15-state inputs.

Reads the exact source-bound N0 rich operator capture from run 283. It compares
an authored residual-only Newton solve with the accurate projection policy's
unchanged residual plus dimensionless correction criterion. It does not invoke
Metal, change parameters, or alter the constitutive equations.
"""
from pathlib import Path
import sys, json, hashlib, time, collections
import numpy as np
BASE=Path('/Users/home/cardboard-implementation-20261014')
CAP=BASE/'solver/recycle-source-default-first-control-283/operator-capture'
MANIFEST=BASE/'solver/recycle-source-default-first-control-283/output/manifest.json'
PRED=BASE/'solver/recycle-source-default-first-control-283/prediction.json'
LINER=BASE/'snapshots/source-accepted-correction-regional-recycle-085/matter/materials/hajali2009_liner_hill_bilinear_fit.nmatter'
MEDIUM=BASE/'snapshots/source-accepted-correction-regional-recycle-085/matter/materials/hajali2009_medium_hill_bilinear_fit.nmatter'
SRC=BASE/'snapshots/source-accepted-correction-regional-recycle-085/matter/src/metal/common.metalinc'
ANALYSIS=BASE/'analysis/bilinear15-fp64-replay-267'
SCALED=BASE/'analysis/local-projection-scaled-correction-082'
sys.path.insert(0,str(ANALYSIS)); sys.path.insert(0,str(SCALED))
import bilinear15_reference as ref
import replay_scaled_policy as pol
OUT=Path(__file__).parent

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def rows_from_capture():
    records=[json.loads(x) for x in CAP.read_text().splitlines() if x.strip()]
    header=records[0]
    mats={int(x['index']):x for x in header['materials']}
    selected=[]
    native_counts=[]
    for r in records:
        if r.get('record')!='tetrahedron' or r.get('material_index') not in (0,1): continue
        env=int(r['environment']);mi=int(r['material_index']); m=mats[mi]
        c=r['center']; off=int(m['parameter_offset']); n=int(r['material_parameter_count'])
        p=np.asarray(header['parameters_by_environment'][env][off:off+n],dtype=np.float32).astype(np.float64)
        F=(np.asarray(c['F_local'],dtype=np.float32).astype(np.float64)+
           np.asarray(c['F_local_low'],dtype=np.float32).astype(np.float64))
        old=np.asarray(c['accepted_state'],dtype=np.float32).astype(np.float64)
        z=np.array(old,copy=True)
        z[6:12]=ref.stiffness(p)@(ref.strain(F)-old[:6])
        native=int(round(float(c['projection'][2]))) if c.get('projection') else None
        selected.append({'tet':int(r['tet_id']),'env':env,'mat':mi,'p':p,'F':F,'old':old,'seed':z,'native_iters':native})
    return header, selected

def run():
    start=time.time(); header, rows=rows_from_capture()
    savedmax=pol.MAXITER
    baseline=[]; accurate=[]; accurate_budget=16
    pol.MAXITER=32
    for i,row in enumerate(rows):
        z,info=pol.project_existing(row['p'],row['F'],row['old'],row['seed'])
        baseline.append((z,info))
        if i and i%500==0: print(f'baseline {i}/{len(rows)}',flush=True)
    pol.MAXITER=accurate_budget
    for i,row in enumerate(rows):
        z,info=pol.project_scaled(row['p'],row['F'],row['old'],row['seed'])
        if info['status']!='converged':
            pol.MAXITER=32; z32,info32=pol.project_scaled(row['p'],row['F'],row['old'],row['seed'])
            info['retry32']={'status':info32['status'],'iterations':info32['iterations'],'residual_inf_scaled':info32.get('residual_inf_scaled'),'scaled_correction_inf':info32.get('scaled_correction_inf')}
            z,info=z32,info32
        accurate.append((z,info))
        if i and i%500==0: print(f'accurate {i}/{len(rows)}',flush=True)
    pol.MAXITER=savedmax
    status=collections.Counter(x['status'] for _,x in accurate)
    iters=[int(x['iterations']) for _,x in accurate]
    native_by_status=collections.Counter()
    max_abs=0.0; max_rel=0.0; worst=[]; max_resfrac=0.0; max_corrfrac=0.0
    baseline_fail=[]
    per=[]
    for row,(zb,ib),(za,ia) in zip(rows,baseline,accurate):
        if ib['status']!='converged': baseline_fail.append({'tet':row['tet'],'env':row['env'],'mat':row['mat'],'status':ib['status'],'iterations':ib.get('iterations')})
        d=float(np.max(np.abs(za-zb)))
        rel=float(np.linalg.norm(za-zb)/max(np.linalg.norm(zb),1e-30))
        max_abs=max(max_abs,d);max_rel=max(max_rel,rel)
        resfrac=float(ia.get('residual_inf_scaled',0)/max(ia.get('authored_residual_threshold',1e-300),1e-300))
        corrfrac=float(ia.get('scaled_correction_inf',0)/2e-8)
        max_resfrac=max(max_resfrac,resfrac);max_corrfrac=max(max_corrfrac,corrfrac)
        rec={'tet':row['tet'],'environment':row['env'],'material_index':row['mat'],
             'native_projection_iterations':row['native_iters'],'baseline_iterations':int(ib.get('iterations',-1)),
             'accurate_iterations':int(ia.get('iterations',-1)),'accurate_status':ia['status'],
             'max_abs_state_difference_baseline_accurate':d,'relative_l2_state_difference_baseline_accurate':rel,
             'residual_threshold_fraction':resfrac,'scaled_correction_threshold_fraction':corrfrac,
             'initial_lambda':float(row['seed'][12]),'accepted_lambda':float(row['old'][12])}
        per.append(rec)
        if ia['status']=='converged': native_by_status[str(row['native_iters'])]+=1
    worst=sorted(per,key=lambda x:x['max_abs_state_difference_baseline_accurate'],reverse=True)[:20]
    out={
      'schema':'numi.cardboard.bilinear15-first-control-local-projection-replay.v1',
      'classification':'Independent CPU FP64 equation replay only; does not reproduce Metal FP32/compensated operation ordering or prove the native 288 failure cause.',
      'capture':str(CAP),'capture_sha256':sha(CAP),'header':{k:header.get(k) for k in ['root','newton_iteration','world_fingerprint_hex','execution_fingerprint_hex','environment_count','node_count_per_environment','tetrahedron_count_per_environment','dt_seconds_by_environment']},
      'rows':len(rows),'row_counts_by_environment_material':{f'{e}:{m}':sum(r['env']==e and r['mat']==m for r in rows) for e in sorted(set(r['env'] for r in rows)) for m in (0,1)},
      'source_and_parameter_bindings':{'manifest_sha256':sha(MANIFEST),'prediction_sha256':sha(PRED),'liner_nmatter_sha256':sha(LINER),'medium_nmatter_sha256':sha(MEDIUM),'common_metalinc_sha256':sha(SRC),'reference_equations_sha256':sha(ANALYSIS/'bilinear15_reference.py'),'scaled_policy_replay_sha256':sha(SCALED/'replay_scaled_policy.py')},
      'equations_and_policy':{'state_count':15,'FB_epsilon_parameter_index':26,'authored_residual_threshold':2e-8,'authored_residual_scale_parameter':10.0,'accurate_scaled_correction_threshold':2e-8,'line_search':'original residual infinity norm Armijo, 16 backtracks','initial_guess':'accepted ep and histories; stress state from authored elastic StVK stress update; accumulated multiplier unchanged','budget_tested':{'residual_only':32,'accurate_scaled_correction':16,'retry_for_nonconverged':32},'changed_physics_or_parameters':False},
      'native_source283_projection_counts':dict(collections.Counter(int(r['native_iters']) for r in rows if r['native_iters'] is not None)),
      'accurate_fp64_status_counts':dict(status),'accurate_fp64_iterations':{'min':min(iters),'mean':float(np.mean(iters)),'p95':float(np.percentile(iters,95)),'max':max(iters)},
      'baseline_fp64_failure_count':len(baseline_fail),'baseline_fp64_failures':baseline_fail[:25],
      'baseline_vs_accurate_roots':{'max_abs_state_difference':max_abs,'max_relative_l2_state_difference':max_rel,'rows_absdiff_gt_1e-8':sum(x['max_abs_state_difference_baseline_accurate']>1e-8 for x in per),'rows_absdiff_gt_1e-6':sum(x['max_abs_state_difference_baseline_accurate']>1e-6 for x in per),'worst_rows':worst},
      'accurate_gate_worst_fractions':{'residual_threshold':max_resfrac,'scaled_correction_threshold':max_corrfrac},
      'elapsed_seconds':time.time()-start,'per_row':per}
    outp=OUT/'replay.json';outp.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:out[k] for k in ['rows','row_counts_by_environment_material','native_source283_projection_counts','accurate_fp64_status_counts','accurate_fp64_iterations','baseline_vs_accurate_roots','accurate_gate_worst_fractions','elapsed_seconds']},indent=2))
if __name__=='__main__':run()

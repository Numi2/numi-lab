#!/usr/bin/env python3
"""Budget-only comparison on the 283 captured N0 local inputs; CPU FP64 only."""
from pathlib import Path
import sys,json,hashlib,time,collections
import numpy as np
BASE=Path('/Users/home/cardboard-implementation-20261014')
OUT=Path(__file__).parent
sys.path.insert(0,str(BASE/'analysis/accurate-source15-first-control-localprojection-288'))
import replay as capture_replay
sys.path.insert(0,str(BASE/'analysis/bilinear15-fp64-replay-267'))
import bilinear15_reference as ref
sys.path.insert(0,str(BASE/'analysis/local-projection-scaled-correction-082'))
import replay_scaled_policy as pol

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    t0=time.time(); h,rows=capture_replay.rows_from_capture(); out={}
    roots={}; per={}
    for budget in (16,32,64):
        pol.MAXITER=budget; solved=[]; stat=collections.Counter(); its=[]; fails=[]
        for r in rows:
            z,info=pol.project_scaled(r['p'],r['F'],r['old'],r['seed'])
            stat[info['status']]+=1; its.append(int(info['iterations']))
            if info['status']!='converged': fails.append({'tet':r['tet'],'environment':r['env'],'material_index':r['mat'],'status':info['status'],'iterations':info['iterations'],'residual':info.get('residual_inf_scaled'),'correction':info.get('scaled_correction_inf')})
            solved.append(z)
        roots[budget]=solved
        out[str(budget)]={'status_counts':dict(stat),'iterations':{'min':min(its),'mean':float(np.mean(its)),'p95':float(np.percentile(its,95)),'max':max(its)},'failures':fails}
        print('budget',budget,'status',dict(stat),'maxiters',max(its),flush=True)
    for budget in (32,64):
        deltas=[]
        for a,b in zip(roots[16],roots[budget]): deltas.append(float(np.max(np.abs(a-b))))
        out[f'root_compare_16_vs_{budget}']={'max_abs_state_difference':max(deltas),'rows_over_1e-12':sum(x>1e-12 for x in deltas),'rows_over_1e-8':sum(x>1e-8 for x in deltas)}
    out['schema']='numi.cardboard.bilinear15-first-control-local-projection-budget-sweep.v1'
    out['classification']='CPU FP64 equation replay of exact captured 283 first-control F/accepted-state/material-parameter inputs; not a native replay of failed run 288.'
    out['capture_sha256']=sha(BASE/'solver/recycle-source-default-first-control-283/operator-capture')
    out['equation_reference_sha256']=sha(BASE/'analysis/bilinear15-fp64-replay-267/bilinear15_reference.py')
    out['policy_replay_sha256']=sha(BASE/'analysis/local-projection-scaled-correction-082/replay_scaled_policy.py')
    out['rows']=len(rows); out['epsilon']=1e-8; out['authored_residual_tolerance']=2e-8; out['scaled_correction_tolerance']=2e-8; out['line_search']='unchanged authored infinity-norm Armijo, 16 backtracks'
    out['elapsed_seconds']=time.time()-t0
    (OUT/'budget_sweep.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:out[k] for k in ['16','32','64','root_compare_16_vs_32','root_compare_16_vs_64','elapsed_seconds']},indent=2))
if __name__=='__main__': main()

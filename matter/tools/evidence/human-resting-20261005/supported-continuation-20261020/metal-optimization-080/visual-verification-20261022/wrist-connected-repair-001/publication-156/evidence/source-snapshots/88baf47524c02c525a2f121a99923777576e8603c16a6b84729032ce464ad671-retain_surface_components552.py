"""Keep only exactly measured regional improvements of a checked proposal.

Inputs can contain failed point regions; those complete nonlinear-stencil
components are reverted to the zero-contained baseline. Final545 remains
mandatory because mixed unknown self interfaces have not been certified.
"""
from pathlib import Path
import argparse,json,sys
import numpy as np
H=Path(__file__).resolve().parent.parent/'component-area-backtracking-549';sys.path.insert(0,str(H))
from component_area_backtracking549 import components,read_npz,sha,same
def counts(report,fo,n):
 if not report['complete'] or report['bank_count']!=185 or report['expected_bank_count']!=185 or len(report['banks'])!=185:raise ValueError('incomplete185inventory')
 keys=[(b['body'],b['stable_bone_id']) for b in report['banks']]
 if len(set(keys))!=185:raise ValueError('duplicate banks')
 v=np.zeros(n,np.int64);fixed=0
 for b in report['banks']:
  result=b.get('result') or {};pairs=result.get('triangle_pairs',[])
  if len(pairs)!=result.get('count',0):raise ValueError('paircount mismatch')
  for face,_ in pairs:
   if face<0 or face>=len(fo):raise ValueError('invalid face')
   ci=fo[face]
   if ci<0:fixed+=1
   else:v[ci]+=1
 return v,fixed,keys
def main():
 ap=argparse.ArgumentParser()
 for k in ['state','proposal','full-check','context','output']:ap.add_argument('--'+k,type=Path,required=True)
 a=ap.parse_args()
 if a.output.exists():raise FileExistsError(a.output)
 st=json.loads(a.state.read_text());base=Path(st['candidate']);bp=Path(st['point_summary']);bs=Path(st['surface_report'])
 for k in ['candidate','point_summary','surface_report']:
  if sha(st[k])!=st[k+'_sha256']:raise ValueError('baseline state pin')
 B=read_npz(base);C=read_npz(a.proposal/'candidate.npz');cr=json.loads((a.proposal/'report.json').read_text())
 if cr['candidate_sha256']!=sha(a.proposal/'candidate.npz') or cr['base_sha256']!=sha(base):raise ValueError('proposal pins')
 if set(B)!=set(C) or any(not same(B[k],C[k]) for k in B if k!='points_world_m'):raise ValueError('nonpoint payload changed')
 ck=json.loads((a.full_check/'report.json').read_text())
 if ck['candidate_sha256']!=cr['candidate_sha256'] or ck['status']!='COMPLETE_EXACT_CONSTRUCTION_CHECKS' or ck['self_pairs']!=0:raise ValueError('complete checked trial required')
 x=B['points_world_m'];target=C['points_world_m'];f=B['source_O_faces'];free=np.array(cr['free_nodes'],np.int64)
 ctx=json.loads(a.context.read_text())
 if ctx['candidate_sha256']!=sha(base):raise ValueError('context pin')
 groups,proof=components(len(x),f,free,np.load(ctx['seed_pairs_path'],allow_pickle=False))
 owner=np.full(len(x),-1,np.int64)
 for i,g in enumerate(groups):owner[g]=i
 outside=np.flatnonzero(owner<0)
 if not same(x[outside],target[outside]):raise ValueError('proposal outside change')
 fo=owner[f].max(1);br=json.loads(bs.read_text());sr=json.loads((a.full_check/'audit-159/report.json').read_text())
 if sr['candidate_sha256']!=cr['candidate_sha256'] or br['candidate_sha256']!=sha(base):raise ValueError('surface pins')
 bv,bfix,bkeys=counts(br,fo,len(groups));tv,tfix,tkeys=counts(sr,fo,len(groups))
 if bkeys!=tkeys or bfix!=tfix:raise ValueError('bank orfixedinventory mismatch')
 pr=json.loads((a.full_check/'audit-130/summary.json').read_text());bpr=json.loads(bp.read_text())
 if pr['candidate_sha256']!=cr['candidate_sha256'] or not pr['complete_surface_audit'] or pr['exact_PLC_surface_pairs']!=0 or len(pr['inside_witnesses'])!=pr['inside_node_bank_pairs']:raise ValueError('incomplete pointwitnesses')
 if bpr['inside_node_bank_pairs'] or bpr['exact_PLC_surface_pairs']:raise ValueError('baseline pointfail')
 bad={int(owner[int(w['output_node'])]) for w in pr['inside_witnesses']}
 if -1 in bad:raise ValueError('contained fixed point')
 out=x.copy();choices=[]
 for i,g in enumerate(groups):
  keep=i not in bad and tv[i]<bv[i]
  if keep:out[g]=target[g]
  choices.append(dict(component=i,nodes=g.tolist(),baseline_pairs=int(bv[i]),trial_pairs=int(tv[i]),contained_witness=bool(i in bad),selected_trial=bool(keep)))
 predicted=int(bfix+sum(tv[i] if z['selected_trial'] else bv[i] for i,z in enumerate(choices)))
 if predicted>=int(bfix+bv.sum()):raise ValueError('noexactpredictedprogress')
 a.output.mkdir();payload=dict(B);payload['points_world_m']=out;op=a.output/'candidate.npz';np.savez_compressed(op,**payload)
 pins={str(p):sha(p) for p in [Path(__file__),H/'component_area_backtracking549.py',a.state,a.proposal/'report.json',a.proposal/'candidate.npz',a.full_check/'report.json',a.full_check/'audit-130/summary.json',a.full_check/'audit-159/report.json',a.context,Path(ctx['seed_pairs_path'])]}
 report=dict(status='CONSTRUCTION_PROPOSAL_WRITTEN_UNADMITTED',candidate_path=str(op),candidate_sha256=sha(op),baseline=str(base),baseline_sha256=sha(base),proof=proof,choices=choices,predicted_surface_pairs=predicted,all_nonpoint_payload_bitwise_unchanged=True,source_pins=pins,native_admission=False,accepted_physical_steps=0,exact_full545_required=True)
 (a.output/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
 print(json.dumps(dict(predicted_surface_pairs=predicted,selected_components=[z['component'] for z in choices if z['selected_trial']],native_admission=False)))
if __name__=='__main__':main()

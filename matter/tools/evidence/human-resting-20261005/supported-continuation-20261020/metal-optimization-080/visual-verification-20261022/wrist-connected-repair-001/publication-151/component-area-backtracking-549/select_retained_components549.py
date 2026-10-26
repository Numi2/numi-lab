import numpy as np,json,sys
from pathlib import Path
h=Path('/Users/numi/Work/numi-human-connected-surface-069/D/component-area-backtracking-549');sys.path.insert(0,str(h));from component_area_backtracking549 import components,read_npz,sha,same
n=h.parent/'surface-clearance-546';bp=n/'raw-lp-trial-005-full-plane/fit/candidate.npz';b=read_npz(bp);x=b['points_world_m'];f=b['source_O_faces']
cs=[(n/'round-013-full-plane-trust4/fit/candidate.npz',n/'round-013-full-plane-trust4/first-check/report.json'),(h/'run-retained-011-004/proposal/candidate.npz',h/'run-retained-011-004/full-check/report.json')];xs=[];counts=[]
for p,rp in cs:
 ck=json.loads(rp.read_text());assert ck['candidate_sha256']==sha(p) and ck['self_pairs']==ck['contained_pairs']==0 and ck['status']=='COMPLETE_EXACT_CONSTRUCTION_CHECKS'
 c=read_npz(p);assert set(c)==set(b) and all(same(c[k],b[k]) for k in b if k!='points_world_m' and not k.startswith('local_witness_cluster_'));xs.append(c['points_world_m'])
 j=json.loads((rp.parent/'audit-159/report.json').read_text());assert j['complete'] and j['bank_count']==185 and j['candidate_sha256']==sha(p);v=np.zeros(len(f),int)
 for bank in j['banks']:
  for pair in (bank.get('result') or {}).get('triangle_pairs',[]):v[int(pair[0])]+=1
 counts.append(v)
free=np.unique(np.concatenate([np.flatnonzero(np.any(z!=x,axis=1)) for z in xs]));ctx=json.loads((n/'round-012-full-plane-trust4/constraints/constraint-context.json').read_text());g,proof=components(len(x),f,free,np.load(ctx['seed_pairs_path']));owner=np.full(len(x),-1)
for i,a in enumerate(g):owner[a]=i
fo=owner[f].max(1);out=x.copy();choices=[]
for i,a in enumerate(g):
 totals=[int(v[fo==i].sum()) for v in counts];pick=int(np.argmin(totals));out[a]=xs[pick][a];choices.append({'component':i,'nodes':a.tolist(),'counts':totals,'chosen':pick})
o=h/'combined-normal013-component011';o.mkdir(exist_ok=False);c=dict(b);c['points_world_m']=out;np.savez_compressed(o/'candidate.npz',**c)
r={'status':'CONSTRUCTION_PROPOSAL_WRITTEN_UNADMITTED','candidate_sha256':sha(o/'candidate.npz'),'candidate_path':str(o/'candidate.npz'),'baseline':str(bp),'baseline_sha256':sha(bp),'inputs':[{'candidate':str(p),'candidate_sha256':sha(p),'full545_report':str(rp),'full545_report_sha256':sha(rp)} for p,rp in cs],'proof':proof,'choices':choices,'predicted_pair_count_from_exact_face_ownership':sum(min(v['counts']) for v in choices),'all_nonpoint_payload_bitwise_unchanged_from_baseline':True,'construction_diagnostics_in_npz':'inherited baseline records; this report owns new mixed proposal','native_admission':False,'accepted_physical_steps':0}
(o/'report.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k not in ['choices','inputs']}))

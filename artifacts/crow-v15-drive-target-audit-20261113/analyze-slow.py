from pathlib import Path
import gzip,hashlib,json,math
R=Path('/Users/n/Developer/numi-crow-rl-20261107');A=R/'artifacts/crow-v15-drive-target-audit-20261113';trial=A/'push-nowings-20261127-rebound-slow-target';ref=R/'artifacts/crow-v15-slow-leg-push-20261113/push-nowings-20261127-rebound-slow160'
def load(p,targets=False):
 d={'state':{},'contact':{},'targets':{},'errors':[]}
 with gzip.open(p/'run.stderr.gz','rt') as f:
  for line in f:
   if line.startswith('CROW_SUPPORT_SAMPLE '):
    x=json.loads(line.split(' ',1)[1]);key=(x['lane'],x['step']);d['state'].setdefault(key,[]).append(json.dumps(x,sort_keys=True,separators=(',',':')))
   elif line.startswith('CROW_TWO_TOE_SUBSTEP '):
    z=line.split();t,k,e=map(int,z[1:4]);key=e,t,k;assert key not in d['contact'];d['contact'][key]=z[4:]
    if int(z[4])>int(z[5]):d['errors'].append(['overflow',e,t,k])
   elif targets and line.startswith('CROW_DRIVE_TARGET '):
    z=line.split();t,e=map(int,z[1:3]);v=list(map(float,z[3:]));assert len(v)==40 and all(map(math.isfinite,v));assert (e,t) not in d['targets'];d['targets'][e,t]=v
   elif line.startswith('CROW_SUBSTEP_ERROR'):d['errors'].append(line.strip())
 return d
old=load(ref);new=load(trial,True)
def cmp(a,b):
 keys=a.keys()&b.keys();bad=[k for k in sorted(keys) if a[k]!=b[k]];return {'reference':len(a),'observed':len(b),'shared':len(keys),'mismatches':len(bad),'missing_reference':len(b.keys()-a.keys()),'missing_observed':len(a.keys()-b.keys()),'first_mismatch':list(bad[0]) if bad else None}
parity={k:cmp(old[k],new[k]) for k in ('state','contact')};parity.update(reference_errors=old['errors'],audit_errors=new['errors'],target_samples=len(new['targets']));(A/'slow-parity.json').write_text(json.dumps(parity,indent=2)+'\n');print(json.dumps(parity))
assert not old['errors'] and not new['errors'] and len(new['targets'])==46*8
assert all(parity[k]['mismatches']==0 and parity[k]['missing_reference']==0 and parity[k]['missing_observed']==0 for k in ('state','contact'))
rows=[]
for e in range(8):
 for t in range(335,381):
  v=new['targets'][e,t];x=json.loads(new['state'][e,t][0]);rows.append({'lane':e,'step':t,'left_target':v[14],'right_target':v[17],'left_q':x['q'][15],'right_q':x['q'][18]})
out={'rows':rows,'candidate_target_lt_lower_count':sum(v[k]<-.1-1e-6 for v in rows for k in ('left_target','right_target')),'candidate_target_gt_upper_count':sum(v[k]>1.55+1e-6 for v in rows for k in ('left_target','right_target')),'target_min':min(v[k] for v in rows for k in ('left_target','right_target')),'target_max':max(v[k] for v in rows for k in ('left_target','right_target')),'host_sha256':hashlib.sha256((R/'build/crow-v15-drive-target-audit-20261113/libmetalrobo.dylib').read_bytes()).hexdigest(),'scope':'Copied position targets prior to solver limit clamp, recorded at first substep; one used development seed. Knee joint position limits [-0.10,1.55] rad. Earlier -0.70 request condition measured.'};(A/'slow-analysis.json').write_text(json.dumps(out,indent=2)+'\n');print({k:out[k] for k in ('target_min','target_max','candidate_target_lt_lower_count','candidate_target_gt_upper_count')})

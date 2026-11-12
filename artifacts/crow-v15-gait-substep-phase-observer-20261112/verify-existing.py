from pathlib import Path
import gzip,json,math
A=Path('artifacts/crow-v15-gait-substep-phase-observer-20261112');P=Path('artifacts/crow-v15-magpie-phase-fit-screen-20261112/candidate-20261127-rebound-phase');q={};s={};contact={};errs=[]
with gzip.open(P/'run.stderr.gz','rt') as f:
 for l in f:
  if l.startswith('CROW_SUPPORT_SAMPLE '):
   x=json.loads(l.split(' ',1)[1]);e,t=x['lane'],x['step']
   if 201<=t<=300:assert (e,t) not in s;s[e,t]=x
  elif l.startswith('CROW_Q_SUBSTEP '):
   z=l.split();t,k,e=map(int,z[1:4]);v=list(map(float,z[4:]));assert len(v)==41 and all(map(math.isfinite,v)) and (e,t,k) not in q;q[e,t,k]=v
  elif l.startswith('CROW_TWO_TOE_SUBSTEP '):
   z=l.split();t,k,e=map(int,z[1:4]);
   if 201<=t<=300:
    assert (e,t,k) not in contact;contact[e,t,k]=z[4:]
    if int(z[4])>int(z[5]):errs.append(['overflow',e,t,k])
  elif l.startswith('CROW_SUBSTEP_ERROR'):errs.append(l.strip())
assert len(s)==800 and len(q)==len(contact)==12800
maxend=max(abs(a-b) for e,t in s for a,b in zip(q[e,t,15],s[e,t]['q']))
with gzip.open(P/'replay.json.gz','rt') as f:r=json.load(f)['payload']
assert r['environment']==5
frames={x['step']+1:x for x in r['frames'] if 200<=x['step']<=299};assert len(frames)==100
maxreplay=max(abs(a-b) for t,v in frames.items() for a,b in zip(s[5,t]['q'],v['q']))
out={'source':'artifacts/crow-v15-magpie-phase-fit-screen-20261112/candidate-20261127-rebound-phase','q_substeps':len(q),'contact_substeps':len(contact),'endpoint_samples':len(s),'max_q_error_final_substep_to_endpoint':maxend,'max_q_error_endpoint_to_lane5_replay':maxreplay,'errors':errs,'pass':maxend<1e-6 and maxreplay<1e-6 and not errs};(A/'verification.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2));assert out['pass']

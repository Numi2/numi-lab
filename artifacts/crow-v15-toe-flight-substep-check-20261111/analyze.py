from pathlib import Path
import collections,gzip,json
A=Path('artifacts/crow-v15-toe-flight-substep-check-20261111');B=Path('artifacts/crow-v15-two-segment-contact-flight-handoff-20261111');s={};ref={};v={};sub={};rsub={};errors=[]
def read(p,states,contacts,readv):
 with gzip.open(p/'stderr.gz','rt') as f:
  for l in f:
   if l.startswith('CROW_SUPPORT_SAMPLE '):
    x=json.loads(l.split(' ',1)[1]);states[x['lane'],x['step']]=x
   elif l.startswith('CROW_TWO_TOE_SUBSTEP '):
    a=l.split();t,k,e=map(int,a[1:4]);contacts[e,t,k]=a[4:]
   elif readv and l.startswith('CROW_TOE_V_SUBSTEP '):
    a=l.split();t,k,e=map(int,a[1:4]);v[e,t,k]=list(map(float,a[4:]))
   elif readv and ('CROW_SUBSTEP_ERROR' in l or 'failed' in l.lower()):errors.append(l.strip())
read(A,s,sub,True);read(B,ref,rsub,False);out=[]
for e in range(8):
 diff=sum(s.get((e,t))!=ref.get((e,t)) for t in range(1,601));cdiff=sum(sub.get((e,t,k))!=rsub.get((e,t,k)) for t in range(1,601) for k in range(16));missing=sum(len(v.get((e,t,k),[]))!=16 for t in range(1,601) for k in range(16));endpoint=[];hits=[]
 for t in range(1,601):
  if (e,t) in s and (e,t,15) in v:endpoint.extend(abs(a-b) for a,b in zip(s[e,t]['v'][22:38],v[e,t,15]))
  for k in range(16):
   vv=v.get((e,t,k),[]);ix=[i for i,x in enumerate(vv) if abs(x)>=11.999]
   if ix:hits.append({'step':t,'substep':k,'toes':ix})
 out.append({'lane':e,'different_accepted_samples':diff,'different_contact_substeps':cdiff,'missing_velocity_substeps':missing,'endpoint_velocity_max_error':max(endpoint) if endpoint else None,'all_toe_cap_hits':hits,'flight_window_cap_hits':[x for x in hits if x['step']>=356]})
(A/'analysis.json').write_text(json.dumps({'lanes':out,'errors':errors,'scope':'development starts, 3-second post-release flight window; eligibility is conservatively inferred, no full-route or landing qualification'},indent=2)+'\n');print([(x['lane'],len(x['all_toe_cap_hits']),len(x['flight_window_cap_hits']),x['different_accepted_samples'],x['different_contact_substeps']) for x in out])

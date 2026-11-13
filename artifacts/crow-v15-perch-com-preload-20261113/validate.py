from pathlib import Path
import gzip,json,math
A=Path('artifacts/crow-v15-perch-com-preload-20261113');old=Path('artifacts/crow-v15-toe-release-regrip-20261111/trace.log.gz');now=A/'trace.log.gz';data={};ref={};hips={};com={};events={}
def scan(path,store):
 with gzip.open(path,'rt') as f:
  for line in f:
   z=line.split()
   if not z:continue
   k=z[0]
   if k in ('step','control','event'):
    e,t=map(int,z[1:3]);store[e,t,k]=z[3:]
    if path==now and k=='event':events[e,t]=z[3:]
   elif path==now and k=='hiptarget':
    e,t,qi=map(int,z[1:4]);assert (e,t,qi) not in hips;hips[e,t,qi]=list(map(float,z[4:]))
   elif path==now and k=='com':
    e,t=map(int,z[1:3]);assert (e,t) not in com;com[e,t]=list(map(float,z[3:]))
scan(now,data);scan(old,ref);analysis=json.loads((A/'analysis.json').read_text());rows=[]
for e in range(8):
 oldrelease=next(t for t in range(350) if (e%2,t,'event') in ref and int(ref[e%2,t,'event'][0])==1);missing=[];different=[]
 for t in range(oldrelease):
  for k in ('step','control'):
   q=data.get((e,t,k));r=ref.get((e%2,t,k))
   if q is None or r is None:missing.append((t,k))
   elif q!=r:different.append((t,k))
 hmiss=[];hbad=[];cmiss=[];readybad=[];release=next((t for t in range(350) if (e,t) in events and int(events[e,t][0])==1),None)
 for t in range(350):
  for qi in (14,17):
   v=hips.get((e,t,qi))
   if v is None:hmiss.append((t,qi))
   elif len(v)!=3 or not all(map(math.isfinite,v)) or not(v[1]-1e-5<=v[0]<=v[2]+1e-5):hbad.append((t,qi,v))
  x=com.get((e,t));ev=events.get((e,t))
  if x is None or ev is None:cmiss.append(t);continue
  cx,cv,preparing,ready=x;up0,up1,front0,rear0,front1,rear1=map(float,ev[2:8]);supported=up0>.000006 and up1>.000006 and up0+up1>.0003 and front0>.000001 and front1>.000001 and rear0<-.000001 and rear1<-.000001
  prev=com.get((e,t-1),[0,0,0,0]);want=prev[3]+1 if preparing and supported and abs(cx-.015)<=.005 and abs(cv)<=.02 else 0 if preparing else prev[3]
  if ready!=want:readybad.append((t,x,want))
 release_ready=release is not None and release>0 and (e,release-1) in com and com[e,release-1][3]>=20
 x=analysis['lanes'][e];passed=x['pass'] and not missing and not different and not hmiss and not hbad and not cmiss and not readybad and release_ready and analysis['missing_substeps']==0 and analysis['bad_substeps']==0 and not analysis['errors'];rows.append({'lane':e,'old_release_zero_based':oldrelease,'preload_prefix_missing':len(missing),'preload_prefix_differences':len(different),'hip_target_missing':len(hmiss),'hip_target_out_of_range':len(hbad),'com_missing':len(cmiss),'readiness_mismatch':len(readybad),'readiness_first_mismatch':readybad[:1],'release_zero_based':release,'release_after_readiness':release_ready,'previous_grip_screen_pass':x['pass'],'passed':passed})
out={'rows':rows,'any_pass':any(r['passed'] for r in rows),'scope':'Constructed stationary perch only, not airborne landing. COM observer independent validation and drive effort audit required before integration.'};(A/'validation.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))

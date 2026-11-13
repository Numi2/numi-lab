from pathlib import Path
import gzip,json,math
A=Path('artifacts/crow-v15-launch-aero-observer-20261113');B=Path('artifacts/crow-v15-leg-wing-attribution-20261113');P=A/'low-straight-20261126-rebound-early-observer';ref=json.load(gzip.open(B/'push-window.json.gz','rt'));cur=json.load(gzip.open(A/'observed-window.json.gz','rt'))
def mapdata(d):
 r={'errors':d['errors'],'state':{(x['lane'],x['step']):x for x in d['state']}}
 for typ,n in (('target',2),('contact',3),('leg',3),('toe',3),('aero',3)):
  r[typ]={tuple(x[:n]):x[n:] for x in d[typ]};assert len(r[typ])==len(d[typ]),typ
 return r
ref,cur=mapdata(ref),mapdata(cur);parity={}
for e in range(8):
 bad=[]
 for t in range(1,501):
  a=ref['state'].get((e,t));b=cur['state'].get((e,t));
  if a is None or b is None or any(a[k]!=b[k] for k in ('q','v','policy_action','contacts')):bad.append(t)
 parity[str(e)]={'mismatch_count':len(bad),'first_mismatches':bad[:5]}
subparity={}
for typ in ('target','contact','leg','toe'):
 keys=set(ref[typ])|set(cur[typ]);bad=[k for k in keys if ref[typ].get(k)!=cur[typ].get(k)];subparity[typ]={'keys':len(keys),'mismatch_count':len(bad),'first_mismatches':[list(k) for k in sorted(bad)[:5]]}
coverage=[];rows=[]
for e in range(8):
 missing=[(t,k) for t in range(335,401) for k in range(16) if (e,t,k) not in cur['aero']];coverage.append({'lane':e,'missing_substeps':len(missing),'first_missing':missing[:5]})
 for t in range(335,401):
  for k in range(16):
   v=cur['aero'].get((e,t,k));
   if v is not None:assert len(v)==13 and all(math.isfinite(x) for x in v)
 if e in (6,7) and not missing:
  v=[cur['aero'][e,t,k] for t in range(371,391) for k in range(16)];dt=.02/16;rows.append({'lane':e,'aero_vertical_impulse_371_390_Ns':sum(x[3] for x in v)*dt,'body_left_wing_right_wing_tail_vertical_impulse_Ns':[sum(x[i] for x in v)*dt for i in (4,5,6,7)],'peak_abs_aero_vertical_force_N':max(abs(x[3]) for x in v)})
inputs=json.loads((P/'inputs.json').read_text());ri=json.loads((Path('artifacts/crow-v15-low-pronation-load-20261113/low-straight-20261126-rebound-low-straight/inputs.json')).read_text());same=all(inputs[k]==ri[k] for k in ('metallib_sha256','frozen_flight_sha256','landing_policy_sha256','route_policy_sha256'));status=json.loads((P/'run-result.json').read_text());ok=not cur['errors'] and same and status['exit_code']==0 and all(x['mismatch_count']==0 for x in parity.values()) and all(x['mismatch_count']==0 for x in subparity.values()) and all(x['missing_substeps']==0 for x in coverage)
out={'scope':'observational host parity and uncalibrated model aero force; not biological or mission qualification','state_parity':parity,'substep_and_target_parity':subparity,'identical_physics_inputs':same,'observer_errors':cur['errors'],'aero_coverage':coverage,'launch_aero':rows,'passed':ok};(A/'analysis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))

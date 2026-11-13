from pathlib import Path
import gzip,json,sys
A=Path('artifacts/crow-v15-full-pronation-observer-20261113');P=A/sys.argv[1];R=Path('artifacts/crow-v15-phase-pronation-turn-20261113/signed-turn-20261126-rebound-phase')
def scan(path):
 state={};eff={};errs=[]
 with gzip.open(path,'rt') as f:
  for l in f:
   if l.startswith('CROW_SUPPORT_SAMPLE '):
    x=json.loads(l.split(' ',1)[1]);k=x['lane'],x['step'];assert k not in state;state[k]=x;continue
   z=l.split()
   if not z:continue
   if 'OBSERVER_ERROR' in z[0] or z[0]=='CROW_SUBSTEP_ERROR':errs.append(l.strip())
   if z[0]=='CROW_PRONATION_EFFORT':
    t,sub,e=map(int,z[1:4]);
    if 650<=t<=1150:
     k=(e,t,sub);assert k not in eff;eff[k]=list(map(float,z[4:]))
 return state,eff,errs
a,ea,err=scan(P/'run.stderr.gz');b,eb,errb=scan(R/'run.stderr.gz');missing=sorted(set(a)^set(b));bad=[]
for k in sorted(set(a)&set(b)):
 if any(a[k][f]!=b[k][f] for f in ('q','v','policy_action','contacts')):bad.append(k)
missinge=sorted(set(ea)^set(eb));bade=[k for k in set(ea)&set(eb) if ea[k]!=eb[k]]
new=json.loads((P/'run.stdout').read_text())['first_episode_navigation'];old=json.loads((R/'run.stdout').read_text())['first_episode_navigation'];newi=json.loads((P/'inputs.json').read_text());oldi=json.loads((R/'inputs.json').read_text());sameinput=all(newi[k]==oldi[k] for k in ('metallib_sha256','frozen_flight_sha256','landing_policy_sha256','route_policy_sha256'))
out={'state_records':len(a),'baseline_state_records':len(b),'missing_state_keys':len(missing),'state_mismatch_count':len(bad),'first_state_mismatches':bad[:10],'overlap_effort_records':len(ea),'baseline_overlap_effort_records':len(eb),'missing_effort_keys':len(missinge),'effort_mismatch_count':len(bade),'first_effort_mismatches':bade[:10],'same_first_episode_metadata':new==old,'same_nonhost_inputs':sameinput,'observer_errors':err+errb,'passed':not (missing or bad or missinge or bade or err or errb) and new==old and sameinput};(P/'observer-parity.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))

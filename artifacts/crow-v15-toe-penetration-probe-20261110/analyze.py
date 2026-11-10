from pathlib import Path
import json,statistics
A=Path('artifacts/crow-v15-toe-penetration-probe-20261110');contacts={};cases={}
for line in (A/'results.txt').read_text().splitlines():
 v=line.split()
 if not v:continue
 if v[0]=='contact':
  key=(int(v[1]),int(v[2]));contacts.setdefault(key,[]).append((int(v[4]),v[5],float(v[8]),float(v[9])))
 elif v[0]=='case_result':
  key=(int(v[1]),int(v[2]));cases[key]=v
out=[]
for case in range(2):
 for nsub in (4,8,16):
  k=(case,nsub);v=cases.get(k);r={'case':case,'substeps':nsub,'record':v,'status':'missing' if v is None else v[4]}
  if v and v[4]=='completed':
   t=int(v[v.index('transition')+1]);s=int(v[v.index('success')+1]);release=int(v[v.index('release')+1]);r.update(transition=t,success=s,release=release)
   rows=[x for x in contacts.get(k,[]) if release-10<=x[0]<release];d=[x[3] for x in rows];r['hold_contact_count']=len(d);r['hold_steps_with_contacts']=len({x[0] for x in rows});r['minimum_recorded_contact_separation_m']=min(d) if d else None;r['median_recorded_contact_separation_m']=statistics.median(d) if d else None;r['hold_toes']=sorted({x[1] for x in rows})
  out.append(r)
result={'cases':out,'limits':'Recorded solver contact separations during final ten pre-release control samples; not independent accepted-pose SAT, substep maxima, a v15 full episode, or biological validation. Exclude no failures.'}
(A/'analysis.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

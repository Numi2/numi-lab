from pathlib import Path
import gzip,json
A=Path('artifacts/crow-v15-foot-support-substep-audit-20261110');O=Path('artifacts/crow-v15-foot-position-feedback-screen-20261110');old={};new={};final={};interval={};phase={};outcomes={}
for path,dest in ((O/'results.txt.gz',old),(A/'results.txt.gz',new)):
 with gzip.open(path,'rt') as f:
  for line in f:
   v=line.split()
   if not v or len(v)<4 or v[1]!='1' or v[2] not in ('4','5'):continue
   if v[0]=="case_result":
    if path.parent==A:outcomes[v[2]]=" ".join(v[3:])
    continue
   k=(int(v[2]),int(v[3]))
   if v[0]=='trace':dest[k]=v[4:];phase[k]=int(v[4])
   elif path.parent==A and v[0]=='foot_support':final[k]=float(v[4])
   elif path.parent==A and v[0]=='foot_interval':interval[k]=float(v[4])
   elif path.parent==A and v[0]=='case_result':outcomes[v[2]]=' '.join(v[3:])
keys={(a,t) for a in (4,5) for t in range(200)};diff=[k for k in sorted(keys) if old.get(k)!=new.get(k)];rows=[]
for a in (4,5):
 ks=sorted(k for k in keys if k[0]==a and k in final and k in interval)
 def stats(xs):
  return {'samples':len(xs),'final_gate_on':sum(final[k]>.0002/16 for k in xs),'interval_gate_on':sum(interval[k]>.0002 for k in xs),'missed_final_gate':sum(final[k]<=.0002/16 and interval[k]>.0002 for k in xs),'final_only':sum(final[k]>.0002/16 and interval[k]<=.0002 for k in xs)}
 rows.append({'arm':a,'outcome':outcomes.get(str(a),'missing'),'all':stats(ks),'release':stats([k for k in ks if phase.get(k)==4])})
out={'exact_trajectory_parity':not diff and len(old)==400 and len(new)==400,'differences':diff[:20],'rows':rows};(A/'analysis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))

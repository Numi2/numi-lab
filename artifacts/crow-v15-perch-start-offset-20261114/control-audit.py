from pathlib import Path
import gzip,json
A=Path('artifacts/crow-v15-perch-start-offset-20261114');P=Path('artifacts/crow-v15-pretransfer-pitch-feedback-20261114');rows={}
for label,path in [('offset',A/'trace.log.gz'),('prior',P/'trace.log.gz')]:
 d={}
 for line in gzip.open(path,'rt'):
  z=line.split()
  if z and z[0] in ('step','control','event','hipcommand','jointtarget','rearKnee','rearAnkle','rightHipCommand','balance','pitchcontrol','stage') and z[1]=='0':d.setdefault((int(z[2]),z[0]),[]).append(z[3:])
 rows[label]=d
x=rows['offset'];y=rows['prior'];keys=set(x)|set(y);m=[k for k in sorted(keys) if x.get(k)!=y.get(k)];out={'control_record_count':len(x),'prior_record_count':len(y),'mismatch_count':len(m),'first_mismatches':m[:10]};(A/'control-audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))

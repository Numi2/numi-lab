from pathlib import Path
import gzip,json
D=Path('artifacts/crow-v15-microstep-contact-observer-20261110');old=Path('artifacts/crow-v15-force-normalized-grip-screen-20261110/results.txt.gz');trace={};ref={};observed={};outcomes={}
with gzip.open(old,'rt') as f:
 for line in f:
  v=line.split()
  if v and v[0]=='trace' and int(v[1])==1 and int(v[2])==1 and int(v[3])<40:ref[int(v[3])]=v[4:]
with gzip.open(D/'results.txt.gz','rt') as f:
 for line in f:
  v=line.split()
  if not v:continue
  if v[0]=='trace':trace[(int(v[2]),int(v[3]))]=v[4:]
  elif v[0]=='case_result':outcomes[int(v[2])]=' '.join(v[3:])
  elif v[0]=='subcontact' and int(v[2])==1:
   t,m=int(v[3]),int(v[4]);name=v[5];nx=float(v[6]);imp=float(v[7]);side='rear' if 'rear_toe' in name else 'front'
   z=observed.setdefault((t,name),[0.,0.]);z[0]+=imp
   if m==15:z[1]+=imp
base=[t for t in range(40) if ref.get(t)!=trace.get((0,t))];split=[t for t in range(40) if trace.get((0,t))!=trace.get((1,t))]
rows=[]
for (t,name),(total,last) in sorted(observed.items()):rows.append({'step':t,'toe':name,'interval_impulse':total,'last_substep_impulse':last,'interval_above_force_threshold':total>.0002,'last_above_force_threshold':last>.0000125})
miss=[r for r in rows if r['interval_above_force_threshold'] and not r['last_above_force_threshold']]
out={'baseline_different_steps':base,'split_different_steps':split,'outcomes':outcomes,'interval_contact_records':rows,'threshold_exceeds_interval_but_not_last_count':len(miss)};(D/'analysis.json').write_text(json.dumps(out,indent=2)+'\n')
print('baseline mismatch',len(base),'split mismatch',len(split),'outcomes',outcomes,'interval-only detections',len(miss))

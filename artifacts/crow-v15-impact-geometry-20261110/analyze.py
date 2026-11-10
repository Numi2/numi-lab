from pathlib import Path
import json,math
A=Path('artifacts/crow-v15-impact-geometry-20261110');base=Path('artifacts/crow-v15-impact-grip-probe-20261110/trace.log')
def traces(path):
 out={}
 for s in map(str.split,path.open()):
  if s and s[0]=='trace' and int(s[1])==1 and int(s[3])<40:out[tuple(map(int,s[1:4]))]=s[4:]
 return out
b=traces(base);r=traces(A/'trace.log');parity=len(r)==120 and all(r[k]==b[k] for k in r)
rows=[];xmin=5.02902364730835-.025
for s in map(str.split,(A/'trace.log').open()):
 if not s or s[0]!='fk' or 'rear_toe' not in s[4]:continue
 arm,t=int(s[2]),int(s[3]);x,y,z=map(float,s[5:8]);qx,qy,qz,qw=map(float,s[8:12]);rot00=1-2*(qy*qy+qz*qz);rot01=2*(qx*qy-qz*qw);rot02=2*(qx*qz+qy*qw);extent=abs(rot00)*.018+abs(rot01)*.0025+abs(rot02)*.0092;rows.append({'arm':arm,'step':t,'toe':s[4],'center_x':x,'box_projected_max_x':x+extent,'perch_near_x':xmin,'x_gap_lower_bound':max(0,xmin-x-extent)})
result={'trace_parity_120_samples':parity,'perch_near_x':xmin,'rear_toes':rows,'note':'Positive x-axis projection gap proves separation on that axis; zero would not prove collision.'};(A/'analysis.json').write_text(json.dumps(result,indent=2)+'\n');print('exact sampled GPU trace parity:',parity)
for arm in range(3):
 for t in (0,10,20,30):
  print('arm',arm,'step',t,'rear toe x-axis gaps (m)',[round(x['x_gap_lower_bound'],4) for x in rows if x['arm']==arm and x['step']==t])

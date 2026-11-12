from pathlib import Path
import collections,gzip,json,math,statistics
A=Path('artifacts/crow-v15-final-contact-status-audit-20261112');D=A/'candidate-20261127-rebound-fine';status={};sensors={};finalbody={};endv={};endq={};scene={}
with gzip.open(D/'run.stderr.gz','rt') as f:
 for line in f:
  z=line.split()
  if not z:continue
  if z[0]=='CROW_FINAL_CONTACT_STATUS':
   t,e,active,required,code,bstride,sstride=map(int,z[1:]);status[e,t]=[active,required,code,bstride,sstride]
  elif z[0]=='CROW_SENSOR_VALUE':t,e=map(int,z[1:3]);sensors[e,t]=int(float(z[3]))
  elif z[0]=='CROW_FINAL_BODY':t,e,b=map(int,z[1:4]);finalbody[e,t,b]=list(map(float,z[4:]))
  elif z[0]=='CROW_FINAL_SCENE':t,e=map(int,z[1:3]);scene[e,t]=list(map(float,z[3:]))
  elif z[0] in ('CROW_V_SUBSTEP','CROW_Q_SUBSTEP'):
   t,k,e=map(int,z[1:4]);
   if k==15 and 201<=t<=300:(endv if z[0]=='CROW_V_SUBSTEP' else endq)[e,t]=list(map(float,z[4:]))
assert len(status)==len(sensors)==len(scene)==len(endv)==len(endq)==800,(len(status),len(sensors),len(scene),len(endv),len(endq));assert len(finalbody)==2400
rows=[]
for e in range(8):
 st=[status[e,t] for t in range(201,301)];b=[finalbody[e,t,0] for t in range(201,301)];vv=[endv[e,t] for t in range(201,301)];qq=[endq[e,t] for t in range(201,301)];
 rows.append({'lane':e,'final_active_contacts_distribution':collections.Counter(x[0] for x in st).most_common(),'final_required_constraints_distribution':collections.Counter(x[1] for x in st).most_common(),'codes':sorted(set(x[2] for x in st)),'body_and_scene_strides':sorted(set(tuple(x[3:]) for x in st)),'sensor_values':collections.Counter(sensors[e,t] for t in range(201,300)).most_common(3),'root_final_body_linear_vs_accepted_v_max':max(abs(b[i][3+j]-vv[i][j]) for i in range(100) for j in range(3)),'root_final_body_pos_vs_accepted_q_max':max(abs(b[i][j]-qq[i][j]) for i in range(100) for j in range(3)),'root_final_body_linear_abs_mean':statistics.mean(abs(b[i][3+j]) for i in range(100) for j in range(3))})
out={'rows':rows,'scope':'copied final contact status, body state, scene state and task state after the native command buffer; compared to recorded last-substep generalized q/v','limitations':'The final body buffer need not be materialized at accepted q/v; position and velocity mismatch is evidence about representation/timing, not necessarily a solver defect.'};(A/'status-analysis.json').write_text(json.dumps(out,indent=2)+'\n');[print(x) for x in rows]

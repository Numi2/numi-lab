from pathlib import Path
import gzip,json,math
A=Path('artifacts/crow-v15-grip-body-balance-audit-20261114');P=Path('artifacts/crow-v15-bilateral-stance-recenter-20261114/trace.log.gz');feet={};com={};poses={};events={}
with (A/'output.tsv').open() as f:
 for line in f:
  a=line.split();e,t=map(int,a[1:3]);
  if a[0]=='com':com[e,t]=(float(a[3]),list(map(float,a[4:])))
  if a[0]=='body' and a[3] in ('crow_left_foot','crow_right_foot'):feet[e,t,a[3]]=list(map(float,a[4:]))
with gzip.open(P,'rt') as f:
 for line in f:
  a=line.split()
  if not a:continue
  if a[0]=='pose' and a[3] in ('crow_left_foot','crow_right_foot'):poses[int(a[1]),int(a[2]),a[3]]=list(map(float,a[4:]))
  if a[0]=='event':events[int(a[1]),int(a[2])]=list(map(float,a[3:]))
rows=[];errors=[]
for e in (0,1):
 for t in range(270,331,5):
  m,c=com[e,t];dist=[]
  for name in ('crow_left_foot','crow_right_foot'):
   if (e,t,name) not in poses:errors.append([e,t,name]);continue
   dist.append(math.dist(feet[e,t,name][:3],poses[e,t,name][:3]))
  ev=events[e,t];rows.append({'lane':e,'zero_based_interval':t,'mass_kg':m,'com_xyz_m':c,'com_x_relative_perch_top_edge_m':c[0]-.025,'left_foot_origin_xyz_m':feet[e,t,'crow_left_foot'][:3],'right_foot_origin_xyz_m':feet[e,t,'crow_right_foot'][:3],'max_fk_foot_origin_error_m':max(dist) if dist else None,'last_substep_front_normal_x_impulses_Ns':[ev[4],ev[6]],'last_substep_rear_normal_x_impulses_Ns':[ev[5],ev[7]]})
ok=not errors and len(rows)==26 and all(r['max_fk_foot_origin_error_m'] is not None and r['max_fk_foot_origin_error_m']<1e-6 and abs(r['mass_kg']-.45006)<1e-5 for r in rows)
out={'mapping_check_pass':ok,'missing_poses':errors,'max_fk_foot_origin_error_m':max((r['max_fk_foot_origin_error_m'] or 0) for r in rows),'rows':rows,'scope':'post-hoc accepted endpoint FK; modeled COM relative to the top edge alone is not a support polygon, contact slip, causal proof or biological validation'};(A/'analysis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
